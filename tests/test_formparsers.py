from __future__ import annotations

import os
import threading
from collections.abc import Generator
from contextlib import AbstractContextManager, nullcontext as does_not_raise
from io import BytesIO
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, ClassVar
from unittest import mock

import anyio
import pytest

from starlette.applications import Starlette
from starlette.datastructures import Headers, UploadFile
from starlette.formparsers import FormParser, MultiPartException, MultiPartParser, _user_safe_decode
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse
from starlette.routing import Mount
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from tests.types import TestClientFactory


class ForceMultipartDict(dict[Any, Any]):
    def __bool__(self) -> bool:
        return True


# FORCE_MULTIPART is an empty dict that boolean-evaluates as `True`.
FORCE_MULTIPART = ForceMultipartDict()


async def app(scope: Scope, receive: Receive, send: Send) -> None:
    request = Request(scope, receive)
    data = await request.form()
    output: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, UploadFile):
            content = await value.read()
            output[key] = {
                "filename": value.filename,
                "size": value.size,
                "content": content.decode(),
                "content_type": value.content_type,
            }
        else:
            output[key] = value
    await request.close()
    response = JSONResponse(output)
    await response(scope, receive, send)


async def multi_items_app(scope: Scope, receive: Receive, send: Send) -> None:
    request = Request(scope, receive)
    data = await request.form()
    output: dict[str, list[Any]] = {}
    for key, value in data.multi_items():
        if key not in output:
            output[key] = []
        if isinstance(value, UploadFile):
            content = await value.read()
            output[key].append(
                {
                    "filename": value.filename,
                    "size": value.size,
                    "content": content.decode(),
                    "content_type": value.content_type,
                }
            )
        else:
            output[key].append(value)
    await request.close()
    response = JSONResponse(output)
    await response(scope, receive, send)


async def app_with_headers(scope: Scope, receive: Receive, send: Send) -> None:
    request = Request(scope, receive)
    data = await request.form()
    output: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, UploadFile):
            content = await value.read()
            output[key] = {
                "filename": value.filename,
                "size": value.size,
                "content": content.decode(),
                "content_type": value.content_type,
                "headers": list(value.headers.items()),
            }
        else:
            output[key] = value
    await request.close()
    response = JSONResponse(output)
    await response(scope, receive, send)


async def app_read_body(scope: Scope, receive: Receive, send: Send) -> None:
    request = Request(scope, receive)
    # Read bytes, to force request.stream() to return the already parsed body
    await request.body()
    data = await request.form()
    output = {}
    for key, value in data.items():
        output[key] = value
    await request.close()
    response = JSONResponse(output)
    await response(scope, receive, send)


async def app_monitor_thread(scope: Scope, receive: Receive, send: Send) -> None:
    """Helper app to monitor what thread the app was called on.

    This can later be used to validate thread/event loop operations.
    """
    request = Request(scope, receive)

    # Make sure we parse the form
    await request.form()
    await request.close()

    # Send back the current thread id
    response = JSONResponse({"thread_ident": threading.current_thread().ident})
    await response(scope, receive, send)


def make_app_max_parts(
    max_files: int = 1000,
    max_fields: int = 1000,
    max_part_size: int = 1024 * 1024,
    max_file_size: int | None = None,
    max_total_size: int | None = None,
) -> ASGIApp:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        data = await request.form(
            max_files=max_files,
            max_fields=max_fields,
            max_part_size=max_part_size,
            max_file_size=max_file_size,
            max_total_size=max_total_size,
        )
        output: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, UploadFile):
                content = await value.read()
                output[key] = {
                    "filename": value.filename,
                    "size": value.size,
                    "content": content.decode(),
                    "content_type": value.content_type,
                }
            else:
                output[key] = value
        await request.close()
        response = JSONResponse(output)
        await response(scope, receive, send)

    return app


def test_multipart_request_data(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app)
    response = client.post("/", data={"some": "data"}, files=FORCE_MULTIPART)
    assert response.json() == {"some": "data"}


def test_multipart_request_files(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    path = os.path.join(tmpdir, "test.txt")
    with open(path, "wb") as file:
        file.write(b"<file content>")

    client = test_client_factory(app)
    with open(path, "rb") as f:
        response = client.post("/", files={"test": f})
        assert response.json() == {
            "test": {
                "filename": "test.txt",
                "size": 14,
                "content": "<file content>",
                "content_type": "text/plain",
            }
        }


def test_multipart_request_files_with_content_type(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    path = os.path.join(tmpdir, "test.txt")
    with open(path, "wb") as file:
        file.write(b"<file content>")

    client = test_client_factory(app)
    with open(path, "rb") as f:
        response = client.post("/", files={"test": ("test.txt", f, "text/plain")})
        assert response.json() == {
            "test": {
                "filename": "test.txt",
                "size": 14,
                "content": "<file content>",
                "content_type": "text/plain",
            }
        }


def test_multipart_request_multiple_files(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    path1 = os.path.join(tmpdir, "test1.txt")
    with open(path1, "wb") as file:
        file.write(b"<file1 content>")

    path2 = os.path.join(tmpdir, "test2.txt")
    with open(path2, "wb") as file:
        file.write(b"<file2 content>")

    client = test_client_factory(app)
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        response = client.post("/", files={"test1": f1, "test2": ("test2.txt", f2, "text/plain")})
        assert response.json() == {
            "test1": {
                "filename": "test1.txt",
                "size": 15,
                "content": "<file1 content>",
                "content_type": "text/plain",
            },
            "test2": {
                "filename": "test2.txt",
                "size": 15,
                "content": "<file2 content>",
                "content_type": "text/plain",
            },
        }


def test_multipart_request_multiple_files_with_headers(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    path1 = os.path.join(tmpdir, "test1.txt")
    with open(path1, "wb") as file:
        file.write(b"<file1 content>")

    path2 = os.path.join(tmpdir, "test2.txt")
    with open(path2, "wb") as file:
        file.write(b"<file2 content>")

    client = test_client_factory(app_with_headers)
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        response = client.post(
            "/",
            files=[
                ("test1", (None, f1)),
                ("test2", ("test2.txt", f2, "text/plain", {"x-custom": "f2"})),
            ],
        )
        assert response.json() == {
            "test1": "<file1 content>",
            "test2": {
                "filename": "test2.txt",
                "size": 15,
                "content": "<file2 content>",
                "content_type": "text/plain",
                "headers": [
                    [
                        "content-disposition",
                        'form-data; name="test2"; filename="test2.txt"',
                    ],
                    ["x-custom", "f2"],
                    ["content-type", "text/plain"],
                ],
            },
        }


def test_multi_items(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    path1 = os.path.join(tmpdir, "test1.txt")
    with open(path1, "wb") as file:
        file.write(b"<file1 content>")

    path2 = os.path.join(tmpdir, "test2.txt")
    with open(path2, "wb") as file:
        file.write(b"<file2 content>")

    client = test_client_factory(multi_items_app)
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        response = client.post(
            "/",
            data={"test1": "abc"},
            files=[("test1", f1), ("test1", ("test2.txt", f2, "text/plain"))],
        )
        assert response.json() == {
            "test1": [
                "abc",
                {
                    "filename": "test1.txt",
                    "size": 15,
                    "content": "<file1 content>",
                    "content_type": "text/plain",
                },
                {
                    "filename": "test2.txt",
                    "size": 15,
                    "content": "<file2 content>",
                    "content_type": "text/plain",
                },
            ]
        }


def test_multipart_request_mixed_files_and_data(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app)
    response = client.post(
        "/",
        data=(
            # data
            b"--a7f7ac8d4e2e437c877bb7b8d7cc549c\r\n"  # type: ignore
            b'Content-Disposition: form-data; name="field0"\r\n\r\n'
            b"value0\r\n"
            # file
            b"--a7f7ac8d4e2e437c877bb7b8d7cc549c\r\n"
            b'Content-Disposition: form-data; name="file"; filename="file.txt"\r\n'
            b"Content-Type: text/plain\r\n\r\n"
            b"<file content>\r\n"
            # data
            b"--a7f7ac8d4e2e437c877bb7b8d7cc549c\r\n"
            b'Content-Disposition: form-data; name="field1"\r\n\r\n'
            b"value1\r\n"
            b"--a7f7ac8d4e2e437c877bb7b8d7cc549c--\r\n"
        ),
        headers={"Content-Type": ("multipart/form-data; boundary=a7f7ac8d4e2e437c877bb7b8d7cc549c")},
    )
    assert response.json() == {
        "file": {
            "filename": "file.txt",
            "size": 14,
            "content": "<file content>",
            "content_type": "text/plain",
        },
        "field0": "value0",
        "field1": "value1",
    }


class ThreadTrackingSpooledTemporaryFile(SpooledTemporaryFile[bytes]):
    """Helper class to track which threads performed the rollover operation.

    This is not threadsafe/multi-test safe.
    """

    rollover_threads: ClassVar[set[int | None]] = set()

    def rollover(self) -> None:
        ThreadTrackingSpooledTemporaryFile.rollover_threads.add(threading.current_thread().ident)
        super().rollover()


@pytest.fixture
def mock_spooled_temporary_file() -> Generator[None]:
    try:
        with mock.patch("starlette.formparsers.SpooledTemporaryFile", ThreadTrackingSpooledTemporaryFile):
            yield
    finally:
        ThreadTrackingSpooledTemporaryFile.rollover_threads.clear()


def test_multipart_request_large_file_rollover_in_background_thread(
    mock_spooled_temporary_file: None, test_client_factory: TestClientFactory
) -> None:
    """Test that Spooled file rollovers happen in background threads."""
    data = BytesIO(b" " * (MultiPartParser.spool_max_size + 1))

    client = test_client_factory(app_monitor_thread)
    response = client.post("/", files=[("test_large", data)])
    assert response.status_code == 200

    # Parse the event thread id from the API response and ensure we have one
    app_thread_ident = response.json().get("thread_ident")
    assert app_thread_ident is not None

    # Ensure the app thread was not the same as the rollover one and that a rollover thread exists
    assert app_thread_ident not in ThreadTrackingSpooledTemporaryFile.rollover_threads
    assert len(ThreadTrackingSpooledTemporaryFile.rollover_threads) == 1


def test_multipart_request_with_charset_for_filename(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app)
    response = client.post(
        "/",
        data=(
            # file
            b"--a7f7ac8d4e2e437c877bb7b8d7cc549c\r\n"  # type: ignore
            b'Content-Disposition: form-data; name="file"; filename="\xe6\x96\x87\xe6\x9b\xb8.txt"\r\n'
            b"Content-Type: text/plain\r\n\r\n"
            b"<file content>\r\n"
            b"--a7f7ac8d4e2e437c877bb7b8d7cc549c--\r\n"
        ),
        headers={"Content-Type": ("multipart/form-data; charset=utf-8; boundary=a7f7ac8d4e2e437c877bb7b8d7cc549c")},
    )
    assert response.json() == {
        "file": {
            "filename": "文書.txt",
            "size": 14,
            "content": "<file content>",
            "content_type": "text/plain",
        }
    }


def test_multipart_request_without_charset_for_filename(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app)
    response = client.post(
        "/",
        data=(
            # file
            b"--a7f7ac8d4e2e437c877bb7b8d7cc549c\r\n"  # type: ignore
            b'Content-Disposition: form-data; name="file"; filename="\xe7\x94\xbb\xe5\x83\x8f.jpg"\r\n'
            b"Content-Type: image/jpeg\r\n\r\n"
            b"<file content>\r\n"
            b"--a7f7ac8d4e2e437c877bb7b8d7cc549c--\r\n"
        ),
        headers={"Content-Type": ("multipart/form-data; boundary=a7f7ac8d4e2e437c877bb7b8d7cc549c")},
    )
    assert response.json() == {
        "file": {
            "filename": "画像.jpg",
            "size": 14,
            "content": "<file content>",
            "content_type": "image/jpeg",
        }
    }


def test_multipart_request_with_encoded_value(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app)
    response = client.post(
        "/",
        data=(
            b"--20b303e711c4ab8c443184ac833ab00f\r\n"  # type: ignore
            b"Content-Disposition: form-data; "
            b'name="value"\r\n\r\n'
            b"Transf\xc3\xa9rer\r\n"
            b"--20b303e711c4ab8c443184ac833ab00f--\r\n"
        ),
        headers={"Content-Type": ("multipart/form-data; charset=utf-8; boundary=20b303e711c4ab8c443184ac833ab00f")},
    )
    assert response.json() == {"value": "Transférer"}


def test_urlencoded_request_data(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app)
    response = client.post("/", data={"some": "data"})
    assert response.json() == {"some": "data"}


def test_no_request_data(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app)
    response = client.post("/")
    assert response.json() == {}


def test_urlencoded_percent_encoding(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app)
    response = client.post("/", data={"some": "da ta"})
    assert response.json() == {"some": "da ta"}


def test_urlencoded_percent_encoding_keys(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app)
    response = client.post("/", data={"so me": "data"})
    assert response.json() == {"so me": "data"}


def test_urlencoded_multi_field_app_reads_body(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app_read_body)
    response = client.post("/", data={"some": "data", "second": "key pair"})
    assert response.json() == {"some": "data", "second": "key pair"}


def test_multipart_multi_field_app_reads_body(tmpdir: Path, test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(app_read_body)
    response = client.post("/", data={"some": "data", "second": "key pair"}, files=FORCE_MULTIPART)
    assert response.json() == {"some": "data", "second": "key pair"}


def test_user_safe_decode_helper() -> None:
    result = _user_safe_decode(b"\xc4\x99\xc5\xbc\xc4\x87", "utf-8")
    assert result == "ężć"


def test_user_safe_decode_ignores_wrong_charset() -> None:
    result = _user_safe_decode(b"abc", "latin-8")
    assert result == "abc"


@pytest.mark.parametrize(
    "app,expectation",
    [
        (app, pytest.raises(MultiPartException)),
        (Starlette(routes=[Mount("/", app=app)]), does_not_raise()),
    ],
)
def test_missing_boundary_parameter(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    with expectation:
        res = client.post(
            "/",
            data=(
                # file
                b'Content-Disposition: form-data; name="file"; filename="\xe6\x96\x87\xe6\x9b\xb8.txt"\r\n'  # type: ignore
                b"Content-Type: text/plain\r\n\r\n"
                b"<file content>\r\n"
            ),
            headers={"Content-Type": "multipart/form-data; charset=utf-8"},
        )
        assert res.status_code == 400
        assert res.text == "Missing boundary in multipart."


@pytest.mark.parametrize(
    "app,expectation",
    [
        (app, pytest.raises(MultiPartException)),
        (Starlette(routes=[Mount("/", app=app)]), does_not_raise()),
    ],
)
def test_missing_name_parameter_on_content_disposition(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    with expectation:
        res = client.post(
            "/",
            data=(
                # data
                b"--a7f7ac8d4e2e437c877bb7b8d7cc549c\r\n"  # type: ignore
                b'Content-Disposition: form-data; ="field0"\r\n\r\n'
                b"value0\r\n"
            ),
            headers={"Content-Type": ("multipart/form-data; boundary=a7f7ac8d4e2e437c877bb7b8d7cc549c")},
        )
        assert res.status_code == 400
        assert res.text == 'The Content-Disposition header field "name" must be provided.'


@pytest.mark.parametrize(
    "app,expectation",
    [
        (app, pytest.raises(MultiPartException)),
        (Starlette(routes=[Mount("/", app=app)]), does_not_raise()),
    ],
)
def test_too_many_fields_raise(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    fields = []
    for i in range(1001):
        fields.append(f'--B\r\nContent-Disposition: form-data; name="N{i}";\r\n\r\n\r\n')
    data = "".join(fields).encode("utf-8")
    with expectation:
        res = client.post(
            "/",
            data=data,  # type: ignore
            headers={"Content-Type": ("multipart/form-data; boundary=B")},
        )
        assert res.status_code == 400
        assert res.text == "Too many fields. Maximum number of fields is 1000."


@pytest.mark.parametrize(
    "app,expectation",
    [
        (app, pytest.raises(MultiPartException)),
        (Starlette(routes=[Mount("/", app=app)]), does_not_raise()),
    ],
)
def test_too_many_files_raise(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    fields = []
    for i in range(1001):
        fields.append(f'--B\r\nContent-Disposition: form-data; name="N{i}"; filename="F{i}";\r\n\r\n\r\n')
    data = "".join(fields).encode("utf-8")
    with expectation:
        res = client.post(
            "/",
            data=data,  # type: ignore
            headers={"Content-Type": ("multipart/form-data; boundary=B")},
        )
        assert res.status_code == 400
        assert res.text == "Too many files. Maximum number of files is 1000."


@pytest.mark.parametrize(
    "app,expectation",
    [
        (app, pytest.raises(MultiPartException)),
        (Starlette(routes=[Mount("/", app=app)]), does_not_raise()),
    ],
)
def test_too_many_files_single_field_raise(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    fields = []
    for i in range(1001):
        # This uses the same field name "N" for all files, equivalent to a
        # multifile upload form field
        fields.append(f'--B\r\nContent-Disposition: form-data; name="N"; filename="F{i}";\r\n\r\n\r\n')
    data = "".join(fields).encode("utf-8")
    with expectation:
        res = client.post(
            "/",
            data=data,  # type: ignore
            headers={"Content-Type": ("multipart/form-data; boundary=B")},
        )
        assert res.status_code == 400
        assert res.text == "Too many files. Maximum number of files is 1000."


@pytest.mark.parametrize(
    "app,expectation",
    [
        (app, pytest.raises(MultiPartException)),
        (Starlette(routes=[Mount("/", app=app)]), does_not_raise()),
    ],
)
def test_too_many_files_and_fields_raise(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    fields = []
    for i in range(1001):
        fields.append(f'--B\r\nContent-Disposition: form-data; name="F{i}"; filename="F{i}";\r\n\r\n\r\n')
        fields.append(f'--B\r\nContent-Disposition: form-data; name="N{i}";\r\n\r\n\r\n')
    data = "".join(fields).encode("utf-8")
    with expectation:
        res = client.post(
            "/",
            data=data,  # type: ignore
            headers={"Content-Type": ("multipart/form-data; boundary=B")},
        )
        assert res.status_code == 400
        assert res.text == "Too many files. Maximum number of files is 1000."


@pytest.mark.parametrize(
    "app,expectation",
    [
        (make_app_max_parts(max_fields=1), pytest.raises(MultiPartException)),
        (
            Starlette(routes=[Mount("/", app=make_app_max_parts(max_fields=1))]),
            does_not_raise(),
        ),
    ],
)
def test_max_fields_is_customizable_low_raises(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    fields = []
    for i in range(2):
        fields.append(f'--B\r\nContent-Disposition: form-data; name="N{i}";\r\n\r\n\r\n')
    data = "".join(fields).encode("utf-8")
    with expectation:
        res = client.post(
            "/",
            data=data,  # type: ignore
            headers={"Content-Type": ("multipart/form-data; boundary=B")},
        )
        assert res.status_code == 400
        assert res.text == "Too many fields. Maximum number of fields is 1."


@pytest.mark.parametrize(
    "app,expectation",
    [
        (make_app_max_parts(max_files=1), pytest.raises(MultiPartException)),
        (
            Starlette(routes=[Mount("/", app=make_app_max_parts(max_files=1))]),
            does_not_raise(),
        ),
    ],
)
def test_max_files_is_customizable_low_raises(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    fields = []
    for i in range(2):
        fields.append(f'--B\r\nContent-Disposition: form-data; name="F{i}"; filename="F{i}";\r\n\r\n\r\n')
    data = "".join(fields).encode("utf-8")
    with expectation:
        res = client.post(
            "/",
            data=data,  # type: ignore
            headers={"Content-Type": ("multipart/form-data; boundary=B")},
        )
        assert res.status_code == 400
        assert res.text == "Too many files. Maximum number of files is 1."


def test_max_fields_is_customizable_high(test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(make_app_max_parts(max_fields=2000, max_files=2000))
    fields = []
    for i in range(2000):
        fields.append(f'--B\r\nContent-Disposition: form-data; name="N{i}";\r\n\r\n\r\n')
        fields.append(f'--B\r\nContent-Disposition: form-data; name="F{i}"; filename="F{i}";\r\n\r\n\r\n')
    data = "".join(fields).encode("utf-8")
    data += b"--B--\r\n"
    res = client.post(
        "/",
        data=data,  # type: ignore
        headers={"Content-Type": ("multipart/form-data; boundary=B")},
    )
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["N1999"] == ""
    assert res_data["F1999"] == {
        "filename": "F1999",
        "size": 0,
        "content": "",
        "content_type": None,
    }


@pytest.mark.parametrize(
    "app,expectation",
    [
        (app, pytest.raises(MultiPartException)),
        (Starlette(routes=[Mount("/", app=app)]), does_not_raise()),
    ],
)
def test_max_part_size_exceeds_limit(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    boundary = "------------------------4K1ON9fZkj9uCUmqLHRbbR"

    multipart_data = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="small"\r\n\r\n'
        "small content\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="large"\r\n\r\n'
        + ("x" * 1024 * 1024 + "x")  # 1MB + 1 byte of data
        + "\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Transfer-Encoding": "chunked",
    }

    with expectation:
        response = client.post("/", data=multipart_data, headers=headers)  # type: ignore
        assert response.status_code == 400
        assert response.text == "Part exceeded maximum size of 1024KB."


@pytest.mark.parametrize(
    "app,expectation",
    [
        (make_app_max_parts(max_part_size=1024 * 10), pytest.raises(MultiPartException)),
        (
            Starlette(routes=[Mount("/", app=make_app_max_parts(max_part_size=1024 * 10))]),
            does_not_raise(),
        ),
    ],
)
def test_max_part_size_exceeds_custom_limit(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    boundary = "------------------------4K1ON9fZkj9uCUmqLHRbbR"

    multipart_data = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="small"\r\n\r\n'
        "small content\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="large"\r\n\r\n'
        + ("x" * 1024 * 10 + "x")  # 1MB + 1 byte of data
        + "\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Transfer-Encoding": "chunked",
    }

    with expectation:
        response = client.post("/", content=multipart_data, headers=headers)
        assert response.status_code == 400
        assert response.text == "Part exceeded maximum size of 10KB."


def test_multipart_closes_tempfile_on_oserror(
    test_client_factory: TestClientFactory,
) -> None:
    """Temporary files must be closed when an OSError (e.g. disk full) is raised during parsing."""
    close_called = False

    class FailingSpooledTemporaryFile(SpooledTemporaryFile[bytes]):
        def write(self, s: Any) -> int:
            raise OSError("disk full")

        def close(self) -> None:
            nonlocal close_called
            close_called = True
            super().close()

    async def error_app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        with mock.patch("starlette.formparsers.SpooledTemporaryFile", FailingSpooledTemporaryFile):
            await request.form()

    client = test_client_factory(error_app)
    boundary = "a7f7ac8d4e2e437c877bb7b8d7cc549c"
    content = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
        f"file content\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    with pytest.raises(OSError, match="disk full"):
        client.post("/", content=content, headers=headers)

    assert close_called


def make_multipart_body(boundary: str, parts: list[tuple[str, str | None, bytes]]) -> bytes:
    """Build a multipart body from (name, filename, content) parts."""
    data = b""
    for name, filename, content in parts:
        data += f"--{boundary}\r\n".encode()
        if filename is None:
            data += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        else:
            data += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n\r\n'.encode()
        data += content + b"\r\n"
    data += f"--{boundary}--\r\n".encode()
    return data


@pytest.mark.parametrize(
    "app,expectation",
    [
        (make_app_max_parts(max_file_size=10), pytest.raises(MultiPartException)),
        (
            Starlette(routes=[Mount("/", app=make_app_max_parts(max_file_size=10))]),
            does_not_raise(),
        ),
    ],
)
def test_max_file_size_exceeds_limit(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    boundary = "boundary"
    data = make_multipart_body(boundary, [("file", "test.txt", b"x" * 11)])
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    with expectation:
        response = client.post("/", content=data, headers=headers)
        assert response.status_code == 413
        assert response.text == "File exceeded the maximum size of 10 bytes."


def test_max_file_size_at_limit_passes(test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(make_app_max_parts(max_file_size=10))
    boundary = "boundary"
    data = make_multipart_body(boundary, [("file", "test.txt", b"x" * 10)])
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    response = client.post("/", content=data, headers=headers)
    assert response.status_code == 200
    assert response.json()["file"] == {
        "filename": "test.txt",
        "size": 10,
        "content": "x" * 10,
        "content_type": None,
    }


def test_max_file_size_zero_allows_only_empty_file(test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(make_app_max_parts(max_file_size=0))
    boundary = "boundary"
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    data = make_multipart_body(boundary, [("file", "empty.txt", b"")])
    response = client.post("/", content=data, headers=headers)
    assert response.status_code == 200
    assert response.json()["file"]["size"] == 0

    data = make_multipart_body(boundary, [("file", "test.txt", b"x")])
    with pytest.raises(MultiPartException, match="File exceeded the maximum size of 0 bytes."):
        client.post("/", content=data, headers=headers)


@pytest.mark.parametrize(
    "app,expectation",
    [
        (make_app_max_parts(max_total_size=20), pytest.raises(MultiPartException)),
        (
            Starlette(routes=[Mount("/", app=make_app_max_parts(max_total_size=20))]),
            does_not_raise(),
        ),
    ],
)
def test_max_total_size_counts_files_and_fields(
    app: ASGIApp,
    expectation: AbstractContextManager[Exception],
    test_client_factory: TestClientFactory,
) -> None:
    client = test_client_factory(app)
    boundary = "boundary"
    # 8 bytes of field content + 13 bytes of file content = 21 bytes in total.
    data = make_multipart_body(
        boundary,
        [("field", None, b"x" * 8), ("file", "test.txt", b"x" * 13)],
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    with expectation:
        response = client.post("/", content=data, headers=headers)
        assert response.status_code == 413
        assert response.text == "Total upload size exceeded the maximum of 20 bytes."


def test_max_total_size_at_limit_passes(test_client_factory: TestClientFactory) -> None:
    client = test_client_factory(make_app_max_parts(max_total_size=21))
    boundary = "boundary"
    data = make_multipart_body(
        boundary,
        [("field", None, b"x" * 8), ("file", "test.txt", b"x" * 13)],
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    response = client.post("/", content=data, headers=headers)
    assert response.status_code == 200
    assert response.json()["field"] == "x" * 8
    assert response.json()["file"]["size"] == 13


def test_max_total_size_urlencoded(test_client_factory: TestClientFactory) -> None:
    async def urlencoded_app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        data = await request.form(max_total_size=10)
        await request.close()
        response = JSONResponse(dict(data))
        await response(scope, receive, send)

    client = test_client_factory(urlencoded_app)
    response = client.post("/", data={"key": "x" * 10})  # 10 bytes of content
    assert response.status_code == 200
    assert response.json() == {"key": "x" * 10}

    with pytest.raises(MultiPartException, match="Total upload size exceeded the maximum of 10 bytes."):
        client.post("/", data={"key": "x" * 11})  # 11 bytes of content


@pytest.mark.parametrize("limit_value", [-1, 1.5, "10", True])
def test_invalid_size_limits_raise_early(limit_value: Any) -> None:
    headers = Headers({"Content-Type": "multipart/form-data; boundary=boundary"})

    async def stream() -> Any:
        yield b""  # pragma: no cover

    for limit_name in ("max_file_size", "max_total_size"):
        with pytest.raises(ValueError, match=f"{limit_name} must be a non-negative integer or None"):
            MultiPartParser(headers, stream(), **{limit_name: limit_value})

    with pytest.raises(ValueError, match="max_total_size must be a non-negative integer or None"):
        FormParser(headers, stream(), max_total_size=limit_value)


def test_invalid_size_limits_raise_early_on_request(test_client_factory: TestClientFactory) -> None:
    async def invalid_app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        await request.form(max_file_size=-1)

    client = test_client_factory(invalid_app)
    boundary = "boundary"
    data = make_multipart_body(boundary, [("file", "test.txt", b"content")])
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    with pytest.raises(ValueError, match="max_file_size must be a non-negative integer or None"):
        client.post("/", content=data, headers=headers)


def test_multipart_closes_tempfiles_on_size_limit_error(test_client_factory: TestClientFactory) -> None:
    """All opened temporary files must be closed when a size limit is exceeded mid-upload."""
    closed_files: list[SpooledTemporaryFile[bytes]] = []

    class TrackingSpooledTemporaryFile(SpooledTemporaryFile[bytes]):
        def close(self) -> None:
            closed_files.append(self)
            super().close()

    async def error_app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        with mock.patch("starlette.formparsers.SpooledTemporaryFile", TrackingSpooledTemporaryFile):
            await request.form(max_file_size=5)

    client = test_client_factory(error_app)
    boundary = "boundary"
    # The first file is fully uploaded, the second one exceeds the limit.
    data = make_multipart_body(
        boundary,
        [("file1", "one.txt", b"12345"), ("file2", "two.txt", b"123456")],
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    with pytest.raises(MultiPartException, match="File exceeded the maximum size of 5 bytes."):
        client.post("/", content=data, headers=headers)

    assert len(closed_files) == 2
    for file in closed_files:
        assert file.closed


def test_multipart_closes_tempfiles_on_client_disconnect(test_client_factory: TestClientFactory) -> None:
    """All opened temporary files must be closed when the client disconnects mid-upload."""
    closed_files: list[SpooledTemporaryFile[bytes]] = []

    class TrackingSpooledTemporaryFile(SpooledTemporaryFile[bytes]):
        def close(self) -> None:
            closed_files.append(self)
            super().close()

    async def disconnect_app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        with mock.patch("starlette.formparsers.SpooledTemporaryFile", TrackingSpooledTemporaryFile):
            await request.form()

    boundary = "boundary"
    first_part = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.txt"\r\n\r\npartial content'
    ).encode()

    messages: list[Message] = [
        {"type": "http.request", "body": first_part, "more_body": True},
        {"type": "http.disconnect"},
    ]

    async def receive() -> Message:
        return messages.pop(0)

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())],
    }

    async def send(message: Message) -> None:  # pragma: no cover
        pass

    with pytest.raises(ClientDisconnect):
        anyio.run(disconnect_app, scope, receive, send)

    assert len(closed_files) == 1
    assert closed_files[0].closed
