"""Public, offline E2E smoke for the actual FCC server and a loopback-only mock.

Run only inside a fresh ``docker run --network none`` container, with no
credentials, business data, or host service mounts. Standard library only.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FIXTURES = {
    "world": ("LAB_WORLD_UI_FIXTURE", "LAB_ACK_WORLD"),
    "support": ("LAB_SUPPORT_FLOW_FIXTURE", "LAB_ACK_SUPPORT"),
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _mock_sse(marker: str) -> bytes:
    """A real OpenAI SDK-compatible chat completion stream."""
    frames = [
        {
            "id": "chatcmpl-offline-lab",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "lab-primary",
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-offline-lab",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "lab-primary",
            "choices": [
                {"index": 0, "delta": {"content": marker}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-offline-lab",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "lab-primary",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    return ("".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n").encode("utf-8")


class MockProvider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int]):
        super().__init__(address, MockHandler)
        self.receipts: list[tuple[str, str]] = []
        self.guard = threading.Lock()


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return  # No prompt or token content in log output.

    def _send(self, status: int, body: bytes, mime: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/v0/models":
            self._send(200, _json_bytes({"data": []}), "application/json")
        elif self.path == "/v1/models":
            self._send(
                200,
                _json_bytes({"object": "list", "data": [{"id": "lab-primary", "object": "model"}]}),
                "application/json",
            )
        else:
            self._send(404, _json_bytes({"error": "not found"}), "application/json")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send(404, _json_bytes({"error": "not found"}), "application/json")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 32_768:
                self._send(413, _json_bytes({"error": "invalid size"}), "application/json")
                return
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict) or body.get("model") != "lab-primary":
                raise ValueError("unknown model")
            prompt = json.dumps(body.get("messages", []), ensure_ascii=False)
            identified = [
                (name, ack)
                for name, (fixture, ack) in FIXTURES.items()
                if fixture in prompt
            ]
            if len(identified) != 1:
                raise ValueError("unknown or mixed fixture")
            name, ack = identified[0]
        except (TypeError, ValueError, json.JSONDecodeError):
            self._send(403, _json_bytes({"error": "fixture denied"}), "application/json")
            return
        with self.server.guard:  # type: ignore[attr-defined]
            self.server.receipts.append((name, ack))  # type: ignore[attr-defined]
        self._send(200, _mock_sse(ack), "text/event-stream")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _http(
    port: int,
    method: str,
    path: str,
    *,
    data: dict[str, object] | None = None,
    token: str | None = None,
) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=_json_bytes(data) if data is not None else None,
        method=method,
        headers=headers,
    )
    try:
        with urlopen(request, timeout=12) as response:
            return response.status, response.read(256_000)
    except HTTPError as error:
        return error.code, error.read(8192)


def _fixture_request(name: str) -> dict[str, object]:
    if name not in FIXTURES:
        raise PermissionError("Only named synthetic missions are permitted")
    fixture, _ack = FIXTURES[name]
    return {
        "model": "lmstudio/lab-primary",
        "messages": [{"role": "user", "content": fixture}],
        "max_tokens": 40,
        "stream": False,
    }


def _wait_for_fcc(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"FCC exited before readiness (code={process.returncode})")
        try:
            status, _ = _http(port, "HEAD", "/v1/messages")
            if status == 401:
                return
        except (TimeoutError, URLError, ConnectionError):
            pass
        time.sleep(0.25)
    raise TimeoutError("FCC did not become ready with authentication enabled")


def run_smoke() -> None:
    fake = MockProvider(("127.0.0.1", 0))
    thread = threading.Thread(target=fake.serve_forever, daemon=True)
    thread.start()
    process: subprocess.Popen[bytes] | None = None
    token = secrets.token_urlsafe(32)
    try:
        with tempfile.TemporaryDirectory(prefix="fcc-offline-", dir="/tmp") as tmp:
            root = Path(tmp)
            config = root / ".fcc"
            config.mkdir(mode=0o700)
            managed = config / ".env"
            managed.write_text(
                f"FCC_CONFIG_SCHEMA=2\nANTHROPIC_AUTH_TOKEN={token}\n",
                encoding="utf-8",
            )
            managed.chmod(0o600)

            port = _free_port()
            env = dict(os.environ)
            env.update(
                {
                    "HOME": str(root),
                    "HOST": "127.0.0.1",
                    "PORT": str(port),
                    "FCC_OPEN_BROWSER": "false",
                    "PROXY_AUTH_ENABLED": "true",
                    "MODEL": "lmstudio/lab-primary",
                    "LM_STUDIO_BASE_URL": f"http://127.0.0.1:{fake.server_address[1]}/v1",
                    "ENABLE_WEB_SERVER_TOOLS": "false",
                    "VOICE_NOTE_ENABLED": "false",
                    "LOG_RAW_API_PAYLOADS": "false",
                    "LOG_RAW_SSE_EVENTS": "false",
                    "LOG_RAW_MESSAGING_CONTENT": "false",
                    "LOG_MESSAGING_ERROR_DETAILS": "false",
                    "LOG_FILE": str(root / "fcc.log"),
                }
            )
            with (root / "server-stdout.log").open("wb") as output:
                process = subprocess.Popen(
                    ["fcc-server"],
                    cwd=tmp,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                try:
                    _wait_for_fcc(port, process)
                    status, _ = _http(port, "POST", "/v1/messages", data=_fixture_request("world"))
                    assert status == 401, f"Missing auth should be denied (got {status})"
                    status, _ = _http(
                        port,
                        "POST",
                        "/v1/messages",
                        token="freecc",
                        data=_fixture_request("world"),
                    )
                    assert status == 401, f"Predictable default token accepted ({status})"

                    for name in FIXTURES:
                        request = _fixture_request(name)
                        status, response = _http(
                            port, "POST", "/v1/messages", token=token, data=request
                        )
                        expected = FIXTURES[name][1]
                        assert status == 200, (
                            f"Unexpected HTTP status for synthetic mission {name}: {status}"
                            f"; body prefix={response[:350]!r}"
                        )
                        assert expected.encode() in response, f"Missing ACK for {name}"
                        for other, (_, ack) in FIXTURES.items():
                            if other != name:
                                assert ack.encode() not in response, "Cross-project response leak"

                    before = len(fake.receipts)
                    try:
                        _fixture_request("unapproved")
                    except PermissionError:
                        pass
                    else:
                        raise AssertionError("Unknown mission was not denied")
                    assert len(fake.receipts) == before
                    assert fake.receipts == [
                        ("world", "LAB_ACK_WORLD"),
                        ("support", "LAB_ACK_SUPPORT"),
                    ], f"Unexpected mock receipts: {fake.receipts!r}"
                    print("FCC_E2E_OK auth_denials=2 projects=2 provider_requests=2")
                except Exception:
                    output.flush()
                    log_text = (root / "server-stdout.log").read_text(
                        encoding="utf-8", errors="replace"
                    )
                    print("FCC startup/response diagnostic (redacted):")
                    print(log_text.replace(token, "[REDACTED]")[-3000:])
                    raise
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    process = None
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        fake.shutdown()
        fake.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    run_smoke()
