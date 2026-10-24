"""Exercise Geoazur TLS compatibility without depending on a remote server."""

import http.server
import importlib
import shutil
import socket
import ssl
import subprocess
import threading
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def geoazur(monkeypatch):
    """Import the standalone documentation helper."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "docs"))
    return importlib.import_module("geoazur")


@pytest.fixture
def tls_server(tmp_path, monkeypatch, request):
    """Serve a locally trusted Geoazur certificate with RSA-only TLS 1.2."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl needed to generate the local TLS certificate")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    certificate_host = getattr(request, "param", "www.geoazur.fr")
    subprocess.run(  # noqa: S603 - fixed arguments and a discovered local tool
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            f"/CN={certificate_host}",
            "-addext",
            f"subjectAltName=DNS:{certificate_host}",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        timeout=30,
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/missing":
                self.send_error(404)
            elif self.path in ("/downgrade", "/other-host"):
                self.send_response(302)
                target = (
                    "http://www.geoazur.fr/ok"
                    if self.path == "/downgrade"
                    else "https://other.invalid/ok"
                )
                self.send_header("Location", target)
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"model bytes")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    context.set_ciphers("AES256-GCM-SHA384:AES128-GCM-SHA256")
    context.load_cert_chain(cert, key)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    connect = socket.create_connection

    def local_connection(address, *args, **kwargs):
        assert address[0] in ("www.geoazur.fr", "other.invalid")
        return connect(server.server_address, *args, **kwargs)

    default_context = ssl.create_default_context

    def modern_context():
        ctx = default_context()
        # Some distributions retain RSA in their defaults; reproduce CPython's
        # forward-secret default regardless of the test machine's distribution.
        ctx.set_ciphers("ECDHE+AESGCM")
        return ctx

    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setattr(socket, "create_connection", local_connection)
    monkeypatch.setattr(ssl, "create_default_context", modern_context)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_download_with_verified_rsa_tls(geoazur, tls_server, tmp_path):
    """The actual HTTPS handshake succeeds and the download retains its bytes."""
    target = tmp_path / "model.bin"
    geoazur.download("https://www.geoazur.fr/ok", target)
    assert target.read_bytes() == b"model bytes"
    ctx = geoazur.tls_context(geoazur.MARMOUSI_PAGE)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2


def test_untrusted_certificate_fails(geoazur, tls_server, monkeypatch, tmp_path):
    """Compatibility must not bypass certificate validation."""
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "nonexistent.pem"))
    with pytest.raises(urllib.error.URLError, match="CERTIFICATE_VERIFY_FAILED"):
        geoazur.open_https("https://www.geoazur.fr/ok")


@pytest.mark.parametrize("tls_server", ["wrong.invalid"], indirect=True)
def test_wrong_hostname_fails(geoazur, tls_server):
    """A trusted certificate for the wrong hostname must still be rejected."""
    with pytest.raises(urllib.error.URLError, match="CERTIFICATE_VERIFY_FAILED"):
        geoazur.open_https("https://www.geoazur.fr/ok")


def test_model_checksum_remains_required(geoazur, tls_server, monkeypatch, tmp_path):
    """An HTTPS-successful response with incorrect model bytes is rejected."""
    model = importlib.import_module("get_marmousi")
    monkeypatch.setattr(model, "CANONICAL_VP_URL", "https://www.geoazur.fr/ok")
    target = tmp_path / "vp.bin"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        model.download_vp(target)
    assert not target.exists()
    assert not target.with_suffix(".bin.part").exists()


@pytest.mark.parametrize(
    ("path", "exception", "message"),
    [
        ("downgrade", ValueError, "Refusing non-HTTPS redirect"),
        ("other-host", urllib.error.URLError, "HANDSHAKE_FAILURE"),
        ("missing", urllib.error.HTTPError, "HTTP Error 404"),
    ],
)
def test_redirect_and_http_errors_fail(geoazur, tls_server, path, exception, message):
    """Reject HTTP, keep other hosts' TLS defaults, and report missing pages."""
    with pytest.raises(exception, match=message):
        geoazur.open_https(f"https://www.geoazur.fr/{path}")


@pytest.mark.parametrize("exists", [True, False])
def test_sphinx_requires_successful_check(geoazur, tls_server, monkeypatch, exists):
    """Only a successfully checked exact URL can bypass generic linkcheck."""
    pytest.importorskip("sphinx")
    from sphinx.errors import ExtensionError

    url = "https://www.geoazur.fr/" + ("ok" if exists else "missing")
    monkeypatch.setattr(geoazur, "MARMOUSI_PAGE", url)
    app = SimpleNamespace(
        builder=SimpleNamespace(name="linkcheck"),
        config=SimpleNamespace(linkcheck_ignore=[]),
    )
    if exists:
        geoazur.check_marmousi_link(app)
        assert len(app.config.linkcheck_ignore) == 1
    else:
        with pytest.raises(ExtensionError, match="Geoazur link check failed"):
            geoazur.check_marmousi_link(app)
        assert app.config.linkcheck_ignore == []
