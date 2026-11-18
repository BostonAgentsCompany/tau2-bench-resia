"""The provider tests run in the tau2-bench environment. tau imports pyaudio at load only to play
sound on a speaker; on a host without the PortAudio library, a stand-in takes its place for these tests only."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

try:
    import pyaudio  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("pyaudio")
    stub.paInt8, stub.paInt16, stub.paInt32 = 16, 8, 2
    sys.modules["pyaudio"] = stub


@pytest.fixture
def tls_context(tmp_path, monkeypatch):
    """A real TLS server certificate, trusted only for this local test."""
    import ipaddress
    import ssl
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Fullplex local test")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "local-cert.pem", tmp_path / "local-key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    key_path.chmod(0o600)
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context
