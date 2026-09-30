"""Cross-platform TLS setup (no openssl needed): private CA + server cert for localhost."""
import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

out = Path(__file__).resolve().parent.parent / "certs"
out.mkdir(exist_ok=True)
now = datetime.datetime.now(datetime.timezone.utc)


def name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
ca_cert = (x509.CertificateBuilder().subject_name(name("FedMed Root CA")).issuer_name(name("FedMed Root CA"))
           .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
           .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=365))
           .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
           .sign(ca_key, hashes.SHA256()))

srv_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
srv_cert = (x509.CertificateBuilder().subject_name(name("localhost")).issuer_name(ca_cert.subject)
            .public_key(srv_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .sign(ca_key, hashes.SHA256()))

pem = serialization.Encoding.PEM
(out / "ca.crt").write_bytes(ca_cert.public_bytes(pem))
(out / "server.pem").write_bytes(srv_cert.public_bytes(pem))
(out / "server.key").write_bytes(srv_key.private_bytes(
    pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
print("TLS certs written to", out)