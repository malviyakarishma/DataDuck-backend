import base64
import os
import re
from cryptography.fernet import Fernet, InvalidToken
from app.core.config import settings
from app.core.exceptions import EncryptionError
import logging

logger = logging.getLogger(__name__)


def _get_fernet() -> Fernet:
    """Get Fernet cipher from environment key or generate one."""
    key = settings.ENCRYPTION_KEY
    if not key:
        raise EncryptionError(
            "ENCRYPTION_KEY not set. Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception as e:
        raise EncryptionError(f"Invalid ENCRYPTION_KEY: {e}")


def encrypt_string(plaintext: str) -> str:
    """Encrypt a string and return base64-encoded ciphertext."""
    try:
        fernet = _get_fernet()
        return fernet.encrypt(plaintext.encode()).decode()
    except EncryptionError:
        raise
    except Exception as e:
        raise EncryptionError(f"Encryption failed: {e}")


def decrypt_string(ciphertext: str) -> str:
    """Decrypt a base64-encoded ciphertext string."""
    try:
        fernet = _get_fernet()
        return fernet.decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        raise EncryptionError("Decryption failed — invalid token or wrong key.")
    except EncryptionError:
        raise
    except Exception as e:
        raise EncryptionError(f"Decryption failed: {e}")


def mask_connection_string(connection_string: str) -> str:
    """
    Replace password in a connection string with '****'.
    Handles: scheme://user:password@host/db formats.
    """
    # Pattern: scheme://user:password@host
    pattern = r"(://[^:]+:)([^@]+)(@)"
    masked = re.sub(pattern, r"\1****\3", connection_string)
    return masked


def generate_encryption_key() -> str:
    """Generate a new Fernet key for use as ENCRYPTION_KEY."""
    return Fernet.generate_key().decode()
