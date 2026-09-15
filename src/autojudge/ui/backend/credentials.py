"""Cross-platform OS credential storage. Never fall back to plaintext."""
import base64
import ctypes
import hashlib
import os
import sys
from ctypes import wintypes

SERVICE = 'AutoJudge'


def master_key():
    """Fernet key derived from AUTOJUDGE_CREDENTIALS_KEY; None when unset."""
    secret = os.environ.get('AUTOJUDGE_CREDENTIALS_KEY', '').strip()
    if not secret:
        return None
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode('utf-8')).digest())


class Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]


def transform(value: bytes, decrypt: bool = False) -> bytes:
    if sys.platform != 'win32':
        raise RuntimeError('Encrypted credential storage requires Windows DPAPI')
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    source = Blob(len(value), buffer)
    target = Blob()
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    # CRYPTPROTECT_UI_FORBIDDEN; no machine-wide flag: bound to current Windows user.
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise RuntimeError('Windows credential protection failed')
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(ctypes.cast(target.data, ctypes.c_void_p))


def encrypt(secret: str) -> str:
    return base64.b64encode(transform(secret.encode('utf-8'))).decode('ascii')


def decrypt(ciphertext: str) -> str:
    return transform(base64.b64decode(ciphertext), decrypt=True).decode('utf-8')


def storage_info():
    if sys.platform == 'win32':
        return {'name':'Windows DPAPI','available':True,'persistent':True}
    if master_key() is not None:
        return {'name':'Encrypted container storage','available':True,'persistent':True}
    name = 'macOS Keychain' if sys.platform == 'darwin' else 'Linux Secret Service'
    try:
        import keyring
        backend = keyring.get_keyring()
        available = float(getattr(backend, 'priority', 0)) > 0
    except Exception:
        available = False
    return {'name':name if available else 'Environment only','available':available,'persistent':available}


def store(name: str, secret: str):
    if sys.platform == 'win32':
        return {'ciphertext':encrypt(secret),'storage':'dpapi'}
    if master_key() is not None:
        return {'ciphertext':ferni_encrypt(secret),'storage':'master-key'}
    if not storage_info()['available']:
        raise RuntimeError('Persistent OS credential storage is unavailable')
    try:
        import keyring
        keyring.set_password(SERVICE, name, secret)
    except Exception as exc:
        raise RuntimeError('Persistent OS credential storage is unavailable') from exc
    return {'keyring':True,'storage':'keyring'}


def load(name: str, payload: dict) -> str:
    ciphertext=payload.get('ciphertext','')
    if ciphertext:
        if payload.get('storage')=='master-key':
            return ferni_decrypt(ciphertext)
        return decrypt(ciphertext)
    if not payload.get('keyring'):
        return ''
    try:
        import keyring
        return keyring.get_password(SERVICE,name) or ''
    except Exception as exc:
        raise RuntimeError('Persistent OS credential storage is unavailable') from exc


def remove(name: str, payload: dict | None = None):
    if not payload or payload.get('storage') == 'master-key':
        return
    if not payload.get('keyring'):
        return
    try:
        import keyring
        keyring.delete_password(SERVICE,name)
    except keyring.errors.PasswordDeleteError:
        pass
    except Exception as exc:
        raise RuntimeError('Persistent OS credential storage is unavailable') from exc


def configured(payload: dict) -> bool:
    return bool(payload.get('ciphertext') or payload.get('keyring')) and not payload.get('disabled',False)


def ferni_encrypt(secret: str) -> str:
    try:
        from cryptography.fernet import Fernet
        return Fernet(master_key()).encrypt(secret.encode('utf-8')).decode('ascii')
    except Exception as exc:
        raise RuntimeError('Encrypted credential storage unavailable') from exc


def ferni_decrypt(ciphertext: str) -> str:
    try:
        from cryptography.fernet import Fernet
        return Fernet(master_key()).decrypt(ciphertext.encode('ascii')).decode('utf-8')
    except Exception as exc:
        raise RuntimeError('Encrypted credential storage unavailable') from exc
