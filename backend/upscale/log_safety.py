"""Secret-safe logging: credentials never reach a log line, at any level, from any logger.

Installed once, process-wide, as a `logging` record factory, so it applies to every logger
(UpScale's, httpx2 / httpcore request logs, uvicorn, SDKs) and every handler, including
DEBUG output and formatted tracebacks. Each record's message is rendered and redacted when
it is created (arguments keep their types unless they hold a credential, so formatters
that read `record.args`, like uvicorn's access log, keep working); exception text is
rendered and redacted too.

Redacted:

* every configured credential value (registered from the environment by
  `upscale.config`), wherever it appears: a URL, a header dump, an exception message;
* sensitive URL query parameters (``api-key``, ``api_key``, ``apikey``, ``key``,
  ``token``, ``access_token``, ``secret``, ``password``, ``signature``...), e.g. the
  Helius RPC URL's ``?api-key=``;
* ``Authorization`` / ``Proxy-Authorization`` values and ``Bearer`` tokens;
* API-key style headers (``x-api-key``, ``x-cg-demo-api-key``, ``x-auth-token``...),
  including ``b'...'`` byte-header dumps.

Provider names stay visible ("could not reach Helius"); only credentials are masked.
"""

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlsplit

MASK = "***"
# Shorter values would mask ordinary words; real credentials are longer.
MIN_SECRET_LENGTH = 8

_QUERY = re.compile(
    r"(?i)([?&;](?:api[-_]?key|apikey|key|token|access[-_]?token|refresh[-_]?token|auth|"
    r"secret|client[-_]?secret|password|passwd|signature|sig)=)[^&#\s'\"<>]+"
)
_HEADER = re.compile(
    r"(?i)(['\"]?\b(?:authorization|proxy-authorization|x-api-key|api[-_]?key|"
    r"x-cg-(?:demo|pro)-api-key|x-auth-token|x-access-token|client[-_]?secret|"
    r"access[-_]?token|refresh[-_]?token)\b['\"]?\s*[:=,]\s*(?:b(?=['\"]))?['\"]?)"
    r"(?:(bearer|basic|token)\s+)?[^'\"\s,}&)\]]+"
)
_BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9\-._~+/]{8,}=*")

_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Mask this exact value wherever it appears (ignored if empty or too short)."""
    if value and len(value.strip()) >= MIN_SECRET_LENGTH:
        _secrets.add(value.strip())


def register_url(url: str | None) -> None:
    """A credential-bearing URL (e.g. a private RPC endpoint): the whole URL, its
    query values and long path segments are masked."""
    if not url:
        return
    register_secret(url)
    try:
        parts = urlsplit(url)
    except ValueError:
        return
    for _, value in parse_qsl(parts.query):
        register_secret(value)
    for segment in parts.path.split("/"):
        if len(segment) >= 16:
            register_secret(segment)
    if parts.password:
        register_secret(parts.password)


def register_secrets(values: Iterable[str | None]) -> None:
    for v in values:
        register_secret(v)


def redact(text: str) -> str:
    # Longest first, so a value containing another is masked whole.
    for secret in sorted(_secrets, key=len, reverse=True):
        if secret in text:
            text = text.replace(secret, MASK)
    text = _QUERY.sub(lambda m: m.group(1) + MASK, text)
    text = _HEADER.sub(lambda m: m.group(1) + (f"{m.group(2)} " if m.group(2) else "") + MASK, text)
    return _BEARER.sub(lambda m: f"{m.group(1)} {MASK}", text)


def _safe(value: Any) -> Any:
    """An argument unchanged (type kept, so formatters like "%d" or uvicorn's access log
    still work) unless its text holds a credential; then its redacted text."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return redact(value)
    for text in (str(value), repr(value)):
        if redact(text) != text:
            return redact(str(value))
    return value


def _scrub(record: logging.LogRecord) -> None:
    # The template is judged only once assembled: masking it alone could swallow a
    # placeholder ("Bearer %s") and orphan its argument.
    if isinstance(record.args, Mapping):
        record.args = {k: _safe(v) for k, v in record.args.items()}
    elif isinstance(record.args, tuple):
        record.args = tuple(_safe(a) for a in record.args)
    try:
        message = record.getMessage()
    except Exception:  # a malformed call: keep what can be shown, still redacted
        record.msg, record.args = redact(f"{record.msg} {record.args!r}"), None
    else:
        if redact(message) != message:  # only visible once assembled (e.g. "key=%s")
            record.msg, record.args = redact(message), None
    if record.exc_info:
        text = logging.Formatter().formatException(record.exc_info)
        # Formatters print `exc_text` as is; `exc_info` is dropped so nothing re-renders
        # the raw exception (whose message or chained causes may hold a URL).
        record.exc_text, record.exc_info = redact(text), None
    elif record.exc_text:
        record.exc_text = redact(record.exc_text)
    if record.stack_info:
        record.stack_info = redact(record.stack_info)


_installed = False


def install() -> None:
    """Redact every log record created from now on (idempotent)."""
    global _installed
    if _installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        _scrub(record)
        return record

    logging.setLogRecordFactory(factory)
    _installed = True
