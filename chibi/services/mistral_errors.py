"""Status-code classification of Mistral SDK errors.

The Mistral SDK (3.3.0) exposes no per-status exception classes: every failed
HTTP response surfaces as ``SDKError`` (a ``MistralError`` subclass) carrying
the raw ``httpx2.Response``, and transport-level failures inside the client
surface as plain ``httpx2`` exceptions. Both consumers, the provider adapter's
retry policy and the failover engine's trigger classifier, share this module.

Leaf module: imports nothing from ``chibi.*``, so
``chibi.services.failover`` can import it without creating a cycle
(``chibi.services.providers.tools.utils`` already imports
``chibi.services.failover``).
"""

from typing import Literal

import httpx2
from mistralai.client.errors import SDKError

MistralErrorKind = Literal["rate_limit", "server_error", "auth", "timeout", "network"]


def classify_mistral_exception(exception: BaseException) -> MistralErrorKind | None:
    """Map a Mistral SDK error to a failover trigger kind.

    HTTP errors are classified from ``SDKError.raw_response.status_code``
    (the SDK has no per-status exception classes): 429 is a rate limit, any
    5xx is a server error, and 401/403 are auth triggers (owner decision:
    these codes also arrive when the account is out of credit or cut off,
    which is exactly when another provider can still serve the request, so
    they fail over unconditionally — with no retries, see the adapter's
    retry predicate). Transport failures inside the Mistral client are
    plain ``httpx2`` exceptions: timeouts map to ``timeout`` and every
    other transport error to ``network``.

    Args:
        exception: The exception raised by a Mistral client call.

    Returns:
        The trigger kind, or None when the exception must not trigger
        failover.
    """
    if isinstance(exception, httpx2.TimeoutException):
        return "timeout"
    if isinstance(exception, httpx2.TransportError):
        return "network"
    if isinstance(exception, SDKError):
        status_code = exception.raw_response.status_code
        if status_code == 429:
            return "rate_limit"
        if status_code >= 500:
            return "server_error"
        if status_code in (401, 403):
            return "auth"
        return None
    return None
