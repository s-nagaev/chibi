"""Canonical handling of the ``<chibi>ACK</chibi>`` silence sentinel.

The sentinel is the platform's convention for a silent assistant turn: the
model answers with exactly ``<chibi>ACK</chibi>`` and the answer is persisted
but never delivered. These helpers are the single source of truth for
detecting and sanitizing the sentinel — no other module may re-implement the
predicate (see ``.project/ack_rule_fix/DESIGN_mechanical_silence.md`` §7.1).
"""

ACK_SENTINEL = "<chibi>ACK</chibi>"


def is_pure_ack(answer: str) -> bool:
    """Check whether the answer is exactly the ACK sentinel (silence convention).

    Args:
        answer: The agent's answer text.

    Returns:
        True if the stripped answer equals the sentinel exactly, False otherwise.
    """
    return answer.strip() == ACK_SENTINEL


def strip_ack_sentinel(text: str) -> str:
    """Remove every ACK sentinel occurrence from the text.

    Args:
        text: Text that may contain the sentinel.

    Returns:
        The text with all sentinel occurrences removed.
    """
    return text.replace(ACK_SENTINEL, "")
