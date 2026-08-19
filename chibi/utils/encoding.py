"""Stream encoding utilities."""

import io


def reconfigure_stream_utf8(stream: io.TextIOWrapper) -> None:
    """Reconfigure a text stream to use UTF-8 with replacement errors.

    Args:
        stream: The text stream to reconfigure.
    """
    if isinstance(stream, io.TextIOWrapper):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
