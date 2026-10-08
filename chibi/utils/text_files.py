"""Detection helpers for text-based files sent by users."""

from pathlib import Path

TEXT_MIME_TYPES: frozenset[str] = frozenset(
    {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/javascript",
        "application/x-python",
        "application/toml",
        "application/x-sh",
    }
)

TEXT_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".txt",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".html",
        ".css",
        ".sh",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".rb",
        ".php",
        ".sql",
        ".ini",
        ".cfg",
        ".conf",
        ".log",
        ".xml",
        ".svg",
    }
)


def is_text_file(mime_type: str | None, file_name: str | None) -> bool:
    """Decide whether a file should be treated as readable text.

    The MIME type is checked first against ``text/*`` and a hardcoded
    allowlist of text-like application types. If the MIME type does not
    match, the file extension is used as a fallback, because Telegram
    frequently reports ``application/octet-stream`` for plain-text files
    it cannot classify.

    Args:
        mime_type: Reported MIME type of the file, or None if unknown.
        file_name: Name of the file, or None if unknown.

    Returns:
        True if the file is considered a text file, False otherwise.
    """
    if mime_type is not None:
        if mime_type.startswith("text/") or mime_type in TEXT_MIME_TYPES:
            return True
    if file_name is not None:
        extension = Path(file_name).suffix.lower()
        if extension in TEXT_FILE_EXTENSIONS:
            return True
    return False
