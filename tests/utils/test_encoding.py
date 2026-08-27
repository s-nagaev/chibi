"""Tests for stream encoding utilities."""

import io
from io import StringIO
from pathlib import Path
from typing import cast
from unittest.mock import patch

from chibi.utils.encoding import reconfigure_stream_utf8


class TestReconfigureStreamUtf8:
    """Tests for reconfigure_stream_utf8."""

    def test_stringio_no_exception(self) -> None:
        """StringIO lacks reconfigure and should be handled gracefully."""
        stream = cast(io.TextIOWrapper, StringIO("test"))
        reconfigure_stream_utf8(stream)

    def test_value_error_swallowed(self) -> None:
        """A stream whose reconfigure raises ValueError should not propagate."""
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="ascii")
        with patch.object(stream, "reconfigure", side_effect=ValueError("cannot reconfigure")):
            reconfigure_stream_utf8(stream)

    def test_reconfigures_text_io_wrapper(self, tmp_path: Path) -> None:
        """A real TextIOWrapper should be reconfigured to UTF-8."""
        file_path = tmp_path / "test.txt"
        with open(file_path, "w", encoding="ascii") as stream:
            assert stream.encoding == "ascii"
            reconfigure_stream_utf8(stream)
            assert stream.encoding == "utf-8"
