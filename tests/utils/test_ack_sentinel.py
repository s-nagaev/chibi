"""Unit tests for the canonical ACK sentinel helpers and the streaming gate."""

from chibi.services.providers.streaming import AckSentinelStreamGate
from chibi.utils.ack_sentinel import ACK_SENTINEL, is_pure_ack, strip_ack_sentinel


class TestIsPureAck:
    def test_exact_marker_is_pure_ack(self) -> None:
        assert is_pure_ack("<chibi>ACK</chibi>") is True

    def test_surrounding_whitespace_is_pure_ack(self) -> None:
        assert is_pure_ack("  <chibi>ACK</chibi>\n") is True

    def test_lowercase_marker_is_not_pure_ack(self) -> None:
        assert is_pure_ack("<chibi>ack</chibi>") is False

    def test_marker_inside_text_is_not_pure_ack(self) -> None:
        assert is_pure_ack(f"Here is the answer {ACK_SENTINEL} done") is False

    def test_prefix_of_marker_is_not_pure_ack(self) -> None:
        assert is_pure_ack("<chibi>ACK") is False

    def test_plain_text_is_not_pure_ack(self) -> None:
        assert is_pure_ack("All checks passed") is False

    def test_empty_answer_is_not_pure_ack(self) -> None:
        assert is_pure_ack("") is False


class TestStripAckSentinel:
    def test_removes_marker_from_mixed_text(self) -> None:
        assert strip_ack_sentinel(f"before {ACK_SENTINEL} after") == "before  after"

    def test_removes_multiple_markers(self) -> None:
        assert strip_ack_sentinel(f"{ACK_SENTINEL}x{ACK_SENTINEL}") == "x"

    def test_leaves_plain_text_untouched(self) -> None:
        assert strip_ack_sentinel("ordinary answer") == "ordinary answer"

    def test_leaves_lowercase_marker_untouched(self) -> None:
        assert strip_ack_sentinel("<chibi>ack</chibi>") == "<chibi>ack</chibi>"


class TestAckSentinelStreamGate:
    def test_pure_ack_in_one_chunk_never_emits(self) -> None:
        gate = AckSentinelStreamGate()
        assert gate.filter(ACK_SENTINEL) == ""
        assert gate.flush() == ""

    def test_pure_ack_fragmented_never_emits(self) -> None:
        gate = AckSentinelStreamGate()
        for fragment in ("<chibi", ">ACK", "</chibi", ">"):
            assert gate.filter(fragment) == ""
        assert gate.flush() == ""

    def test_pure_ack_with_surrounding_whitespace_never_emits(self) -> None:
        gate = AckSentinelStreamGate()
        assert gate.filter("  ") == ""
        assert gate.filter(ACK_SENTINEL) == ""
        assert gate.filter(" \n") == ""
        assert gate.flush() == ""

    def test_normal_text_is_released_immediately(self) -> None:
        gate = AckSentinelStreamGate()
        assert gate.filter("Hello") == "Hello"
        assert gate.filter(" world") == " world"
        assert gate.flush() == ""

    def test_leading_whitespace_then_text_is_released_intact(self) -> None:
        gate = AckSentinelStreamGate()
        assert gate.filter("   ") == ""
        assert gate.filter("answer") == "   answer"
        assert gate.flush() == ""

    def test_marker_inside_longer_stream_is_stripped(self) -> None:
        gate = AckSentinelStreamGate()
        assert gate.filter("Result: ") == "Result: "
        assert gate.filter(f"{ACK_SENTINEL} done") == " done"
        assert gate.flush() == ""

    def test_marker_split_across_released_chunks_is_stripped(self) -> None:
        gate = AckSentinelStreamGate()
        assert gate.filter("A") == "A"
        assert gate.filter("<chibi>A") == ""
        assert gate.filter("CK</chib") == ""
        assert gate.filter("i>") == ""
        assert gate.filter("B") == "B"
        assert gate.flush() == ""

    def test_marker_prefix_lookalike_at_end_is_flushed(self) -> None:
        gate = AckSentinelStreamGate()
        assert gate.filter("<chibi") == ""
        assert gate.flush() == "<chibi"

    def test_flush_drains_held_tail_after_release(self) -> None:
        gate = AckSentinelStreamGate()
        assert gate.filter("ok <chibi") == "ok "
        assert gate.flush() == "<chibi"

    def test_candidate_text_diverging_from_marker_releases_whole_buffer(self) -> None:
        gate = AckSentinelStreamGate()
        assert gate.filter("<chibi") == ""
        assert gate.filter("X") == "<chibiX"
        assert gate.flush() == ""
