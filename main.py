import io
import sys
from typing import cast

from chibi.runners.telegram import run_chibi
from chibi.utils.encoding import reconfigure_stream_utf8

reconfigure_stream_utf8(cast(io.TextIOWrapper, sys.stdout))
reconfigure_stream_utf8(cast(io.TextIOWrapper, sys.stderr))

if __name__ == "__main__":
    run_chibi()
