"""Entry point for `python -m chibi`: stdout/stderr are reconfigured to UTF-8 at import time so that
the runner import (deferred below) and everything it pulls in can already log non-ASCII safely."""

import io
import sys
from typing import cast

from chibi.utils.encoding import reconfigure_stream_utf8

reconfigure_stream_utf8(cast(io.TextIOWrapper, sys.stdout))
reconfigure_stream_utf8(cast(io.TextIOWrapper, sys.stderr))

if __name__ == "__main__":
    from chibi.runners.telegram import run_chibi

    run_chibi()
