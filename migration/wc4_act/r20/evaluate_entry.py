"""R20 stage failures request the established consistent-boundary stop."""
import os,sys,runpy,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'migration/wc4_act/r13'),str(ROOT/'migration/wc4_act')]
from stopping import request_stop
try:runpy.run_path(str(Path(__file__).with_name('evaluate.py')),run_name='__main__')
except BaseException as exc:
    if isinstance(exc,SystemExit) and exc.code in [None,0]:raise
    request_stop(Path(os.environ['WC4_EXECUTION_RUN']),'R20 evaluation/export/evidence failure',error=repr(exc),traceback=traceback.format_exc())
    raise SystemExit(75)
