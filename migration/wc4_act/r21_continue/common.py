"""Compatibility binding for unchanged author export/isolated evaluation adapters."""
from runtime import *
RUN,ORDER=authorized_run()
ID=RUN.name
OLD=DATA.parents[2]
def verify():verify_run(RUN)
