#!/usr/bin/env python3
# Kompatibilnost sa starijim nazivom skripte.
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name('00_prepare_archives.py')), run_name='__main__')
