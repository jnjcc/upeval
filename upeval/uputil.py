# -*- coding: utf-8 -*-

from pathlib import Path

def ensure_dir(dir):
    Path(dir).mkdir(parents = True, exist_ok = True)
