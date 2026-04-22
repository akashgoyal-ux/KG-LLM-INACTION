"""Alias module: re-exports everything from 01_zero_shot_entity_extraction.py"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path

_src = Path(__file__).parent / "01_zero_shot_entity_extraction.py"
_spec = importlib.util.spec_from_file_location("_l01_impl", _src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules[__name__].__dict__.update({k: v for k, v in _mod.__dict__.items() if not k.startswith("__")})
