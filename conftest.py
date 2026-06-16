"""conftest.py — ensure the repo root is on sys.path so tests can import shtrunc."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
