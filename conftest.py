"""Pytest configuration - ensures src package is importable."""
import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Pre-import FeatureEngineer so pickle can find it during test loading
try:
    from src.data_preprocessing import FeatureEngineer  # noqa: F401
except ImportError:
    pass