"""
Setup script for backwards compatibility.

Modern installations should use pyproject.toml with pip >= 21.3:
    pip install tocantins-framework

For older pip versions, this setup.py provides fallback support.
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This file exists for backwards compatibility

setup()