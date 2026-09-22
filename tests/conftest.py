"""Shared pytest setup.

JWT_SECRET must be set before anything imports src.core.security, since that
module fails loudly at import time if it's missing. conftest.py is imported
by pytest before test modules are collected, so set it here.
"""

import os

os.environ.setdefault(
    "JWT_SECRET", "pytest-jwt-secret-not-for-production-use-0123456789"
)
