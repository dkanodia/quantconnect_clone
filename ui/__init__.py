"""
UI layer for the backtester Streamlit application.

Sub-modules
-----------
db      — SQLAlchemy 2.0 models, session factory, and ORM query helpers.
models  — Pydantic v2 schemas for serialization and API-style validation.

No backtester engine imports belong here; the ``result`` column on ``Run``
stores a plain ``dict`` (pre-serialized by ``DictReporter``).
"""
