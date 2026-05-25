"""
Pydantic v2 schemas for the backtester UI database layer.

One schema per ORM model for API-style validation and serialization.
All schemas use ``model_config = ConfigDict(from_attributes=True)`` so they
can be constructed from SQLAlchemy ORM objects via::

    schema = FooSchema.model_validate(orm_obj)

Design notes
------------
* ``UserSchema`` deliberately **omits** ``password_hash`` — it must never be
  serialized to clients.
* ``RunSchema.result`` is ``dict | None`` — stores a pre-serialized
  ``BacktestResult`` dict (produced by ``DictReporter``); no engine types here.
* No Streamlit or backtester engine imports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class UserSchema(BaseModel):
    """
    Public representation of a :class:`~ui.db.User` row.

    ``password_hash`` is intentionally excluded.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    role: str
    created_at: datetime
    avatar_initials: str


class RunSchema(BaseModel):
    """
    Representation of a :class:`~ui.db.Run` row.

    ``result`` is ``None`` while the run is in progress; a plain ``dict``
    (the output of ``DictReporter``) once the run completes.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    owner_id: int
    strategy_name: str
    params: dict[str, Any]
    status: str
    visibility: str
    result: dict[str, Any] | None
    created_at: datetime
    tags: list[Any]
    error_message: str | None


class CommentSchema(BaseModel):
    """Representation of a :class:`~ui.db.Comment` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    author_id: int
    body: str
    created_at: datetime
    mentions: list[int]


class StrategySchema(BaseModel):
    """Representation of a :class:`~ui.db.Strategy` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    code: str
    author_id: int
    status: str
    created_at: datetime


class NotificationSchema(BaseModel):
    """Representation of a :class:`~ui.db.Notification` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    recipient_id: int
    type: str
    payload: dict[str, Any]
    read: bool
    created_at: datetime
