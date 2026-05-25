"""
Database layer — SQLAlchemy 2.0 declarative models, session factory, and
ORM query helpers for the backtester Streamlit UI.

Design decisions
----------------
* SQLAlchemy 2.0 style throughout: ``DeclarativeBase``, ``Mapped``,
  ``mapped_column``, ``select()`` — no legacy ``Column()`` or
  ``session.query()``.
* Target: SQLite for development, PostgreSQL for production.
  Connection string is read from the ``DATABASE_URL`` environment variable;
  defaults to ``"sqlite:///./backtester.db"``.
* The ``result`` column on ``Run`` is a plain ``dict`` (pre-serialized by
  ``DictReporter``) — no backtester engine imports here.
* All role / status values are plain strings matching the UI spec exactly
  (``"admin"``, ``"RUNNING"``, ``"PRIVATE"``, …) — no engine enums.
* ``get_db()`` is a ``contextlib.contextmanager`` generator, not a class.
* ``init_db()`` is idempotent: safe to call multiple times.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

from sqlalchemy import (
    Boolean,
    ForeignKey,
    JSON,
    String,
    Text,
    create_engine,
    func,
    or_,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# ---------------------------------------------------------------------------
# Connection string
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./backtester.db")

# Module-level engine singleton (lazily initialised).
_engine: Optional[Engine] = None


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class User(Base):
    """
    Registered application user.

    Attributes
    ----------
    id:
        Auto-increment primary key.
    email:
        Unique login email (max 255 chars).
    password_hash:
        Bcrypt or similar hash — never returned to clients.
    name:
        Display name (max 255 chars).
    role:
        Access level: ``"admin"``, ``"analyst"``, or ``"viewer"``.
    created_at:
        UTC timestamp of account creation.
    avatar_initials:
        Up to 4 uppercase characters derived from ``name`` at creation time.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    avatar_initials: Mapped[str] = mapped_column(String(4), nullable=False)


class Run(Base):
    """
    A single backtest run record.

    Attributes
    ----------
    id:
        UUID string (36 chars) assigned at creation.
    owner_id:
        Foreign key to :class:`User`.
    strategy_name:
        Class name of the strategy that was run.
    params:
        JSON dict of strategy hyperparameters.
    status:
        ``"RUNNING"``, ``"DONE"``, or ``"FAILED"``.
    visibility:
        ``"PRIVATE"``, ``"TEAM"``, or ``"FEATURED"``.
    result:
        Nullable JSON dict — the serialized ``BacktestResult`` produced by
        ``DictReporter``; ``None`` while the run is still in progress.
    created_at:
        UTC timestamp of run creation.
    tags:
        JSON list of string tags for filtering.
    error_message:
        Populated only when ``status == "FAILED"``; ``None`` otherwise.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(255), nullable=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="RUNNING"
    )
    visibility: Mapped[str] = mapped_column(
        String(50), nullable=False, default="PRIVATE"
    )
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Comment(Base):
    """
    A comment left on a :class:`Run`.

    Attributes
    ----------
    id:
        Auto-increment primary key.
    run_id:
        Foreign key to :class:`Run`.
    author_id:
        Foreign key to :class:`User`.
    body:
        Free-text comment body.
    created_at:
        UTC timestamp of comment creation.
    mentions:
        JSON list of ``user_id`` integers that were @-mentioned in the body.
    """

    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    mentions: Mapped[list] = mapped_column(JSON, default=list)


class Strategy(Base):
    """
    A user-submitted trading strategy awaiting review.

    Attributes
    ----------
    id:
        Auto-increment primary key.
    name:
        Unique strategy name (max 255 chars).
    description:
        Free-text description of what the strategy does.
    code:
        Full Python source code of the strategy.
    author_id:
        Foreign key to :class:`User`.
    status:
        ``"DRAFT"``, ``"PENDING"``, or ``"APPROVED"``.
    created_at:
        UTC timestamp of submission.
    """

    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="DRAFT")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Notification(Base):
    """
    An in-app notification delivered to a user.

    Attributes
    ----------
    id:
        Auto-increment primary key.
    recipient_id:
        Foreign key to the target :class:`User`.
    type:
        Event type: ``"RUN_DONE"``, ``"RUN_FAILED"``, ``"COMMENT"``, or
        ``"STRATEGY_APPROVED"``.
    payload:
        JSON dict with context data (e.g. run_id, commenter name).
    read:
        ``False`` until the user explicitly marks it read.
    created_at:
        UTC timestamp of notification creation.
    """

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    recipient_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Engine / session factory
# ---------------------------------------------------------------------------


def get_engine() -> Engine:
    """
    Return the module-level SQLAlchemy ``Engine`` singleton.

    The engine is created lazily on the first call using :data:`DATABASE_URL`.
    Subsequent calls return the same instance.

    Returns
    -------
    Engine
        A configured SQLAlchemy engine.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
    return _engine


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context-manager that yields a bound ``Session`` and closes it on exit.

    Usage::

        with get_db() as db:
            user = get_user_by_email(db, "alice@example.com")

    Yields
    ------
    Session
        An open SQLAlchemy ORM session.
    """
    engine = get_engine()
    with Session(engine) as session:
        yield session


def init_db() -> None:
    """
    Create all tables defined in :data:`Base.metadata` if they do not exist.

    Idempotent — safe to call on every application start-up.
    """
    Base.metadata.create_all(get_engine())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _derive_initials(name: str) -> str:
    """
    Derive up to 4 uppercase initials from a display name.

    Splits *name* on whitespace and takes the first character of each word,
    capped at 4 characters total.

    Parameters
    ----------
    name:
        Display name, e.g. ``"Alice Wonderland"``.

    Returns
    -------
    str
        Initials string, e.g. ``"AW"``.
    """
    words = name.strip().split()
    initials = "".join(w[0].upper() for w in words if w)
    return initials[:4] if initials else "?"


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def get_user_by_email(db: Session, email: str) -> User | None:
    """
    Look up a user by email address.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    email:
        Email address to search for.

    Returns
    -------
    User | None
        The matching ``User`` row, or ``None`` if not found.
    """
    stmt = select(User).where(User.email == email)
    return db.execute(stmt).scalar_one_or_none()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """
    Look up a user by primary key.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    user_id:
        The integer primary key.

    Returns
    -------
    User | None
        The matching ``User`` row, or ``None`` if not found.
    """
    stmt = select(User).where(User.id == user_id)
    return db.execute(stmt).scalar_one_or_none()


def create_user(
    db: Session,
    email: str,
    name: str,
    password_hash: str,
    role: str,
) -> User:
    """
    Insert a new user row and return the persisted instance.

    ``avatar_initials`` is derived automatically from *name*.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    email:
        Unique login email.
    name:
        Display name (used to generate ``avatar_initials``).
    password_hash:
        Pre-hashed password string — never stored in plain text.
    role:
        One of ``"admin"``, ``"analyst"``, ``"viewer"``.

    Returns
    -------
    User
        The newly created and refreshed ``User`` row.
    """
    user = User(
        email=email,
        name=name,
        password_hash=password_hash,
        role=role,
        avatar_initials=_derive_initials(name),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_role(db: Session, user_id: int, role: str) -> User:
    """
    Change the role of an existing user.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    user_id:
        Primary key of the user to update.
    role:
        New role string (``"admin"``, ``"analyst"``, or ``"viewer"``).

    Returns
    -------
    User
        The updated and refreshed ``User`` row.

    Raises
    ------
    ValueError
        If no user with *user_id* exists.
    """
    user = get_user_by_id(db, user_id)
    if user is None:
        raise ValueError(f"User {user_id!r} not found.")
    user.role = role
    db.commit()
    db.refresh(user)
    return user


def list_users(db: Session) -> list[User]:
    """
    Return all users in insertion order.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.

    Returns
    -------
    list[User]
        Every row in the ``users`` table.
    """
    stmt = select(User)
    return list(db.execute(stmt).scalars().all())


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def create_run(
    db: Session,
    owner_id: int,
    strategy_name: str,
    params: dict[str, Any],
    visibility: str,
    tags: list[Any],
) -> Run:
    """
    Insert a new run row with ``status="RUNNING"`` and a fresh UUID.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    owner_id:
        Primary key of the owning :class:`User`.
    strategy_name:
        Class name of the strategy being run.
    params:
        Dict of strategy hyperparameters.
    visibility:
        ``"PRIVATE"``, ``"TEAM"``, or ``"FEATURED"``.
    tags:
        List of string tags.

    Returns
    -------
    Run
        The newly created and refreshed ``Run`` row.
    """
    run = Run(
        id=str(uuid.uuid4()),
        owner_id=owner_id,
        strategy_name=strategy_name,
        params=params,
        visibility=visibility,
        tags=tags,
        status="RUNNING",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: str) -> Run | None:
    """
    Look up a run by its UUID primary key.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    run_id:
        UUID string of the run.

    Returns
    -------
    Run | None
        The matching ``Run`` row, or ``None`` if not found.
    """
    stmt = select(Run).where(Run.id == run_id)
    return db.execute(stmt).scalar_one_or_none()


def update_run_status(
    db: Session,
    run_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> Run:
    """
    Update the status (and optionally result/error) of an existing run.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    run_id:
        UUID of the run to update.
    status:
        New status: ``"RUNNING"``, ``"DONE"``, or ``"FAILED"``.
    result:
        Serialized backtest result dict; stored only when not ``None``.
    error_message:
        Error detail; stored only when not ``None``.

    Returns
    -------
    Run
        The updated and refreshed ``Run`` row.

    Raises
    ------
    ValueError
        If no run with *run_id* exists.
    """
    run = get_run(db, run_id)
    if run is None:
        raise ValueError(f"Run {run_id!r} not found.")
    run.status = status
    if result is not None:
        run.result = result
    if error_message is not None:
        run.error_message = error_message
    db.commit()
    db.refresh(run)
    return run


def list_runs_for_user(db: Session, user_id: int, role: str) -> list[Run]:
    """
    Return runs visible to a user according to role-based visibility rules.

    Rules
    -----
    * ``"admin"`` — sees every run regardless of visibility.
    * ``"analyst"`` / ``"viewer"`` — sees own runs (any visibility) **plus**
      all runs with ``visibility`` in ``{"TEAM", "FEATURED"}``.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    user_id:
        Primary key of the requesting user.
    role:
        Role string determining which runs are visible.

    Returns
    -------
    list[Run]
        Runs visible to the requesting user.
    """
    if role == "admin":
        stmt = select(Run)
    else:
        stmt = select(Run).where(
            or_(
                Run.owner_id == user_id,
                Run.visibility.in_(["TEAM", "FEATURED"]),
            )
        )
    return list(db.execute(stmt).scalars().all())


def list_featured_runs(db: Session) -> list[Run]:
    """
    Return all runs with ``visibility == "FEATURED"``.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.

    Returns
    -------
    list[Run]
        All featured runs.
    """
    stmt = select(Run).where(Run.visibility == "FEATURED")
    return list(db.execute(stmt).scalars().all())


def update_run_visibility(db: Session, run_id: str, visibility: str) -> Run:
    """
    Change the visibility of an existing run.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    run_id:
        UUID of the run to update.
    visibility:
        New visibility: ``"PRIVATE"``, ``"TEAM"``, or ``"FEATURED"``.

    Returns
    -------
    Run
        The updated and refreshed ``Run`` row.

    Raises
    ------
    ValueError
        If no run with *run_id* exists.
    """
    run = get_run(db, run_id)
    if run is None:
        raise ValueError(f"Run {run_id!r} not found.")
    run.visibility = visibility
    db.commit()
    db.refresh(run)
    return run


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


def create_comment(
    db: Session,
    run_id: str,
    author_id: int,
    body: str,
    mentions: list[int],
) -> Comment:
    """
    Insert a new comment on a run.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    run_id:
        UUID of the run being commented on.
    author_id:
        Primary key of the commenting :class:`User`.
    body:
        Free-text comment body.
    mentions:
        List of ``user_id`` integers @-mentioned in *body*.

    Returns
    -------
    Comment
        The newly created and refreshed ``Comment`` row.
    """
    comment = Comment(
        run_id=run_id,
        author_id=author_id,
        body=body,
        mentions=mentions,
    )
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return comment


def list_comments_for_run(db: Session, run_id: str) -> list[Comment]:
    """
    Return all comments for a run ordered by creation time (oldest first).

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    run_id:
        UUID of the run whose comments to fetch.

    Returns
    -------
    list[Comment]
        Comments in chronological order.
    """
    stmt = (
        select(Comment)
        .where(Comment.run_id == run_id)
        .order_by(Comment.created_at)
    )
    return list(db.execute(stmt).scalars().all())


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def create_strategy(
    db: Session,
    name: str,
    description: str,
    code: str,
    author_id: int,
) -> Strategy:
    """
    Submit a new strategy with ``status="DRAFT"``.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    name:
        Unique strategy name.
    description:
        Human-readable description of the strategy logic.
    code:
        Full Python source code.
    author_id:
        Primary key of the submitting :class:`User`.

    Returns
    -------
    Strategy
        The newly created and refreshed ``Strategy`` row.
    """
    strategy = Strategy(
        name=name,
        description=description,
        code=code,
        author_id=author_id,
        status="DRAFT",
    )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    return strategy


def list_strategies(
    db: Session,
    status: str | None = None,
) -> list[Strategy]:
    """
    Return strategies, optionally filtered by status.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    status:
        If provided, only strategies with this exact status are returned
        (e.g. ``"APPROVED"``).  Pass ``None`` to return all strategies.

    Returns
    -------
    list[Strategy]
        Matching strategy rows.
    """
    stmt = select(Strategy)
    if status is not None:
        stmt = stmt.where(Strategy.status == status)
    return list(db.execute(stmt).scalars().all())


def update_strategy_status(
    db: Session,
    strategy_id: int,
    status: str,
) -> Strategy:
    """
    Change the review status of a strategy.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    strategy_id:
        Primary key of the strategy to update.
    status:
        New status: ``"DRAFT"``, ``"PENDING"``, or ``"APPROVED"``.

    Returns
    -------
    Strategy
        The updated and refreshed ``Strategy`` row.

    Raises
    ------
    ValueError
        If no strategy with *strategy_id* exists.
    """
    stmt = select(Strategy).where(Strategy.id == strategy_id)
    strategy = db.execute(stmt).scalar_one_or_none()
    if strategy is None:
        raise ValueError(f"Strategy {strategy_id!r} not found.")
    strategy.status = status
    db.commit()
    db.refresh(strategy)
    return strategy


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


def create_notification(
    db: Session,
    recipient_id: int,
    type: str,
    payload: dict[str, Any],
) -> Notification:
    """
    Insert a new notification for a user.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    recipient_id:
        Primary key of the target :class:`User`.
    type:
        Event type string: ``"RUN_DONE"``, ``"RUN_FAILED"``, ``"COMMENT"``,
        or ``"STRATEGY_APPROVED"``.
    payload:
        Context dict included in the notification (e.g. ``{"run_id": "…"}``).

    Returns
    -------
    Notification
        The newly created and refreshed ``Notification`` row.
    """
    notification = Notification(
        recipient_id=recipient_id,
        type=type,
        payload=payload,
    )
    db.add(notification)
    db.commit()
    db.refresh(notification)
    return notification


def list_notifications(
    db: Session,
    recipient_id: int,
    unread_only: bool = False,
) -> list[Notification]:
    """
    Return notifications for a user, optionally restricted to unread ones.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    recipient_id:
        Primary key of the recipient user.
    unread_only:
        When ``True``, only notifications with ``read=False`` are returned.

    Returns
    -------
    list[Notification]
        Matching notification rows.
    """
    stmt = select(Notification).where(
        Notification.recipient_id == recipient_id
    )
    if unread_only:
        stmt = stmt.where(Notification.read == False)  # noqa: E712
    return list(db.execute(stmt).scalars().all())


def mark_notification_read(db: Session, notification_id: int) -> Notification:
    """
    Mark a single notification as read.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    notification_id:
        Primary key of the notification to mark.

    Returns
    -------
    Notification
        The updated and refreshed ``Notification`` row.

    Raises
    ------
    ValueError
        If no notification with *notification_id* exists.
    """
    stmt = select(Notification).where(Notification.id == notification_id)
    notification = db.execute(stmt).scalar_one_or_none()
    if notification is None:
        raise ValueError(f"Notification {notification_id!r} not found.")
    notification.read = True
    db.commit()
    db.refresh(notification)
    return notification


def mark_all_notifications_read(db: Session, recipient_id: int) -> int:
    """
    Mark every unread notification for *recipient_id* as read.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    recipient_id:
        Primary key of the user whose notifications to mark.

    Returns
    -------
    int
        The number of notifications that were updated.
    """
    stmt = select(Notification).where(
        Notification.recipient_id == recipient_id,
        Notification.read == False,  # noqa: E712
    )
    notifications = list(db.execute(stmt).scalars().all())
    count = len(notifications)
    for notification in notifications:
        notification.read = True
    db.commit()
    return count


def unread_count(db: Session, recipient_id: int) -> int:
    """
    Return the number of unread notifications for *recipient_id*.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    recipient_id:
        Primary key of the user to count for.

    Returns
    -------
    int
        Count of ``Notification`` rows with ``read=False``.
    """
    stmt = (
        select(func.count())
        .select_from(Notification)
        .where(
            Notification.recipient_id == recipient_id,
            Notification.read == False,  # noqa: E712
        )
    )
    return db.execute(stmt).scalar_one()
