"""
Authentication layer for the backtester Streamlit UI.

Responsibilities
----------------
* Password hashing / verification via ``passlib`` + ``bcrypt``.
* Session management — read/write :data:`st.session_state` using the
  fixed key constants below.
* Role-based access control via :func:`require_role`.
* Admin seeding via :func:`seed_admin`.

Import discipline
-----------------
``streamlit`` is imported at the module level so that test monkeypatching
works without a live Streamlit runtime::

    monkeypatch.setattr("ui.auth.st.session_state", {})

No Streamlit APIs (``st.title``, ``st.write``, etc.) are called at module
level — only inside the functions that need them.

No backtester engine imports here; all persistence goes through
:mod:`ui.db`.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ui.db import User, create_user, get_user_by_email

# ---------------------------------------------------------------------------
# Password context — bcrypt with automatic hash deprecation handling
# ---------------------------------------------------------------------------

_pwd_context: CryptContext = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Session-state key constants
# Pages and tests rely on these exact strings — do not rename.
# ---------------------------------------------------------------------------

_KEY_USER_ID: str = "auth_user_id"       # int
_KEY_ROLE: str = "auth_role"             # str
_KEY_NAME: str = "auth_name"             # str
_KEY_EMAIL: str = "auth_email"           # str
_KEY_AVATAR: str = "auth_avatar_initials"  # str

_AUTH_KEYS: tuple[str, ...] = (
    _KEY_USER_ID,
    _KEY_ROLE,
    _KEY_NAME,
    _KEY_EMAIL,
    _KEY_AVATAR,
)


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using bcrypt.

    Each call produces a unique hash (bcrypt generates a random salt
    internally), so the same password will yield different hashes on
    successive calls — all of which still verify correctly.

    Parameters
    ----------
    password:
        The plain-text password to hash. **Never log or store this value.**

    Returns
    -------
    str
        A bcrypt hash string suitable for storage in the ``password_hash``
        column of :class:`~ui.db.User`.
    """
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plain-text password against a stored bcrypt hash.

    Parameters
    ----------
    plain:
        The plain-text password supplied by the user at login.
    hashed:
        The stored bcrypt hash to compare against.

    Returns
    -------
    bool
        ``True`` if *plain* matches *hashed*; ``False`` otherwise.
    """
    return _pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def login(user: User) -> None:
    """
    Write the authenticated user's info into Streamlit session state.

    Sets the following keys:

    * ``"auth_user_id"``        → ``int``
    * ``"auth_role"``           → ``str``
    * ``"auth_name"``           → ``str``
    * ``"auth_email"``          → ``str``
    * ``"auth_avatar_initials"``→ ``str``

    Parameters
    ----------
    user:
        The authenticated :class:`~ui.db.User` ORM instance (must have
        its attributes loaded while the SQLAlchemy session is still open).
    """
    st.session_state[_KEY_USER_ID] = user.id
    st.session_state[_KEY_ROLE] = user.role
    st.session_state[_KEY_NAME] = user.name
    st.session_state[_KEY_EMAIL] = user.email
    st.session_state[_KEY_AVATAR] = user.avatar_initials


def logout() -> None:
    """
    Remove all authentication keys from Streamlit session state.

    Other keys in ``session_state`` are left untouched. Safe to call
    even when no user is currently logged in (no-op in that case).
    """
    for key in _AUTH_KEYS:
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    """
    Return ``True`` if a user is currently logged in.

    Checks for the presence of ``"auth_user_id"`` in ``session_state``
    — the key written by :func:`login` and cleared by :func:`logout`.

    Returns
    -------
    bool
        ``True`` when the session contains valid auth info.
    """
    return _KEY_USER_ID in st.session_state


def get_current_user() -> dict[str, Any] | None:
    """
    Return the current user's auth dict from session state, or ``None``.

    Returns
    -------
    dict[str, Any] | None
        A dict with keys ``"user_id"``, ``"name"``, ``"email"``,
        ``"role"``, ``"avatar_initials"`` when authenticated; ``None``
        if no user is logged in.
    """
    if not is_authenticated():
        return None
    return {
        "user_id": st.session_state[_KEY_USER_ID],
        "name": st.session_state[_KEY_NAME],
        "email": st.session_state[_KEY_EMAIL],
        "role": st.session_state[_KEY_ROLE],
        "avatar_initials": st.session_state[_KEY_AVATAR],
    }


# ---------------------------------------------------------------------------
# Role enforcement
# ---------------------------------------------------------------------------


def require_role(allowed_roles: list[str]) -> None:
    """
    Enforce role-based access control on the current Streamlit page.

    Calls ``st.stop()`` — not ``raise`` or ``return`` — to halt page
    rendering in both failure branches:

    * **Not authenticated** — shows a "Please log in" warning and stops.
    * **Role not allowed** — shows an "Access denied" error and stops.

    If the user's role is in *allowed_roles*, this function returns
    normally and the page continues to render.

    Parameters
    ----------
    allowed_roles:
        List of role strings permitted to view the current page
        (e.g. ``["admin", "analyst"]``).
    """
    if not is_authenticated():
        st.warning("Please log in to access this page.")
        st.stop()
    else:
        current = get_current_user()
        if current["role"] not in allowed_roles:  # type: ignore[index]
            st.error(
                f"Access denied. This page requires one of the following "
                f"roles: {', '.join(allowed_roles)}."
            )
            st.stop()


# ---------------------------------------------------------------------------
# Admin seeding
# ---------------------------------------------------------------------------


def seed_admin(db: Session, email: str, name: str, password: str) -> User:
    """
    Create an admin user if one with *email* does not already exist.

    Idempotent — safe to call on every application start-up. If a user
    with the given *email* already exists, it is returned unchanged
    regardless of its current role or password.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    email:
        Email address of the admin account to create or fetch.
    name:
        Display name for the new admin (ignored if the user exists).
    password:
        Plain-text password; hashed before storage via :func:`hash_password`.
        **Never logged or stored in plain text.**

    Returns
    -------
    User
        The existing or newly created :class:`~ui.db.User` row.
    """
    existing = get_user_by_email(db, email)
    if existing is not None:
        return existing
    return create_user(
        db,
        email=email,
        name=name,
        password_hash=hash_password(password),
        role="admin",
    )
