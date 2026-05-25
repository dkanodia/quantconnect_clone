"""
Tests for ui/auth.py.

All tests run without a live Streamlit runtime.  ``st.session_state`` is
replaced with a plain dict via ``monkeypatch``; ``st.stop()``,
``st.warning()``, and ``st.error()`` are patched with ``MagicMock`` in the
``require_role`` tests.

Covers
------
Password utilities
    hash_password produces a bcrypt hash, not the original string
    verify_password True for correct, False for wrong
    hash_password produces different hashes for the same input (random salt)

Login / logout
    login() writes all five auth keys to session state
    logout() removes all five auth keys, leaves other keys intact
    logout() is safe to call on empty state (no-op)

is_authenticated / get_current_user
    False on empty state, True after login, False after logout
    get_current_user returns None when not logged in, correct dict when logged in

require_role
    calls st.stop() when not authenticated (shows warning)
    calls st.stop() when role not in allowed list (shows error)
    does NOT call st.stop() when role is in allowed list

seed_admin
    creates user with role="admin" if email absent
    no-op (no duplicate) if email already exists
    stored password verifies correctly
    returns existing user when called again with same email
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ui.auth import (
    get_current_user,
    hash_password,
    is_authenticated,
    login,
    logout,
    require_role,
    seed_admin,
    verify_password,
)
from ui.db import Base, create_user, get_user_by_email, list_users


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session_state(monkeypatch):
    """
    Replace ``st.session_state`` with a plain dict for the duration of
    the test, then restore the original on teardown (handled by monkeypatch).
    """
    state: dict = {}
    monkeypatch.setattr("ui.auth.st.session_state", state)
    return state


@pytest.fixture
def db() -> Session:
    """Fresh in-memory SQLite session per test — no filesystem I/O."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def analyst_user(db: Session):
    """Analyst user with a known plain-text password ``"correct_password"``."""
    return create_user(
        db,
        email="analyst@example.com",
        name="Alice Analyst",
        password_hash=hash_password("correct_password"),
        role="analyst",
    )


@pytest.fixture
def viewer_user(db: Session):
    """Viewer user — used for role-access denial tests."""
    return create_user(
        db,
        email="viewer@example.com",
        name="Victor Viewer",
        password_hash=hash_password("viewerpass"),
        role="viewer",
    )


# ---------------------------------------------------------------------------
# Shared helper: patch st.stop / st.warning / st.error
# ---------------------------------------------------------------------------


def _patch_streamlit_ui(monkeypatch) -> dict[str, MagicMock]:
    """
    Patch ``st.stop``, ``st.warning``, and ``st.error`` on the ``ui.auth``
    module's ``st`` reference.

    Returns a dict of the three mocks keyed by short name.
    """
    mocks = {
        "stop": MagicMock(),
        "warning": MagicMock(),
        "error": MagicMock(),
    }
    monkeypatch.setattr("ui.auth.st.stop", mocks["stop"])
    monkeypatch.setattr("ui.auth.st.warning", mocks["warning"])
    monkeypatch.setattr("ui.auth.st.error", mocks["error"])
    return mocks


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_is_not_original_password(self) -> None:
        assert hash_password("mypassword") != "mypassword"

    def test_hash_is_a_string(self) -> None:
        assert isinstance(hash_password("anything"), str)

    def test_verify_correct_password_returns_true(self) -> None:
        hashed = hash_password("correct")
        assert verify_password("correct", hashed) is True

    def test_verify_wrong_password_returns_false(self) -> None:
        hashed = hash_password("correct")
        assert verify_password("wrong", hashed) is False

    def test_different_hashes_for_same_input(self) -> None:
        """bcrypt uses a random salt — two hashes of the same input differ."""
        h1 = hash_password("same_input")
        h2 = hash_password("same_input")
        assert h1 != h2

    def test_both_salted_hashes_verify_against_original(self) -> None:
        """Both salted hashes still pass verification."""
        h1 = hash_password("same_input")
        h2 = hash_password("same_input")
        assert verify_password("same_input", h1) is True
        assert verify_password("same_input", h2) is True

    def test_verify_is_case_sensitive(self) -> None:
        hashed = hash_password("Password")
        assert verify_password("password", hashed) is False
        assert verify_password("PASSWORD", hashed) is False


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class TestLogin:
    def test_writes_all_five_auth_keys(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        assert mock_session_state["auth_user_id"] == analyst_user.id
        assert mock_session_state["auth_role"] == analyst_user.role
        assert mock_session_state["auth_name"] == analyst_user.name
        assert mock_session_state["auth_email"] == analyst_user.email
        assert mock_session_state["auth_avatar_initials"] == analyst_user.avatar_initials

    def test_user_id_is_int(self, mock_session_state: dict, analyst_user) -> None:
        login(analyst_user)
        assert isinstance(mock_session_state["auth_user_id"], int)

    def test_login_overwrites_previous_session(
        self, mock_session_state: dict, db: Session
    ) -> None:
        u1 = create_user(
            db, email="u1@ex.com", name="U One",
            password_hash="h", role="viewer",
        )
        u2 = create_user(
            db, email="u2@ex.com", name="U Two",
            password_hash="h", role="admin",
        )
        login(u1)
        login(u2)
        assert mock_session_state["auth_email"] == "u2@ex.com"
        assert mock_session_state["auth_role"] == "admin"

    def test_avatar_initials_written_correctly(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        assert mock_session_state["auth_avatar_initials"] == "AA"  # Alice Analyst


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


class TestLogout:
    def test_clears_all_five_auth_keys(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        logout()
        for key in (
            "auth_user_id", "auth_role", "auth_name",
            "auth_email", "auth_avatar_initials",
        ):
            assert key not in mock_session_state

    def test_leaves_other_keys_untouched(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        mock_session_state["unrelated_key"] = "keep_me"
        mock_session_state["counter"] = 42
        logout()
        assert mock_session_state["unrelated_key"] == "keep_me"
        assert mock_session_state["counter"] == 42

    def test_safe_to_call_on_empty_state(self, mock_session_state: dict) -> None:
        """Logout on an empty session state must not raise."""
        logout()  # no error

    def test_idempotent_double_logout(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        logout()
        logout()  # second call must not raise


# ---------------------------------------------------------------------------
# is_authenticated
# ---------------------------------------------------------------------------


class TestIsAuthenticated:
    def test_returns_false_on_empty_state(self, mock_session_state: dict) -> None:
        assert is_authenticated() is False

    def test_returns_true_after_login(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        assert is_authenticated() is True

    def test_returns_false_after_logout(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        logout()
        assert is_authenticated() is False


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    def test_returns_none_when_not_authenticated(
        self, mock_session_state: dict
    ) -> None:
        assert get_current_user() is None

    def test_returns_none_after_logout(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        logout()
        assert get_current_user() is None

    def test_returns_correct_dict_when_authenticated(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        result = get_current_user()
        assert result is not None
        assert result["user_id"] == analyst_user.id
        assert result["name"] == analyst_user.name
        assert result["email"] == analyst_user.email
        assert result["role"] == analyst_user.role
        assert result["avatar_initials"] == analyst_user.avatar_initials

    def test_result_contains_exactly_five_keys(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        result = get_current_user()
        assert set(result.keys()) == {  # type: ignore[arg-type]
            "user_id", "name", "email", "role", "avatar_initials",
        }

    def test_result_does_not_contain_password_hash(
        self, mock_session_state: dict, analyst_user
    ) -> None:
        login(analyst_user)
        result = get_current_user()
        assert "password_hash" not in result  # type: ignore[operator]


# ---------------------------------------------------------------------------
# require_role
# ---------------------------------------------------------------------------


class TestRequireRole:
    # ── not authenticated ──────────────────────────────────────────────────

    def test_calls_stop_when_not_authenticated(
        self, mock_session_state: dict, monkeypatch
    ) -> None:
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin"])
        mocks["stop"].assert_called_once()

    def test_shows_warning_when_not_authenticated(
        self, mock_session_state: dict, monkeypatch
    ) -> None:
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin"])
        mocks["warning"].assert_called_once()

    def test_does_not_call_error_when_not_authenticated(
        self, mock_session_state: dict, monkeypatch
    ) -> None:
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin"])
        mocks["error"].assert_not_called()

    # ── authenticated but wrong role ───────────────────────────────────────

    def test_calls_stop_when_role_is_viewer_not_admin(
        self, mock_session_state: dict, viewer_user, monkeypatch
    ) -> None:
        login(viewer_user)
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin"])
        mocks["stop"].assert_called_once()

    def test_shows_error_when_role_not_in_allowed_list(
        self, mock_session_state: dict, viewer_user, monkeypatch
    ) -> None:
        login(viewer_user)
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin"])
        mocks["error"].assert_called_once()

    def test_does_not_show_warning_when_authenticated_wrong_role(
        self, mock_session_state: dict, viewer_user, monkeypatch
    ) -> None:
        login(viewer_user)
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin"])
        mocks["warning"].assert_not_called()

    def test_calls_stop_for_analyst_when_only_admin_allowed(
        self, mock_session_state: dict, analyst_user, monkeypatch
    ) -> None:
        login(analyst_user)
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin"])
        mocks["stop"].assert_called_once()

    # ── authenticated with correct role ────────────────────────────────────

    def test_does_not_call_stop_when_analyst_in_allowed_list(
        self, mock_session_state: dict, analyst_user, monkeypatch
    ) -> None:
        login(analyst_user)
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin", "analyst"])
        mocks["stop"].assert_not_called()

    def test_does_not_call_stop_for_admin_when_admin_allowed(
        self, mock_session_state: dict, db: Session, monkeypatch
    ) -> None:
        admin = create_user(
            db, email="admin@ex.com", name="Admin A",
            password_hash="h", role="admin",
        )
        login(admin)
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin"])
        mocks["stop"].assert_not_called()

    def test_does_not_call_stop_for_viewer_in_allowed_list(
        self, mock_session_state: dict, viewer_user, monkeypatch
    ) -> None:
        login(viewer_user)
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["admin", "analyst", "viewer"])
        mocks["stop"].assert_not_called()

    def test_no_ui_calls_when_access_granted(
        self, mock_session_state: dict, analyst_user, monkeypatch
    ) -> None:
        login(analyst_user)
        mocks = _patch_streamlit_ui(monkeypatch)
        require_role(["analyst"])
        mocks["stop"].assert_not_called()
        mocks["warning"].assert_not_called()
        mocks["error"].assert_not_called()


# ---------------------------------------------------------------------------
# seed_admin
# ---------------------------------------------------------------------------


class TestSeedAdmin:
    def test_creates_admin_user_with_correct_role(self, db: Session) -> None:
        user = seed_admin(
            db, email="admin@example.com",
            name="Super Admin", password="adminpass!",
        )
        assert user.role == "admin"
        assert user.email == "admin@example.com"
        assert user.name == "Super Admin"

    def test_avatar_initials_derived_from_name(self, db: Session) -> None:
        user = seed_admin(
            db, email="admin@example.com",
            name="Sam Admin", password="pw",
        )
        assert user.avatar_initials == "SA"

    def test_stored_password_passes_verify(self, db: Session) -> None:
        seed_admin(
            db, email="admin@example.com",
            name="Admin", password="secret_pw",
        )
        user = get_user_by_email(db, "admin@example.com")
        assert user is not None
        assert verify_password("secret_pw", user.password_hash) is True

    def test_wrong_password_does_not_verify(self, db: Session) -> None:
        seed_admin(
            db, email="admin@example.com",
            name="Admin", password="secret_pw",
        )
        user = get_user_by_email(db, "admin@example.com")
        assert user is not None
        assert verify_password("wrong_password", user.password_hash) is False

    def test_noop_when_email_already_exists(self, db: Session) -> None:
        seed_admin(db, email="admin@example.com", name="Admin", password="pw1")
        seed_admin(db, email="admin@example.com", name="Admin Again", password="pw2")
        users_with_email = [
            u for u in list_users(db) if u.email == "admin@example.com"
        ]
        assert len(users_with_email) == 1

    def test_returns_existing_user_on_second_call(self, db: Session) -> None:
        first = seed_admin(db, email="admin@example.com", name="Admin", password="pw")
        second = seed_admin(db, email="admin@example.com", name="Admin", password="pw2")
        assert first.id == second.id

    def test_seed_does_not_affect_other_users(self, db: Session) -> None:
        other = create_user(
            db, email="other@example.com", name="Other User",
            password_hash="h", role="viewer",
        )
        seed_admin(db, email="admin@example.com", name="Admin", password="pw")
        users = list_users(db)
        assert len(users) == 2
        emails = {u.email for u in users}
        assert "other@example.com" in emails
        # Confirm other user's role is unchanged
        other_refetched = get_user_by_email(db, "other@example.com")
        assert other_refetched.role == "viewer"  # type: ignore[union-attr]
