"""
Tests for ui/app.py routing and ui/components/sidebar.py.

All tests run without a live Streamlit runtime.  ``st.*`` attributes are
replaced with ``MagicMock`` objects via ``monkeypatch``.

Covers
------
main() — setup
    set_page_config is called exactly once per main() invocation
    init_db is called on every main() invocation (idempotent)
    global CSS is injected via st.markdown on every load

main() — session-state defaults
    "page" defaults to "login" on first load
    "selected_run_id" defaults to None
    "compare_run_ids" defaults to []
    "run_history_filters" defaults to {}
    all four default keys are present after the first call
    existing session-state values are not overwritten

main() — authentication gate
    unauthenticated: login_page() is called
    unauthenticated: st.stop() is called
    unauthenticated: sidebar() is NOT called
    authenticated: sidebar() is called
    authenticated: st.stop() is NOT called
    authenticated: login_page() is NOT called

main() — page routing
    routes to the correct page function for a known page key
    unknown page key → resets session_state["page"] to "dashboard"
    unknown page key → calls st.rerun()

sidebar() — role-based nav visibility
    admin role sees all 7 nav items
    analyst role sees 6 nav items (no Admin Panel)
    viewer role sees 5 nav items (no New Run, no Admin Panel)

sidebar() — logout
    clicking the logout button calls auth.logout()
    clicking the logout button calls st.rerun()

sidebar() — notification badge
    badge IS rendered when unread_count > 0
    badge is NOT rendered when unread_count == 0
    badge label includes the exact unread count
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, call

import pytest

from ui.app import main, PAGES
from ui.components.sidebar import sidebar


# ===========================================================================
# Fixtures — main()
# ===========================================================================


@pytest.fixture
def mock_main(monkeypatch) -> dict:
    """
    Patch all external dependencies of :func:`~ui.app.main`.

    Returns a dict of mocks keyed by short name.  ``state`` is the plain
    dict standing in for ``st.session_state``.
    """
    state: dict = {}

    set_page_config = MagicMock()
    stop = MagicMock()
    rerun = MagicMock()
    markdown = MagicMock()
    init_db_mock = MagicMock()
    login_page_mock = MagicMock()
    sidebar_mock = MagicMock()
    is_authenticated_mock = MagicMock(return_value=False)

    monkeypatch.setattr("ui.app.st.set_page_config", set_page_config)
    monkeypatch.setattr("ui.app.st.session_state", state)
    monkeypatch.setattr("ui.app.st.stop", stop)
    monkeypatch.setattr("ui.app.st.rerun", rerun)
    monkeypatch.setattr("ui.app.st.markdown", markdown)
    monkeypatch.setattr("ui.app.init_db", init_db_mock)
    monkeypatch.setattr("ui.app.login_page", login_page_mock)
    monkeypatch.setattr("ui.app.sidebar", sidebar_mock)
    monkeypatch.setattr("ui.app.is_authenticated", is_authenticated_mock)

    return {
        "state": state,
        "set_page_config": set_page_config,
        "stop": stop,
        "rerun": rerun,
        "markdown": markdown,
        "init_db": init_db_mock,
        "login_page": login_page_mock,
        "sidebar": sidebar_mock,
        "is_authenticated": is_authenticated_mock,
    }


# ===========================================================================
# Fixtures — sidebar()
# ===========================================================================


@pytest.fixture
def mock_sidebar_deps(monkeypatch) -> dict:
    """
    Patch all external dependencies of :func:`~ui.components.sidebar.sidebar`.

    Returns a dict of mocks keyed by short name.  ``sidebar_widget`` is the
    MagicMock standing in for ``st.sidebar``; its ``.button`` attribute
    returns ``False`` (not clicked) by default.
    """
    state: dict = {"page": "dashboard"}

    mock_sb = MagicMock()
    mock_sb.button.return_value = False

    mock_rerun = MagicMock()

    mock_db_session = MagicMock()

    @contextmanager
    def _mock_get_db():
        yield mock_db_session

    mock_unread = MagicMock(return_value=0)
    mock_logout_fn = MagicMock()

    monkeypatch.setattr("ui.components.sidebar.st.sidebar", mock_sb)
    monkeypatch.setattr("ui.components.sidebar.st.session_state", state)
    monkeypatch.setattr("ui.components.sidebar.st.rerun", mock_rerun)
    monkeypatch.setattr("ui.components.sidebar.get_db", _mock_get_db)
    monkeypatch.setattr("ui.components.sidebar.unread_count", mock_unread)
    monkeypatch.setattr("ui.components.sidebar.logout", mock_logout_fn)

    return {
        "sidebar_widget": mock_sb,
        "state": state,
        "rerun": mock_rerun,
        "unread_count": mock_unread,
        "logout": mock_logout_fn,
    }


# ---------------------------------------------------------------------------
# Sidebar helper utilities
# ---------------------------------------------------------------------------


def _patch_current_user(monkeypatch, role: str) -> dict:
    """
    Patch ``get_current_user`` inside ``ui.components.sidebar`` to return a
    synthetic user dict with the given *role*.

    Returns the user dict so tests can introspect it if needed.
    """
    user: dict = {
        "user_id": 1,
        "name": "Test User",
        "email": "test@example.com",
        "role": role,
        "avatar_initials": "TU",
    }
    monkeypatch.setattr(
        "ui.components.sidebar.get_current_user",
        MagicMock(return_value=user),
    )
    return user


def _nav_button_count(button_mock: MagicMock) -> int:
    """
    Count the number of nav-item button calls, excluding the logout button
    (``"🚪 Logout"``) and the notification badge button (``"🔔 N unread"``).

    The nav item labelled ``"🔔 Notifications"`` IS counted — only the
    dynamic badge that contains ``"unread"`` is excluded.
    """
    count = 0
    for c in button_mock.call_args_list:
        label: str = c[0][0] if c[0] else ""
        is_logout = label.startswith("🚪")
        is_badge = "unread" in label  # badge text: "🔔 N unread"
        if not is_logout and not is_badge:
            count += 1
    return count


# ===========================================================================
# Tests — main() setup
# ===========================================================================


class TestMainSetup:
    def test_set_page_config_called_exactly_once(self, mock_main: dict) -> None:
        main()
        mock_main["set_page_config"].assert_called_once()

    def test_set_page_config_has_correct_title(self, mock_main: dict) -> None:
        main()
        _, kwargs = mock_main["set_page_config"].call_args
        assert kwargs.get("page_title") == "Backtester"

    def test_set_page_config_wide_layout(self, mock_main: dict) -> None:
        main()
        _, kwargs = mock_main["set_page_config"].call_args
        assert kwargs.get("layout") == "wide"

    def test_init_db_called_on_every_load(self, mock_main: dict) -> None:
        main()
        main()
        assert mock_main["init_db"].call_count == 2

    def test_global_css_injected_via_markdown(self, mock_main: dict) -> None:
        main()
        mock_main["markdown"].assert_called()
        # At least one call should include <style>
        any_css = any(
            "<style>" in str(c)
            for c in mock_main["markdown"].call_args_list
        )
        assert any_css


# ===========================================================================
# Tests — main() session-state defaults
# ===========================================================================


class TestMainSessionDefaults:
    def test_page_defaults_to_login(self, mock_main: dict) -> None:
        main()
        assert mock_main["state"]["page"] == "login"

    def test_selected_run_id_defaults_to_none(self, mock_main: dict) -> None:
        main()
        assert mock_main["state"]["selected_run_id"] is None

    def test_compare_run_ids_defaults_to_empty_list(self, mock_main: dict) -> None:
        main()
        assert mock_main["state"]["compare_run_ids"] == []

    def test_run_history_filters_defaults_to_empty_dict(self, mock_main: dict) -> None:
        main()
        assert mock_main["state"]["run_history_filters"] == {}

    def test_all_four_default_keys_present(self, mock_main: dict) -> None:
        main()
        state = mock_main["state"]
        assert "page" in state
        assert "selected_run_id" in state
        assert "compare_run_ids" in state
        assert "run_history_filters" in state

    def test_existing_page_value_not_overwritten(self, mock_main: dict) -> None:
        """setdefault must not clobber a page already set in session state."""
        mock_main["state"]["page"] = "run_history"
        mock_main["is_authenticated"].return_value = True
        mock_main["state"]["page"] = "run_history"  # pre-existing

        # Provide the page so routing doesn't reset it
        page_mock = MagicMock()
        original_pages = PAGES.copy()
        original_pages["run_history"] = page_mock
        import ui.app as app_mod
        app_mod.PAGES = original_pages

        try:
            main()
        finally:
            app_mod.PAGES = PAGES  # restore

        assert mock_main["state"]["page"] == "run_history"

    def test_existing_compare_ids_not_overwritten(self, mock_main: dict) -> None:
        mock_main["state"]["compare_run_ids"] = ["abc", "def"]
        main()
        assert mock_main["state"]["compare_run_ids"] == ["abc", "def"]


# ===========================================================================
# Tests — main() authentication gate
# ===========================================================================


class TestMainAuthGate:
    # ── Unauthenticated ────────────────────────────────────────────────────

    def test_unauthenticated_renders_login_page(self, mock_main: dict) -> None:
        mock_main["is_authenticated"].return_value = False
        main()
        mock_main["login_page"].assert_called_once()

    def test_unauthenticated_calls_stop(self, mock_main: dict) -> None:
        mock_main["is_authenticated"].return_value = False
        main()
        mock_main["stop"].assert_called_once()

    def test_unauthenticated_sidebar_not_rendered(self, mock_main: dict) -> None:
        mock_main["is_authenticated"].return_value = False
        main()
        mock_main["sidebar"].assert_not_called()

    # ── Authenticated ──────────────────────────────────────────────────────

    def _auth_setup(self, mock_main: dict, monkeypatch, page_key: str = "dashboard") -> None:
        """Helper: configure mocks for an authenticated user on *page_key*."""
        mock_main["is_authenticated"].return_value = True
        mock_main["state"]["page"] = page_key
        page_fn = MagicMock()
        import ui.app as app_mod
        monkeypatch.setattr(
            "ui.app.PAGES",
            {**app_mod.PAGES, page_key: page_fn},
        )
        return page_fn

    def test_authenticated_renders_sidebar(
        self, mock_main: dict, monkeypatch
    ) -> None:
        self._auth_setup(mock_main, monkeypatch)
        main()
        mock_main["sidebar"].assert_called_once()

    def test_authenticated_stop_not_called(
        self, mock_main: dict, monkeypatch
    ) -> None:
        self._auth_setup(mock_main, monkeypatch)
        main()
        mock_main["stop"].assert_not_called()

    def test_authenticated_login_page_not_called(
        self, mock_main: dict, monkeypatch
    ) -> None:
        self._auth_setup(mock_main, monkeypatch)
        main()
        mock_main["login_page"].assert_not_called()

    def test_authenticated_correct_page_function_called(
        self, mock_main: dict, monkeypatch
    ) -> None:
        page_fn = self._auth_setup(mock_main, monkeypatch, "dashboard")
        main()
        page_fn.assert_called_once()


# ===========================================================================
# Tests — main() page routing
# ===========================================================================


class TestMainRouting:
    def test_routes_to_dashboard(
        self, mock_main: dict, monkeypatch
    ) -> None:
        mock_main["is_authenticated"].return_value = True
        mock_main["state"]["page"] = "dashboard"
        dashboard_fn = MagicMock()
        monkeypatch.setattr(
            "ui.app.PAGES",
            {"dashboard": dashboard_fn},
        )
        main()
        dashboard_fn.assert_called_once()

    def test_routes_to_admin(
        self, mock_main: dict, monkeypatch
    ) -> None:
        mock_main["is_authenticated"].return_value = True
        mock_main["state"]["page"] = "admin"
        admin_fn = MagicMock()
        monkeypatch.setattr("ui.app.PAGES", {"admin": admin_fn})
        main()
        admin_fn.assert_called_once()

    def test_unknown_page_resets_to_dashboard(
        self, mock_main: dict
    ) -> None:
        mock_main["is_authenticated"].return_value = True
        mock_main["state"]["page"] = "this_page_does_not_exist"
        main()
        assert mock_main["state"]["page"] == "dashboard"

    def test_unknown_page_calls_rerun(self, mock_main: dict) -> None:
        mock_main["is_authenticated"].return_value = True
        mock_main["state"]["page"] = "this_page_does_not_exist"
        main()
        mock_main["rerun"].assert_called_once()

    def test_unknown_page_does_not_call_any_page_fn(
        self, mock_main: dict, monkeypatch
    ) -> None:
        mock_main["is_authenticated"].return_value = True
        mock_main["state"]["page"] = "ghost_page"
        page_fn = MagicMock()
        monkeypatch.setattr(
            "ui.app.PAGES",
            {"dashboard": page_fn, "login": MagicMock()},
        )
        main()
        # The ghost key isn't in PAGES so no page fn should be called
        page_fn.assert_not_called()


# ===========================================================================
# Tests — sidebar() role-based nav visibility
# ===========================================================================


class TestSidebarNavItems:
    def test_admin_sees_all_seven_nav_items(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "admin")
        sidebar()
        count = _nav_button_count(mock_sidebar_deps["sidebar_widget"].button)
        assert count == 7

    def test_analyst_sees_six_nav_items(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "analyst")
        sidebar()
        count = _nav_button_count(mock_sidebar_deps["sidebar_widget"].button)
        assert count == 6

    def test_viewer_sees_five_nav_items(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "viewer")
        sidebar()
        count = _nav_button_count(mock_sidebar_deps["sidebar_widget"].button)
        assert count == 5

    def test_viewer_cannot_see_new_run(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "viewer")
        sidebar()
        button_mock = mock_sidebar_deps["sidebar_widget"].button
        labels = [c[0][0] for c in button_mock.call_args_list if c[0]]
        assert not any("New Run" in lbl for lbl in labels)

    def test_viewer_cannot_see_admin_panel(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "viewer")
        sidebar()
        button_mock = mock_sidebar_deps["sidebar_widget"].button
        labels = [c[0][0] for c in button_mock.call_args_list if c[0]]
        assert not any("Admin" in lbl for lbl in labels)

    def test_analyst_cannot_see_admin_panel(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "analyst")
        sidebar()
        button_mock = mock_sidebar_deps["sidebar_widget"].button
        labels = [c[0][0] for c in button_mock.call_args_list if c[0]]
        assert not any("Admin Panel" in lbl for lbl in labels)

    def test_admin_sees_admin_panel(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "admin")
        sidebar()
        button_mock = mock_sidebar_deps["sidebar_widget"].button
        labels = [c[0][0] for c in button_mock.call_args_list if c[0]]
        assert any("Admin Panel" in lbl for lbl in labels)

    def test_nav_button_click_sets_page_and_reruns(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        """Clicking the Dashboard nav button writes page and calls rerun."""
        _patch_current_user(monkeypatch, "admin")

        def _button_side_effect(label: str, key: str = "", **kwargs):  # type: ignore[override]
            return key == "nav_dashboard"

        mock_sidebar_deps["sidebar_widget"].button.side_effect = _button_side_effect
        sidebar()

        assert mock_sidebar_deps["state"]["page"] == "dashboard"
        mock_sidebar_deps["rerun"].assert_called()


# ===========================================================================
# Tests — sidebar() logout
# ===========================================================================


class TestSidebarLogout:
    def test_logout_button_calls_auth_logout(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "analyst")

        def _button_side_effect(label: str, key: str = "", **kwargs):  # type: ignore[override]
            return key == "logout_btn"

        mock_sidebar_deps["sidebar_widget"].button.side_effect = _button_side_effect
        sidebar()
        mock_sidebar_deps["logout"].assert_called_once()

    def test_logout_button_calls_rerun(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "analyst")

        def _button_side_effect(label: str, key: str = "", **kwargs):  # type: ignore[override]
            return key == "logout_btn"

        mock_sidebar_deps["sidebar_widget"].button.side_effect = _button_side_effect
        sidebar()
        mock_sidebar_deps["rerun"].assert_called()

    def test_no_logout_when_button_not_clicked(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "analyst")
        # default: button always returns False
        sidebar()
        mock_sidebar_deps["logout"].assert_not_called()


# ===========================================================================
# Tests — sidebar() notification badge
# ===========================================================================


class TestSidebarNotificationBadge:
    def test_badge_rendered_when_unread_positive(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "analyst")
        mock_sidebar_deps["unread_count"].return_value = 5
        sidebar()
        button_mock = mock_sidebar_deps["sidebar_widget"].button
        # Badge label contains "unread"; the nav item "🔔 Notifications" does not
        badge_calls = [
            c for c in button_mock.call_args_list
            if c[0] and "unread" in c[0][0]
        ]
        assert len(badge_calls) == 1

    def test_badge_not_rendered_when_unread_zero(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "analyst")
        # unread_count fixture already returns 0
        sidebar()
        button_mock = mock_sidebar_deps["sidebar_widget"].button
        badge_calls = [
            c for c in button_mock.call_args_list
            if c[0] and "unread" in c[0][0]
        ]
        assert len(badge_calls) == 0

    def test_badge_label_contains_exact_count(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "admin")
        mock_sidebar_deps["unread_count"].return_value = 12
        sidebar()
        button_mock = mock_sidebar_deps["sidebar_widget"].button
        badge_labels = [
            c[0][0] for c in button_mock.call_args_list
            if c[0] and "unread" in c[0][0]
        ]
        assert badge_labels, "Expected a badge button to be rendered"
        assert "12" in badge_labels[0]

    def test_badge_click_navigates_to_notifications(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        _patch_current_user(monkeypatch, "analyst")
        mock_sidebar_deps["unread_count"].return_value = 3

        def _button_side_effect(label: str, key: str = "", **kwargs):  # type: ignore[override]
            return key == "notif_badge"

        mock_sidebar_deps["sidebar_widget"].button.side_effect = _button_side_effect
        sidebar()

        assert mock_sidebar_deps["state"]["page"] == "notifications"
        mock_sidebar_deps["rerun"].assert_called()

    def test_unread_count_queried_for_current_user(
        self, mock_sidebar_deps: dict, monkeypatch
    ) -> None:
        """unread_count must be called with the current user's ID."""
        user = _patch_current_user(monkeypatch, "admin")
        sidebar()
        mock_sidebar_deps["unread_count"].assert_called_once()
        _, pos_args = mock_sidebar_deps["unread_count"].call_args
        # Second positional arg is user_id
        called_user_id = mock_sidebar_deps["unread_count"].call_args[0][1]
        assert called_user_id == user["user_id"]
