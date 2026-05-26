"""
Tests for ui/pages/run_detail.py.

All Streamlit and DB calls are mocked — no live runtime or database needed.

Covers
------
guards
    no selected_run_id shows warning and back button
    run not found in DB shows error
    non-admin viewer of a private run sees permission error
    admin can view any private run

header
    title contains the strategy name
    back button navigates to run_history and calls rerun

metrics
    four st.metric calls rendered for sharpe, cagr, max_dd, win_rate
    dash shown for each metric when run has no result
    metric value formatted correctly from run result

equity curve
    equity_chart called when result contains equity_curve key
    equity_chart NOT called when equity_curve is absent from result

parameters
    st.dataframe called with params rows when params exist
    caption shown when params is empty

visibility control
    selectbox shown when user is owner of the run
    admin sees FEATURED in visibility options
    non-owner analyst does NOT see visibility selectbox
    owner with RUNNING status does NOT see visibility selectbox
    update_run_visibility called when Update button clicked with new value

comments
    existing comment body rendered via st.markdown
    no-comments caption shown when comment list is empty
    comment form shown for analyst
    comment form NOT shown for viewer
    create_comment called on valid form submission
    empty comment body does not call create_comment
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock
import uuid

import pytest

from ui.pages.run_detail import run_detail_page


# ===========================================================================
# Shared test data
# ===========================================================================

_ANALYST_USER = {
    "user_id": 1,
    "name": "Alice Analyst",
    "email": "alice@example.com",
    "role": "analyst",
    "avatar_initials": "AA",
}

_ADMIN_USER = {
    "user_id": 2,
    "name": "Bob Admin",
    "email": "bob@example.com",
    "role": "admin",
    "avatar_initials": "BA",
}

_VIEWER_USER = {
    "user_id": 3,
    "name": "Carol Viewer",
    "email": "carol@example.com",
    "role": "viewer",
    "avatar_initials": "CV",
}

_RESULT_FULL = {
    "metrics": {
        "sharpe_ratio": 1.5,
        "cagr": 0.20,
        "max_drawdown": -0.10,
        "win_rate": 0.55,
    },
    "equity_curve": {"2020-01-01": 100_000, "2020-06-01": 115_000, "2020-12-31": 120_000},
}

_RESULT_NO_EQUITY = {
    "metrics": {
        "sharpe_ratio": 1.5,
        "cagr": 0.20,
        "max_drawdown": -0.10,
        "win_rate": 0.55,
    },
}


def _make_run(
    run_id: str = "test-run-id",
    owner_id: int = 1,
    strategy_name: str = "SMA Strategy",
    status: str = "DONE",
    visibility: str = "TEAM",
    result: dict | None = None,
    params: dict | None = None,
    tags: list | None = None,
) -> MagicMock:
    r = MagicMock()
    r.id = run_id
    r.owner_id = owner_id
    r.strategy_name = strategy_name
    r.status = status
    r.visibility = visibility
    r.result = result if result is not None else _RESULT_NO_EQUITY.copy()
    r.params = params if params is not None else {"initial_capital": 100_000.0, "symbols": ["AAPL"]}
    r.tags = tags or []
    return r


def _make_comment(
    comment_id: int = 1,
    author_id: int = 1,
    body: str = "Great strategy!",
) -> MagicMock:
    c = MagicMock()
    c.id = comment_id
    c.author_id = author_id
    c.body = body
    c.created_at = datetime(2024, 1, 15, 10, 30)
    return c


def _make_db_user(user_id: int = 1, name: str = "Alice Analyst") -> MagicMock:
    u = MagicMock()
    u.id = user_id
    u.name = name
    return u


# ===========================================================================
# Fixture
# ===========================================================================


@pytest.fixture
def mock_run_detail(monkeypatch):
    """
    Patch every external dependency of run_detail_page() and return a control
    dict.  Callers can adjust mocks before calling run_detail_page().
    """
    state: dict = {"page": "run_detail", "selected_run_id": "test-run-id"}

    # ── Auth ───────────────────────────────────────────────────────────────
    monkeypatch.setattr("ui.pages.run_detail.require_role", MagicMock())
    monkeypatch.setattr(
        "ui.pages.run_detail.get_current_user",
        MagicMock(return_value=_ANALYST_USER),
    )

    # ── DB ─────────────────────────────────────────────────────────────────
    mock_db = MagicMock()

    @contextmanager
    def _mock_get_db():
        yield mock_db

    monkeypatch.setattr("ui.pages.run_detail.get_db", _mock_get_db)

    mock_run = _make_run()
    mock_get_run = MagicMock(return_value=mock_run)
    monkeypatch.setattr("ui.pages.run_detail.get_run", mock_get_run)

    mock_owner = _make_db_user(user_id=1, name="Alice Analyst")
    mock_get_user_by_id = MagicMock(return_value=mock_owner)
    monkeypatch.setattr("ui.pages.run_detail.get_user_by_id", mock_get_user_by_id)

    mock_comments: list = []
    mock_list_comments = MagicMock(return_value=mock_comments)
    monkeypatch.setattr("ui.pages.run_detail.list_comments_for_run", mock_list_comments)

    mock_db_user = _make_db_user(user_id=1, name="Alice Analyst")
    mock_list_users = MagicMock(return_value=[mock_db_user])
    monkeypatch.setattr("ui.pages.run_detail.list_users", mock_list_users)

    mock_update_vis = MagicMock()
    monkeypatch.setattr("ui.pages.run_detail.update_run_visibility", mock_update_vis)

    mock_create_comment = MagicMock()
    monkeypatch.setattr("ui.pages.run_detail.create_comment", mock_create_comment)

    # ── equity_chart component ─────────────────────────────────────────────
    mock_equity_chart = MagicMock()
    monkeypatch.setattr("ui.pages.run_detail.equity_chart", mock_equity_chart)

    # ── Streamlit mocks ────────────────────────────────────────────────────
    def _make_col() -> MagicMock:
        col = MagicMock()
        col.__enter__ = MagicMock(return_value=col)
        col.__exit__ = MagicMock(return_value=False)
        return col

    def _cols_side_effect(spec) -> list:
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_make_col() for _ in range(n)]

    mock_form_ctx = MagicMock()
    mock_form_ctx.__enter__ = MagicMock(return_value=mock_form_ctx)
    mock_form_ctx.__exit__ = MagicMock(return_value=False)

    st_mocks = {
        "title": MagicMock(),
        "subheader": MagicMock(),
        "caption": MagicMock(),
        "columns": MagicMock(side_effect=_cols_side_effect),
        "markdown": MagicMock(),
        "metric": MagicMock(),
        "dataframe": MagicMock(),
        # Visibility selectbox defaults to same as run.visibility → no update
        "selectbox": MagicMock(return_value="TEAM"),
        "button": MagicMock(return_value=False),
        "warning": MagicMock(),
        "error": MagicMock(),
        "success": MagicMock(),
        "rerun": MagicMock(),
        "info": MagicMock(),
        "form": MagicMock(return_value=mock_form_ctx),
        "text_area": MagicMock(return_value=""),
        "form_submit_button": MagicMock(return_value=False),
    }
    for name, mock in st_mocks.items():
        monkeypatch.setattr(f"ui.pages.run_detail.st.{name}", mock)

    monkeypatch.setattr("ui.pages.run_detail.st.session_state", state)

    return {
        "state": state,
        "run": mock_run,
        "get_run": mock_get_run,
        "owner": mock_owner,
        "comments": mock_comments,
        "list_comments": mock_list_comments,
        "list_users": mock_list_users,
        "update_vis": mock_update_vis,
        "create_comment": mock_create_comment,
        "equity_chart": mock_equity_chart,
        "st": st_mocks,
    }


# ===========================================================================
# Tests — guards
# ===========================================================================


class TestRunDetailGuards:
    def test_no_run_id_shows_warning(self, mock_run_detail: dict) -> None:
        mock_run_detail["state"]["selected_run_id"] = None
        run_detail_page()
        mock_run_detail["st"]["warning"].assert_called_once()

    def test_no_run_id_does_not_fetch_run(self, mock_run_detail: dict) -> None:
        mock_run_detail["state"]["selected_run_id"] = None
        run_detail_page()
        mock_run_detail["get_run"].assert_not_called()

    def test_no_run_id_back_button_navigates(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["state"]["selected_run_id"] = None
        mock_run_detail["st"]["button"].return_value = True
        run_detail_page()
        assert mock_run_detail["state"]["page"] == "run_history"

    def test_run_not_found_shows_error(self, mock_run_detail: dict) -> None:
        mock_run_detail["get_run"].return_value = None
        run_detail_page()
        mock_run_detail["st"]["error"].assert_called_once()
        error_text = str(mock_run_detail["st"]["error"].call_args[0][0])
        assert "not found" in error_text.lower() or "test-run-id" in error_text

    def test_private_run_not_visible_to_other_analyst(
        self, mock_run_detail: dict, monkeypatch
    ) -> None:
        """Non-owner analyst cannot view a PRIVATE run."""
        private_run = _make_run(owner_id=99, visibility="PRIVATE")
        mock_run_detail["get_run"].return_value = private_run
        # Current user is analyst with user_id=1 (not the owner)
        run_detail_page()
        mock_run_detail["st"]["error"].assert_called_once()
        error_text = str(mock_run_detail["st"]["error"].call_args[0][0])
        assert "permission" in error_text.lower()

    def test_admin_can_view_private_run_of_another_user(
        self, mock_run_detail: dict, monkeypatch
    ) -> None:
        """Admin bypasses visibility check."""
        monkeypatch.setattr(
            "ui.pages.run_detail.get_current_user",
            MagicMock(return_value=_ADMIN_USER),
        )
        private_run = _make_run(owner_id=99, visibility="PRIVATE")
        mock_run_detail["get_run"].return_value = private_run
        run_detail_page()
        # No permission error shown
        mock_run_detail["st"]["error"].assert_not_called()
        # Title must be shown
        mock_run_detail["st"]["title"].assert_called_once()

    def test_team_run_visible_to_non_owner_viewer(
        self, mock_run_detail: dict, monkeypatch
    ) -> None:
        """TEAM-visibility run is accessible to any authenticated user."""
        monkeypatch.setattr(
            "ui.pages.run_detail.get_current_user",
            MagicMock(return_value=_VIEWER_USER),
        )
        team_run = _make_run(owner_id=99, visibility="TEAM")
        mock_run_detail["get_run"].return_value = team_run
        run_detail_page()
        mock_run_detail["st"]["error"].assert_not_called()


# ===========================================================================
# Tests — header
# ===========================================================================


class TestRunDetailHeader:
    def test_title_contains_strategy_name(self, mock_run_detail: dict) -> None:
        run_detail_page()
        title_call = mock_run_detail["st"]["title"].call_args[0][0]
        assert "SMA Strategy" in title_call

    def test_back_button_navigates_to_run_history(
        self, mock_run_detail: dict
    ) -> None:
        def _btn(*args, key: str = "", **kwargs):
            return key == "rd_back"

        mock_run_detail["st"]["button"].side_effect = _btn
        run_detail_page()
        assert mock_run_detail["state"]["page"] == "run_history"
        mock_run_detail["st"]["rerun"].assert_called()

    def test_header_markdown_contains_status(
        self, mock_run_detail: dict
    ) -> None:
        run_detail_page()
        markdown_texts = [
            str(c[0][0]) for c in mock_run_detail["st"]["markdown"].call_args_list
        ]
        assert any("DONE" in t for t in markdown_texts)


# ===========================================================================
# Tests — metrics
# ===========================================================================


class TestRunDetailMetrics:
    def test_four_metric_calls_rendered(self, mock_run_detail: dict) -> None:
        run_detail_page()
        # st.metric must be called at least 4 times (once per metric)
        assert mock_run_detail["st"]["metric"].call_count >= 4

    def test_metrics_show_dash_when_no_result(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["run"].result = None
        run_detail_page()
        metric_values = [
            str(c[0][1] if len(c[0]) > 1 else c[1].get("value", ""))
            for c in mock_run_detail["st"]["metric"].call_args_list
        ]
        assert any("—" in v for v in metric_values)

    def test_sharpe_formatted_correctly(self, mock_run_detail: dict) -> None:
        mock_run_detail["run"].result = _RESULT_FULL.copy()
        run_detail_page()
        metric_values = [
            str(c[0][1] if len(c[0]) > 1 else c[1].get("value", ""))
            for c in mock_run_detail["st"]["metric"].call_args_list
        ]
        assert any("1.50" in v for v in metric_values)

    def test_cagr_formatted_as_percentage(self, mock_run_detail: dict) -> None:
        mock_run_detail["run"].result = _RESULT_FULL.copy()
        run_detail_page()
        metric_values = [
            str(c[0][1] if len(c[0]) > 1 else c[1].get("value", ""))
            for c in mock_run_detail["st"]["metric"].call_args_list
        ]
        assert any("20.0%" in v for v in metric_values)


# ===========================================================================
# Tests — equity curve
# ===========================================================================


class TestRunDetailEquityCurve:
    def test_equity_chart_called_when_present(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["run"].result = _RESULT_FULL.copy()
        run_detail_page()
        mock_run_detail["equity_chart"].assert_called_once()

    def test_equity_chart_not_called_when_absent(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["run"].result = _RESULT_NO_EQUITY.copy()
        run_detail_page()
        mock_run_detail["equity_chart"].assert_not_called()

    def test_equity_chart_not_called_when_no_result(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["run"].result = None
        run_detail_page()
        mock_run_detail["equity_chart"].assert_not_called()

    def test_equity_chart_receives_correct_data(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["run"].result = _RESULT_FULL.copy()
        run_detail_page()
        args = mock_run_detail["equity_chart"].call_args[0]
        assert args[0] == _RESULT_FULL["equity_curve"]


# ===========================================================================
# Tests — parameters
# ===========================================================================


class TestRunDetailParams:
    def test_dataframe_shown_when_params_exist(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["run"].params = {"initial_capital": 100_000.0, "symbols": ["AAPL"]}
        run_detail_page()
        mock_run_detail["st"]["dataframe"].assert_called_once()

    def test_caption_shown_when_params_empty(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["run"].params = {}
        run_detail_page()
        mock_run_detail["st"]["dataframe"].assert_not_called()
        caption_texts = [
            str(c[0][0]) for c in mock_run_detail["st"]["caption"].call_args_list
        ]
        assert any("No parameters" in t for t in caption_texts)

    def test_caption_shown_when_params_none(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["run"].params = None
        run_detail_page()
        mock_run_detail["st"]["dataframe"].assert_not_called()


# ===========================================================================
# Tests — visibility control
# ===========================================================================


class TestRunDetailVisibility:
    def test_owner_sees_visibility_selectbox(
        self, mock_run_detail: dict
    ) -> None:
        # run.owner_id == user_id == 1, status == DONE → can_change=True
        run_detail_page()
        selectbox_keys = [
            c.kwargs.get("key") or (c.args[1] if len(c.args) > 1 else None)
            for c in mock_run_detail["st"]["selectbox"].call_args_list
        ]
        assert "rd_visibility" in selectbox_keys

    def test_admin_sees_featured_option(
        self, mock_run_detail: dict, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "ui.pages.run_detail.get_current_user",
            MagicMock(return_value=_ADMIN_USER),
        )
        run_detail_page()
        all_option_sets = [
            c.kwargs.get("options")
            for c in mock_run_detail["st"]["selectbox"].call_args_list
        ]
        assert any(
            opts is not None and "FEATURED" in opts
            for opts in all_option_sets
        )

    def test_analyst_owner_does_not_see_featured_option(
        self, mock_run_detail: dict
    ) -> None:
        run_detail_page()
        all_option_sets = [
            c.kwargs.get("options")
            for c in mock_run_detail["st"]["selectbox"].call_args_list
        ]
        # Options for analyst owner should only be PRIVATE and TEAM
        assert all(
            opts is None or "FEATURED" not in opts
            for opts in all_option_sets
        )

    def test_non_owner_analyst_does_not_see_visibility_selectbox(
        self, mock_run_detail: dict
    ) -> None:
        # Run owned by someone else → non-owner analyst cannot change
        mock_run_detail["run"].owner_id = 99
        run_detail_page()
        selectbox_keys = [
            c.kwargs.get("key")
            for c in mock_run_detail["st"]["selectbox"].call_args_list
        ]
        assert "rd_visibility" not in selectbox_keys

    def test_running_status_hides_visibility_control(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["run"].status = "RUNNING"
        run_detail_page()
        selectbox_keys = [
            c.kwargs.get("key")
            for c in mock_run_detail["st"]["selectbox"].call_args_list
        ]
        assert "rd_visibility" not in selectbox_keys

    def test_update_visibility_called_on_button_click(
        self, mock_run_detail: dict
    ) -> None:
        # Selectbox returns PRIVATE (different from run.visibility=TEAM)
        mock_run_detail["st"]["selectbox"].return_value = "PRIVATE"

        def _btn(*args, key: str = "", **kwargs):
            return key == "rd_update_vis"

        mock_run_detail["st"]["button"].side_effect = _btn
        run_detail_page()
        mock_run_detail["update_vis"].assert_called_once()
        args = mock_run_detail["update_vis"].call_args[0]
        # args: (db, run_id, new_vis)
        assert args[1] == "test-run-id"
        assert args[2] == "PRIVATE"


# ===========================================================================
# Tests — comments
# ===========================================================================


class TestRunDetailComments:
    def test_no_comments_shows_caption(self, mock_run_detail: dict) -> None:
        # comments list is empty by default
        run_detail_page()
        caption_texts = [
            str(c[0][0]) for c in mock_run_detail["st"]["caption"].call_args_list
        ]
        assert any("No comments" in t for t in caption_texts)

    def test_existing_comment_body_rendered(
        self, mock_run_detail: dict
    ) -> None:
        comment = _make_comment(author_id=1, body="Excellent Sharpe ratio!")
        mock_run_detail["comments"].append(comment)
        run_detail_page()
        markdown_texts = [
            str(c[0][0]) for c in mock_run_detail["st"]["markdown"].call_args_list
        ]
        assert any("Excellent Sharpe ratio!" in t for t in markdown_texts)

    def test_comment_author_name_rendered(
        self, mock_run_detail: dict
    ) -> None:
        comment = _make_comment(author_id=1, body="Looks good")
        mock_run_detail["comments"].append(comment)
        run_detail_page()
        markdown_texts = [
            str(c[0][0]) for c in mock_run_detail["st"]["markdown"].call_args_list
        ]
        # author_id=1 → "Alice Analyst" (from mock_list_users)
        assert any("Alice Analyst" in t for t in markdown_texts)

    def test_comment_form_shown_for_analyst(
        self, mock_run_detail: dict
    ) -> None:
        run_detail_page()
        mock_run_detail["st"]["form"].assert_called_once()

    def test_comment_form_not_shown_for_viewer(
        self, mock_run_detail: dict, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "ui.pages.run_detail.get_current_user",
            MagicMock(return_value=_VIEWER_USER),
        )
        run_detail_page()
        mock_run_detail["st"]["form"].assert_not_called()

    def test_create_comment_called_on_valid_submission(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["st"]["form_submit_button"].return_value = True
        mock_run_detail["st"]["text_area"].return_value = "Great risk metrics!"
        run_detail_page()
        mock_run_detail["create_comment"].assert_called_once()
        args = mock_run_detail["create_comment"].call_args[1]
        assert args.get("run_id") == "test-run-id" or (
            mock_run_detail["create_comment"].call_args[0][1] == "test-run-id"
        )

    def test_create_comment_body_stripped(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["st"]["form_submit_button"].return_value = True
        mock_run_detail["st"]["text_area"].return_value = "  nice work  "
        run_detail_page()
        mock_run_detail["create_comment"].assert_called_once()
        call_kwargs = mock_run_detail["create_comment"].call_args[1]
        body_arg = call_kwargs.get(
            "body",
            mock_run_detail["create_comment"].call_args[0][3]
            if len(mock_run_detail["create_comment"].call_args[0]) > 3
            else None,
        )
        assert body_arg == "nice work"

    def test_empty_comment_body_does_not_call_create_comment(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["st"]["form_submit_button"].return_value = True
        mock_run_detail["st"]["text_area"].return_value = "   "  # whitespace only
        run_detail_page()
        mock_run_detail["create_comment"].assert_not_called()

    def test_rerun_called_after_comment_post(
        self, mock_run_detail: dict
    ) -> None:
        mock_run_detail["st"]["form_submit_button"].return_value = True
        mock_run_detail["st"]["text_area"].return_value = "Great trade!"
        run_detail_page()
        mock_run_detail["st"]["rerun"].assert_called()
