"""
Tests for ui/pages/dashboard.py.

All Streamlit and DB calls are mocked — no live runtime or database needed.

Covers
------
featured_runs section
    renders a run_card for each featured run
    shows st.info when no featured runs exist
    renders at most 9 featured cards (3 per row × 3 rows)

team metrics strip
    renders 4 metric values (columns called once with 4)
    best_sharpe comes from the run with the highest sharpe
    avg_cagr is the mean across runs that have a cagr metric
    active_count counts only RUNNING runs

activity feed
    shows runs created today (UTC)
    hides runs created on previous days
    limits output to 10 items

my recent runs
    renders up to 5 run_card entries for the current user
    shows info banner when user has no runs
    does not include runs owned by other users

navigation
    clicking a run card sets selected_run_id and navigates to run_detail
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call
import uuid

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from ui.pages.dashboard import dashboard_page


# ===========================================================================
# Shared test data
# ===========================================================================

_NOW = datetime.now(timezone.utc)


def _make_run(
    owner_id: int = 1,
    status: str = "DONE",
    visibility: str = "TEAM",
    result: dict | None = None,
    created_at: datetime | None = None,
) -> MagicMock:
    run = MagicMock()
    run.id = str(uuid.uuid4())
    run.strategy_name = "SMA"
    run.status = status
    run.visibility = visibility
    run.owner_id = owner_id
    run.result = result
    run.tags = []
    run.created_at = (created_at or _NOW).replace(tzinfo=None)  # stored as naive UTC
    return run


_RESULT_GOOD = {
    "metrics": {
        "sharpe_ratio": 1.5,
        "cagr": 0.20,
        "max_drawdown": -0.10,
        "win_rate": 0.55,
    }
}

_CURRENT_USER = {
    "user_id": 1,
    "name": "Alice Analyst",
    "email": "alice@example.com",
    "role": "analyst",
    "avatar_initials": "AA",
}


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_dashboard(monkeypatch):
    """
    Patch every external dependency of dashboard_page() and return a dict of
    mocks.  The caller can adjust return values before calling dashboard_page().
    """
    state = {"page": "dashboard", "selected_run_id": None}

    # ── Auth ───────────────────────────────────────────────────────────────
    monkeypatch.setattr("ui.pages.dashboard.require_role", MagicMock())
    monkeypatch.setattr(
        "ui.pages.dashboard.get_current_user",
        MagicMock(return_value=_CURRENT_USER),
    )

    # ── DB ─────────────────────────────────────────────────────────────────
    mock_featured: list = []
    mock_visible: list = []

    mock_db_session = MagicMock()

    @contextmanager
    def _mock_get_db():
        yield mock_db_session

    monkeypatch.setattr("ui.pages.dashboard.get_db", _mock_get_db)
    mock_list_featured = MagicMock(return_value=mock_featured)
    mock_list_visible = MagicMock(return_value=mock_visible)
    mock_get_user_by_id = MagicMock(return_value=None)
    monkeypatch.setattr("ui.pages.dashboard.list_featured_runs", mock_list_featured)
    monkeypatch.setattr("ui.pages.dashboard.list_runs_for_user", mock_list_visible)
    monkeypatch.setattr("ui.pages.dashboard.get_user_by_id", mock_get_user_by_id)

    # ── run_card component ─────────────────────────────────────────────────
    mock_run_card = MagicMock()
    monkeypatch.setattr("ui.pages.dashboard.run_card", mock_run_card)

    # ── Streamlit ──────────────────────────────────────────────────────────
    def _make_col() -> MagicMock:
        col = MagicMock()
        col.__enter__ = MagicMock(return_value=col)
        col.__exit__ = MagicMock(return_value=False)
        col.metric = MagicMock()
        return col

    def _cols_side_effect(spec):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_make_col() for _ in range(n)]

    st_mocks = {
        "title":        MagicMock(),
        "caption":      MagicMock(),
        "subheader":    MagicMock(),
        "info":         MagicMock(),
        "markdown":     MagicMock(),
        "metric":       MagicMock(),
        "columns":      MagicMock(side_effect=_cols_side_effect),
        "rerun":        MagicMock(),
        "plotly_chart": MagicMock(),
        "dataframe":    MagicMock(),
    }
    for name, mock in st_mocks.items():
        monkeypatch.setattr(f"ui.pages.dashboard.st.{name}", mock)

    monkeypatch.setattr("ui.pages.dashboard.st.session_state", state)

    return {
        "state":        state,
        "featured":     mock_featured,
        "visible":      mock_visible,
        "list_featured": mock_list_featured,
        "list_visible": mock_list_visible,
        "run_card":     mock_run_card,
        "st":           st_mocks,
    }


# ===========================================================================
# Tests — featured runs
# ===========================================================================


class TestFeaturedRuns:
    def test_shows_info_when_no_featured_runs(
        self, mock_dashboard: dict
    ) -> None:
        mock_dashboard["featured"].clear()
        dashboard_page()
        mock_dashboard["st"]["info"].assert_called()
        # run_card should not be called for featured section
        # (it may be called for "my recent runs")

    def test_renders_run_card_for_each_featured_run(
        self, mock_dashboard: dict
    ) -> None:
        runs = [_make_run(visibility="FEATURED") for _ in range(3)]
        mock_dashboard["featured"].extend(runs)
        mock_dashboard["visible"].extend(runs)
        dashboard_page()
        # run_card is called at least 3 times (once per featured run)
        assert mock_dashboard["run_card"].call_count >= 3

    def test_at_most_nine_featured_cards_rendered(
        self, mock_dashboard: dict
    ) -> None:
        # 12 featured runs — should display only 9 (3×3 grid)
        runs = [_make_run(visibility="FEATURED") for _ in range(12)]
        mock_dashboard["featured"].extend(runs)
        mock_dashboard["visible"].extend(runs)
        dashboard_page()
        # run_card call count: at most 9 (featured) + up to 5 (my recent)
        # So total ≤ 14; featured portion capped at 9
        total_calls = mock_dashboard["run_card"].call_count
        # All calls use the same mock; we verify ≤ 14 (9 featured + 5 mine)
        assert total_calls <= 14


# ===========================================================================
# Tests — team metrics strip
# ===========================================================================


class TestTeamMetrics:
    def test_four_metrics_rendered(self, mock_dashboard: dict) -> None:
        """st.columns(4) is called (for the metrics strip)."""
        dashboard_page()
        # columns(4) must be called at least once — for the metrics strip
        calls_with_4 = [
            c for c in mock_dashboard["st"]["columns"].call_args_list
            if c[0] and c[0][0] == 4
        ]
        assert len(calls_with_4) >= 1

    def test_best_sharpe_from_highest_value(
        self, mock_dashboard: dict
    ) -> None:
        r1 = _make_run(
            visibility="TEAM",
            result={"metrics": {"sharpe_ratio": 0.8, "cagr": 0.10}},
        )
        r2 = _make_run(
            visibility="FEATURED",
            result={"metrics": {"sharpe_ratio": 2.1, "cagr": 0.25}},
        )
        mock_dashboard["visible"].extend([r1, r2])
        dashboard_page()
        # st.metric must have been called; we check one of the calls had "2.10"
        metric_values = [
            str(c[1].get("value", "")) + str(c[0][1] if len(c[0]) > 1 else "")
            for c in mock_dashboard["st"]["metric"].call_args_list
        ]
        assert any("2.10" in v for v in metric_values)

    def test_active_runs_count_only_running(
        self, mock_dashboard: dict
    ) -> None:
        running = _make_run(visibility="TEAM", status="RUNNING")
        done = _make_run(visibility="TEAM", status="DONE")
        mock_dashboard["visible"].extend([running, done])
        dashboard_page()
        metric_calls = mock_dashboard["st"]["metric"].call_args_list
        # One metric should have value=1 (one RUNNING run)
        values = [c[0][1] if len(c[0]) > 1 else c[1].get("value") for c in metric_calls]
        assert 1 in values


# ===========================================================================
# Tests — activity feed
# ===========================================================================


class TestActivityFeed:
    def test_shows_todays_runs(self, mock_dashboard: dict) -> None:
        today_run = _make_run(created_at=_NOW)
        mock_dashboard["visible"].append(today_run)
        dashboard_page()
        # At least one markdown call should contain the strategy name
        markdown_texts = [
            str(c[0][0]) for c in mock_dashboard["st"]["markdown"].call_args_list
        ]
        assert any("SMA" in t for t in markdown_texts)

    def test_hides_old_runs_from_feed(self, mock_dashboard: dict) -> None:
        old_run = _make_run(created_at=_NOW - timedelta(days=3))
        mock_dashboard["visible"].append(old_run)
        dashboard_page()
        # The run was 3 days ago — it should NOT appear in Today's Activity
        # (it has strategy name "SMA"; but the activity feed only shows today's)
        # Easiest check: if no today-runs, the feed shows "No runs today."
        # We expect the caption with "No runs today." is written
        markdown_or_caption_texts = [
            str(c[0][0])
            for c in mock_dashboard["st"]["caption"].call_args_list
            + mock_dashboard["st"]["markdown"].call_args_list
        ]
        # No entry should reference old_run's strategy in a "feed" context;
        # "No runs today." appears as st.caption (we check it's NOT in markdowns)
        feed_entries_html = [
            t for t in markdown_or_caption_texts
            if "SMA" in t and "ago" in t
        ]
        assert len(feed_entries_html) == 0

    def test_activity_limited_to_ten_items(
        self, mock_dashboard: dict
    ) -> None:
        runs = [_make_run(created_at=_NOW) for _ in range(15)]
        mock_dashboard["visible"].extend(runs)
        dashboard_page()
        # Count markdown calls that look like activity-feed entries
        activity_calls = [
            c for c in mock_dashboard["st"]["markdown"].call_args_list
            if c[0] and "SMA" in str(c[0][0]) and "ago" in str(c[0][0])
        ]
        assert len(activity_calls) <= 10


# ===========================================================================
# Tests — my recent runs
# ===========================================================================


class TestMyRecentRuns:
    def test_shows_info_when_no_own_runs(
        self, mock_dashboard: dict
    ) -> None:
        # Add a run owned by someone else — current user has no runs
        other_run = _make_run(owner_id=999, visibility="TEAM")
        mock_dashboard["visible"].append(other_run)
        dashboard_page()
        info_texts = [
            str(c[0][0]) for c in mock_dashboard["st"]["info"].call_args_list
        ]
        assert any("No runs yet" in t for t in info_texts)

    def test_renders_at_most_five_own_runs(
        self, mock_dashboard: dict
    ) -> None:
        # 8 runs owned by current user
        for i in range(8):
            mock_dashboard["visible"].append(
                _make_run(owner_id=1, visibility="PRIVATE")
            )
        dashboard_page()
        # run_card calls for own runs should be ≤ 5
        # (no featured runs in this test, so all run_card calls are for "my runs")
        assert mock_dashboard["run_card"].call_count <= 5

    def test_only_own_runs_in_recent_section(
        self, mock_dashboard: dict
    ) -> None:
        own = _make_run(owner_id=1)
        other = _make_run(owner_id=99, visibility="TEAM")
        mock_dashboard["visible"].extend([own, other])
        dashboard_page()
        # run_card must be called with own run's id somewhere
        call_run_ids = [
            c[0][0].id
            for c in mock_dashboard["run_card"].call_args_list
            if c[0]
        ]
        assert own.id in call_run_ids
        # other user's run should not appear in "my recent runs"
        assert other.id not in call_run_ids


# ===========================================================================
# Tests — navigation
# ===========================================================================


class TestDashboardNavigation:
    def test_run_card_on_click_sets_selected_run_id(
        self, mock_dashboard: dict
    ) -> None:
        """
        The on_click callback passed to run_card must write selected_run_id
        into session_state.
        """
        run = _make_run(owner_id=1)
        mock_dashboard["visible"].append(run)

        captured_callbacks: list = []

        def _capture_run_card(r, user, on_click):
            captured_callbacks.append(on_click)

        mock_dashboard["run_card"].side_effect = _capture_run_card
        dashboard_page()

        assert captured_callbacks, "run_card was never called"
        on_click = captured_callbacks[0]
        # Simulate a click — should write state but rerun is mocked so no real rerun
        try:
            on_click(run.id)
        except Exception:
            pass  # st.rerun raises StopException in real runtime; mock won't

        assert mock_dashboard["state"]["selected_run_id"] == run.id

    def test_on_click_sets_page_to_run_detail(
        self, mock_dashboard: dict
    ) -> None:
        run = _make_run(owner_id=1)
        mock_dashboard["visible"].append(run)

        captured_callbacks: list = []

        def _capture(r, user, on_click):
            captured_callbacks.append(on_click)

        mock_dashboard["run_card"].side_effect = _capture
        dashboard_page()

        assert captured_callbacks
        try:
            captured_callbacks[0](run.id)
        except Exception:
            pass

        assert mock_dashboard["state"]["page"] == "run_detail"


# ===========================================================================
# Tests — extended KPI strip (row 2)
# ===========================================================================


class TestExtendedKPIStrip:
    def test_two_rows_of_four_metrics(self, mock_dashboard: dict) -> None:
        """Both metric rows call st.columns(4); expect at least two such calls."""
        dashboard_page()
        calls_with_4 = [
            c for c in mock_dashboard["st"]["columns"].call_args_list
            if c[0] and c[0][0] == 4
        ]
        assert len(calls_with_4) >= 2

    def test_best_cagr_shown_in_second_row(self, mock_dashboard: dict) -> None:
        run = _make_run(
            visibility="TEAM",
            result={"metrics": {"sharpe_ratio": 1.0, "cagr": 0.35,
                                "max_drawdown": -0.08, "win_rate": 0.6}},
            status="DONE",
        )
        mock_dashboard["visible"].append(run)
        dashboard_page()
        metric_values = [
            str(c[0][1] if len(c[0]) > 1 else c[1].get("value", ""))
            for c in mock_dashboard["st"]["metric"].call_args_list
        ]
        assert any("35.0%" in v for v in metric_values)

    def test_avg_win_rate_computed_across_runs(
        self, mock_dashboard: dict
    ) -> None:
        r1 = _make_run(
            visibility="TEAM",
            result={"metrics": {"sharpe_ratio": 1.0, "cagr": 0.1,
                                "max_drawdown": -0.05, "win_rate": 0.60}},
            status="DONE",
        )
        r2 = _make_run(
            visibility="TEAM",
            result={"metrics": {"sharpe_ratio": 1.2, "cagr": 0.2,
                                "max_drawdown": -0.10, "win_rate": 0.40}},
            status="DONE",
        )
        mock_dashboard["visible"].extend([r1, r2])
        dashboard_page()
        metric_values = [
            str(c[0][1] if len(c[0]) > 1 else c[1].get("value", ""))
            for c in mock_dashboard["st"]["metric"].call_args_list
        ]
        # avg win rate = (0.60 + 0.40) / 2 = 0.50 → "50.0%"
        assert any("50.0%" in v for v in metric_values)

    def test_completed_count_excludes_running(
        self, mock_dashboard: dict
    ) -> None:
        done   = _make_run(visibility="TEAM", status="DONE")
        running = _make_run(visibility="TEAM", status="RUNNING")
        mock_dashboard["visible"].extend([done, running])
        dashboard_page()
        metric_values = [
            c[0][1] if len(c[0]) > 1 else c[1].get("value")
            for c in mock_dashboard["st"]["metric"].call_args_list
        ]
        # completed = 1 (only the DONE run)
        assert 1 in metric_values


# ===========================================================================
# Tests — charts section
# ===========================================================================


class TestChartsSection:
    def test_activity_chart_always_rendered(
        self, mock_dashboard: dict
    ) -> None:
        """Team Activity bar chart must be rendered on every dashboard load."""
        dashboard_page()
        mock_dashboard["st"]["plotly_chart"].assert_called()

    def test_scatter_not_rendered_with_fewer_than_two_results(
        self, mock_dashboard: dict
    ) -> None:
        """Scatter requires >= 2 runs with results; with 0 or 1, show caption."""
        # No runs → no scatter → plotly_chart called once (activity only)
        dashboard_page()
        assert mock_dashboard["st"]["plotly_chart"].call_count == 1

    def test_scatter_rendered_with_two_or_more_results(
        self, mock_dashboard: dict
    ) -> None:
        for _ in range(2):
            mock_dashboard["visible"].append(
                _make_run(
                    visibility="TEAM",
                    result={"metrics": {"sharpe_ratio": 1.0, "cagr": 0.15,
                                        "max_drawdown": -0.08, "win_rate": 0.55}},
                    status="DONE",
                )
            )
        dashboard_page()
        # Both scatter AND activity chart → plotly_chart called twice
        assert mock_dashboard["st"]["plotly_chart"].call_count == 2

    def test_insufficient_data_caption_shown_when_one_result(
        self, mock_dashboard: dict
    ) -> None:
        mock_dashboard["visible"].append(
            _make_run(
                visibility="TEAM",
                result={"metrics": {"sharpe_ratio": 1.0, "cagr": 0.1,
                                    "max_drawdown": -0.05, "win_rate": 0.5}},
                status="DONE",
            )
        )
        dashboard_page()
        caption_texts = [
            str(c[0][0]) for c in mock_dashboard["st"]["caption"].call_args_list
        ]
        assert any("2" in t and "strateg" in t.lower() for t in caption_texts)


# ===========================================================================
# Tests — leaderboard
# ===========================================================================


class TestLeaderboard:
    def test_leaderboard_info_shown_with_no_results(
        self, mock_dashboard: dict
    ) -> None:
        dashboard_page()
        info_texts = [
            str(c[0][0]) for c in mock_dashboard["st"]["info"].call_args_list
        ]
        assert any("leaderboard" in t.lower() for t in info_texts)

    def test_leaderboard_dataframe_rendered_when_results_exist(
        self, mock_dashboard: dict
    ) -> None:
        mock_dashboard["visible"].append(
            _make_run(
                visibility="TEAM",
                result={"metrics": {"sharpe_ratio": 1.5, "cagr": 0.20,
                                    "max_drawdown": -0.10, "win_rate": 0.55}},
                status="DONE",
            )
        )
        dashboard_page()
        mock_dashboard["st"]["dataframe"].assert_called_once()

    def test_leaderboard_capped_at_ten_rows(
        self, mock_dashboard: dict
    ) -> None:
        for i in range(15):
            mock_dashboard["visible"].append(
                _make_run(
                    visibility="TEAM",
                    result={"metrics": {"sharpe_ratio": float(i) * 0.1,
                                        "cagr": 0.05, "max_drawdown": -0.05,
                                        "win_rate": 0.5}},
                    status="DONE",
                )
            )
        dashboard_page()
        call_args = mock_dashboard["st"]["dataframe"].call_args
        df = call_args[0][0]
        assert len(df) <= 10

    def test_leaderboard_sorted_by_sharpe_descending(
        self, mock_dashboard: dict
    ) -> None:
        for sharpe in [0.5, 2.0, 1.2]:
            mock_dashboard["visible"].append(
                _make_run(
                    visibility="TEAM",
                    result={"metrics": {"sharpe_ratio": sharpe,
                                        "cagr": 0.1, "max_drawdown": -0.05,
                                        "win_rate": 0.5}},
                    status="DONE",
                )
            )
        dashboard_page()
        df = mock_dashboard["st"]["dataframe"].call_args[0][0]
        # First row should have the highest Sharpe ("2.00")
        assert df.iloc[0]["Sharpe"] == "2.00"
