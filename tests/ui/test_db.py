"""
Tests for ui/db.py and ui/models.py.

Every test uses an in-memory SQLite database that is created fresh per test
via the ``db`` fixture — no filesystem touches, no shared state between tests.

Covers
------
User CRUD
    create_user → retrievable by email and by id
    update_user_role → role change persisted
    list_users → returns all rows

Run CRUD
    create_run → UUID assigned, status defaults to "RUNNING"
    update_run_status DONE → result dict persisted
    update_run_status FAILED → error_message persisted
    list_runs_for_user → admin / analyst / viewer visibility rules
    list_featured_runs → only FEATURED visibility
    update_run_visibility → visibility change persisted

Comments
    create_comment + list_comments_for_run → correct chronological ordering

Strategies
    create_strategy defaults to DRAFT
    update_strategy_status → APPROVED persisted
    list_strategies(status=...) filters correctly

Notifications
    create_notification + list_notifications(unread_only=True)
    mark_notification_read → read becomes True
    mark_all_notifications_read → correct count returned, all marked
    unread_count → accurate before and after marking

Pydantic schemas
    UserSchema does NOT expose password_hash
    RunSchema round-trips a JSON result dict
    from_attributes=True works on real ORM objects
    CommentSchema, StrategySchema, NotificationSchema validate ORM objects
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ui.db import (
    Base,
    Comment,
    Notification,
    Run,
    Strategy,
    User,
    create_comment,
    create_notification,
    create_run,
    create_strategy,
    create_user,
    get_run,
    get_user_by_email,
    get_user_by_id,
    list_comments_for_run,
    list_featured_runs,
    list_notifications,
    list_runs_for_user,
    list_strategies,
    list_users,
    mark_all_notifications_read,
    mark_notification_read,
    unread_count,
    update_run_status,
    update_run_visibility,
    update_strategy_status,
    update_user_role,
)
from ui.models import (
    CommentSchema,
    NotificationSchema,
    RunSchema,
    StrategySchema,
    UserSchema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> Session:
    """
    Yield a fresh in-memory SQLite session per test.

    Tables are created before the test and the session is closed after.
    No filesystem I/O occurs.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Helpers — fast object construction inside tests
# ---------------------------------------------------------------------------


def _make_user(
    db: Session,
    email: str = "alice@example.com",
    name: str = "Alice Wonderland",
    role: str = "analyst",
) -> User:
    return create_user(
        db, email=email, name=name,
        password_hash="hashed_pw", role=role,
    )


def _make_run(
    db: Session,
    owner_id: int,
    visibility: str = "PRIVATE",
    tags: list | None = None,
) -> Run:
    return create_run(
        db,
        owner_id=owner_id,
        strategy_name="SMAStrategy",
        params={"lookback": 20},
        visibility=visibility,
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


class TestUserCRUD:
    def test_create_user_retrievable_by_email(self, db: Session) -> None:
        user = _make_user(db, email="bob@example.com")
        found = get_user_by_email(db, "bob@example.com")
        assert found is not None
        assert found.id == user.id
        assert found.email == "bob@example.com"

    def test_create_user_retrievable_by_id(self, db: Session) -> None:
        user = _make_user(db, email="carol@example.com")
        found = get_user_by_id(db, user.id)
        assert found is not None
        assert found.email == "carol@example.com"

    def test_get_user_by_email_returns_none_when_missing(self, db: Session) -> None:
        assert get_user_by_email(db, "ghost@example.com") is None

    def test_get_user_by_id_returns_none_when_missing(self, db: Session) -> None:
        assert get_user_by_id(db, 9999) is None

    def test_avatar_initials_derived_from_name(self, db: Session) -> None:
        user = create_user(
            db, email="d@example.com", name="David Eugene Bowie",
            password_hash="x", role="viewer",
        )
        assert user.avatar_initials == "DEB"

    def test_avatar_initials_capped_at_four(self, db: Session) -> None:
        user = create_user(
            db, email="e@example.com", name="Anna Beta Gamma Delta Epsilon",
            password_hash="x", role="viewer",
        )
        assert len(user.avatar_initials) <= 4

    def test_update_user_role_persisted(self, db: Session) -> None:
        user = _make_user(db, role="analyst")
        assert user.role == "analyst"
        updated = update_user_role(db, user.id, "admin")
        assert updated.role == "admin"
        # Re-fetch to confirm persistence
        refetched = get_user_by_id(db, user.id)
        assert refetched.role == "admin"

    def test_update_role_nonexistent_user_raises(self, db: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            update_user_role(db, 9999, "admin")

    def test_list_users_returns_all(self, db: Session) -> None:
        _make_user(db, email="u1@example.com")
        _make_user(db, email="u2@example.com")
        _make_user(db, email="u3@example.com")
        users = list_users(db)
        assert len(users) == 3

    def test_list_users_empty_db(self, db: Session) -> None:
        assert list_users(db) == []


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------


class TestRunCRUD:
    def test_create_run_uuid_assigned(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        assert isinstance(run.id, str)
        assert len(run.id) == 36  # standard UUID string length

    def test_create_run_status_defaults_to_running(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        assert run.status == "RUNNING"

    def test_create_run_result_is_none_initially(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        assert run.result is None

    def test_get_run_returns_correct_row(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        found = get_run(db, run.id)
        assert found is not None
        assert found.id == run.id
        assert found.strategy_name == "SMAStrategy"

    def test_get_run_returns_none_when_missing(self, db: Session) -> None:
        assert get_run(db, "no-such-uuid") is None

    def test_update_run_status_done_with_result(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        result_dict = {"total_return": 0.15, "sharpe_ratio": 1.4}
        updated = update_run_status(
            db, run.id, "DONE", result=result_dict
        )
        assert updated.status == "DONE"
        assert updated.result == result_dict
        assert updated.error_message is None
        # Re-fetch to confirm DB persistence
        refetched = get_run(db, run.id)
        assert refetched.result["sharpe_ratio"] == pytest.approx(1.4)

    def test_update_run_status_failed_with_error_message(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        updated = update_run_status(
            db, run.id, "FAILED", error_message="Division by zero in strategy"
        )
        assert updated.status == "FAILED"
        assert updated.result is None  # no result on failure
        assert updated.error_message == "Division by zero in strategy"
        refetched = get_run(db, run.id)
        assert refetched.error_message == "Division by zero in strategy"

    def test_update_run_status_nonexistent_raises(self, db: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            update_run_status(db, "ghost-uuid", "DONE")

    def test_update_run_visibility(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id, visibility="PRIVATE")
        updated = update_run_visibility(db, run.id, "FEATURED")
        assert updated.visibility == "FEATURED"
        refetched = get_run(db, run.id)
        assert refetched.visibility == "FEATURED"

    def test_update_run_visibility_nonexistent_raises(self, db: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            update_run_visibility(db, "ghost-uuid", "TEAM")

    def test_list_featured_runs_only_featured(self, db: Session) -> None:
        user = _make_user(db)
        _make_run(db, owner_id=user.id, visibility="PRIVATE")
        _make_run(db, owner_id=user.id, visibility="TEAM")
        featured_run = _make_run(db, owner_id=user.id, visibility="FEATURED")
        results = list_featured_runs(db)
        assert len(results) == 1
        assert results[0].id == featured_run.id

    def test_list_featured_runs_empty(self, db: Session) -> None:
        user = _make_user(db)
        _make_run(db, owner_id=user.id, visibility="PRIVATE")
        assert list_featured_runs(db) == []


# ---------------------------------------------------------------------------
# Visibility rules for list_runs_for_user
# ---------------------------------------------------------------------------


class TestRunVisibility:
    """
    Fixture layout
    --------------
    admin_user  → run_admin_private (PRIVATE)
    analyst     → run_analyst_private (PRIVATE), run_analyst_team (TEAM)
    viewer      → run_viewer_featured (FEATURED)
    """

    @pytest.fixture
    def visibility_setup(self, db: Session):
        admin_user = create_user(
            db, email="admin@ex.com", name="Admin A",
            password_hash="h", role="admin",
        )
        analyst = create_user(
            db, email="analyst@ex.com", name="Analyst B",
            password_hash="h", role="analyst",
        )
        viewer = create_user(
            db, email="viewer@ex.com", name="Viewer C",
            password_hash="h", role="viewer",
        )

        run_admin_private = _make_run(db, owner_id=admin_user.id, visibility="PRIVATE")
        run_analyst_private = _make_run(db, owner_id=analyst.id, visibility="PRIVATE")
        run_analyst_team = _make_run(db, owner_id=analyst.id, visibility="TEAM")
        run_viewer_featured = _make_run(db, owner_id=viewer.id, visibility="FEATURED")

        return {
            "admin": admin_user,
            "analyst": analyst,
            "viewer": viewer,
            "run_admin_private": run_admin_private,
            "run_analyst_private": run_analyst_private,
            "run_analyst_team": run_analyst_team,
            "run_viewer_featured": run_viewer_featured,
        }

    def test_admin_sees_all_runs(self, db: Session, visibility_setup: dict) -> None:
        s = visibility_setup
        runs = list_runs_for_user(db, s["admin"].id, role="admin")
        run_ids = {r.id for r in runs}
        assert run_ids == {
            s["run_admin_private"].id,
            s["run_analyst_private"].id,
            s["run_analyst_team"].id,
            s["run_viewer_featured"].id,
        }

    def test_analyst_sees_own_plus_team_and_featured(
        self, db: Session, visibility_setup: dict
    ) -> None:
        s = visibility_setup
        runs = list_runs_for_user(db, s["analyst"].id, role="analyst")
        run_ids = {r.id for r in runs}
        # Own: run_analyst_private + run_analyst_team
        # TEAM: run_analyst_team (already included)
        # FEATURED: run_viewer_featured
        # NOT visible: run_admin_private (admin's PRIVATE)
        assert s["run_analyst_private"].id in run_ids
        assert s["run_analyst_team"].id in run_ids
        assert s["run_viewer_featured"].id in run_ids
        assert s["run_admin_private"].id not in run_ids

    def test_viewer_sees_own_plus_team_and_featured(
        self, db: Session, visibility_setup: dict
    ) -> None:
        s = visibility_setup
        runs = list_runs_for_user(db, s["viewer"].id, role="viewer")
        run_ids = {r.id for r in runs}
        # Own: run_viewer_featured
        # TEAM: run_analyst_team
        # FEATURED: run_viewer_featured (already included)
        # NOT visible: run_admin_private (PRIVATE), run_analyst_private (PRIVATE)
        assert s["run_viewer_featured"].id in run_ids
        assert s["run_analyst_team"].id in run_ids
        assert s["run_admin_private"].id not in run_ids
        assert s["run_analyst_private"].id not in run_ids

    def test_private_run_hidden_from_non_owner_non_admin(
        self, db: Session, visibility_setup: dict
    ) -> None:
        s = visibility_setup
        # analyst cannot see admin's PRIVATE run
        runs = list_runs_for_user(db, s["analyst"].id, role="analyst")
        run_ids = {r.id for r in runs}
        assert s["run_admin_private"].id not in run_ids

        # viewer cannot see analyst's PRIVATE run
        runs = list_runs_for_user(db, s["viewer"].id, role="viewer")
        run_ids = {r.id for r in runs}
        assert s["run_analyst_private"].id not in run_ids


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


class TestComments:
    def test_create_and_list_comment(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        comment = create_comment(
            db, run_id=run.id, author_id=user.id,
            body="Great result!", mentions=[],
        )
        comments = list_comments_for_run(db, run.id)
        assert len(comments) == 1
        assert comments[0].id == comment.id
        assert comments[0].body == "Great result!"

    def test_comments_ordered_by_created_at(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        c1 = create_comment(db, run_id=run.id, author_id=user.id, body="first", mentions=[])
        c2 = create_comment(db, run_id=run.id, author_id=user.id, body="second", mentions=[])
        c3 = create_comment(db, run_id=run.id, author_id=user.id, body="third", mentions=[])
        comments = list_comments_for_run(db, run.id)
        assert [c.id for c in comments] == [c1.id, c2.id, c3.id]

    def test_comment_mentions_stored(self, db: Session) -> None:
        user = _make_user(db)
        other = _make_user(db, email="other@example.com", name="Other User")
        run = _make_run(db, owner_id=user.id)
        comment = create_comment(
            db, run_id=run.id, author_id=user.id,
            body="Hey @Other!", mentions=[other.id],
        )
        fetched = list_comments_for_run(db, run.id)[0]
        assert fetched.mentions == [other.id]

    def test_list_comments_for_run_empty(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        assert list_comments_for_run(db, run.id) == []

    def test_comments_isolated_by_run(self, db: Session) -> None:
        user = _make_user(db)
        run1 = _make_run(db, owner_id=user.id)
        run2 = _make_run(db, owner_id=user.id)
        create_comment(db, run_id=run1.id, author_id=user.id, body="for run1", mentions=[])
        create_comment(db, run_id=run1.id, author_id=user.id, body="also run1", mentions=[])
        create_comment(db, run_id=run2.id, author_id=user.id, body="for run2", mentions=[])
        assert len(list_comments_for_run(db, run1.id)) == 2
        assert len(list_comments_for_run(db, run2.id)) == 1


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class TestStrategies:
    def test_create_strategy_defaults_to_draft(self, db: Session) -> None:
        user = _make_user(db)
        strat = create_strategy(
            db, name="MySMA", description="SMA crossover",
            code="class MySMA: pass", author_id=user.id,
        )
        assert strat.status == "DRAFT"
        assert strat.id is not None

    def test_update_strategy_status_to_approved(self, db: Session) -> None:
        user = _make_user(db)
        strat = create_strategy(
            db, name="MyCross", description="Crossover",
            code="pass", author_id=user.id,
        )
        updated = update_strategy_status(db, strat.id, "APPROVED")
        assert updated.status == "APPROVED"

    def test_update_strategy_status_persisted(self, db: Session) -> None:
        user = _make_user(db)
        strat = create_strategy(
            db, name="MyMom", description="Momentum",
            code="pass", author_id=user.id,
        )
        update_strategy_status(db, strat.id, "PENDING")
        strategies = list_strategies(db)
        assert strategies[0].status == "PENDING"

    def test_update_strategy_nonexistent_raises(self, db: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            update_strategy_status(db, 9999, "APPROVED")

    def test_list_strategies_no_filter(self, db: Session) -> None:
        user = _make_user(db)
        create_strategy(db, name="A", description="d", code="c", author_id=user.id)
        create_strategy(db, name="B", description="d", code="c", author_id=user.id)
        assert len(list_strategies(db)) == 2

    def test_list_strategies_filters_by_status(self, db: Session) -> None:
        user = _make_user(db)
        s1 = create_strategy(db, name="S1", description="d", code="c", author_id=user.id)
        s2 = create_strategy(db, name="S2", description="d", code="c", author_id=user.id)
        s3 = create_strategy(db, name="S3", description="d", code="c", author_id=user.id)

        update_strategy_status(db, s1.id, "APPROVED")
        update_strategy_status(db, s2.id, "APPROVED")
        # s3 stays DRAFT

        approved = list_strategies(db, status="APPROVED")
        assert len(approved) == 2
        approved_ids = {s.id for s in approved}
        assert s1.id in approved_ids
        assert s2.id in approved_ids
        assert s3.id not in approved_ids

    def test_list_strategies_filter_draft(self, db: Session) -> None:
        user = _make_user(db)
        s1 = create_strategy(db, name="D1", description="d", code="c", author_id=user.id)
        s2 = create_strategy(db, name="D2", description="d", code="c", author_id=user.id)
        update_strategy_status(db, s2.id, "PENDING")

        drafts = list_strategies(db, status="DRAFT")
        assert len(drafts) == 1
        assert drafts[0].id == s1.id

    def test_list_strategies_filter_returns_empty_when_none_match(
        self, db: Session
    ) -> None:
        user = _make_user(db)
        create_strategy(db, name="X", description="d", code="c", author_id=user.id)
        assert list_strategies(db, status="APPROVED") == []


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


class TestNotifications:
    def test_create_notification_stored(self, db: Session) -> None:
        user = _make_user(db)
        n = create_notification(
            db, recipient_id=user.id, type="RUN_DONE",
            payload={"run_id": "abc-123"},
        )
        assert n.id is not None
        assert n.read is False
        assert n.type == "RUN_DONE"
        assert n.payload == {"run_id": "abc-123"}

    def test_list_notifications_all(self, db: Session) -> None:
        user = _make_user(db)
        create_notification(db, recipient_id=user.id, type="RUN_DONE", payload={})
        create_notification(db, recipient_id=user.id, type="COMMENT", payload={})
        notifications = list_notifications(db, user.id)
        assert len(notifications) == 2

    def test_list_notifications_unread_only(self, db: Session) -> None:
        user = _make_user(db)
        n1 = create_notification(db, recipient_id=user.id, type="RUN_DONE", payload={})
        n2 = create_notification(db, recipient_id=user.id, type="RUN_FAILED", payload={})
        # Mark n1 read
        mark_notification_read(db, n1.id)
        unread = list_notifications(db, user.id, unread_only=True)
        assert len(unread) == 1
        assert unread[0].id == n2.id

    def test_mark_notification_read(self, db: Session) -> None:
        user = _make_user(db)
        n = create_notification(db, recipient_id=user.id, type="COMMENT", payload={})
        assert n.read is False
        updated = mark_notification_read(db, n.id)
        assert updated.read is True
        # Re-fetch to confirm persistence
        notifications = list_notifications(db, user.id)
        assert notifications[0].read is True

    def test_mark_notification_read_nonexistent_raises(self, db: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            mark_notification_read(db, 9999)

    def test_mark_all_notifications_read_returns_count(self, db: Session) -> None:
        user = _make_user(db)
        create_notification(db, recipient_id=user.id, type="RUN_DONE", payload={})
        create_notification(db, recipient_id=user.id, type="COMMENT", payload={})
        create_notification(db, recipient_id=user.id, type="RUN_FAILED", payload={})
        count = mark_all_notifications_read(db, user.id)
        assert count == 3

    def test_mark_all_notifications_read_all_marked(self, db: Session) -> None:
        user = _make_user(db)
        create_notification(db, recipient_id=user.id, type="RUN_DONE", payload={})
        create_notification(db, recipient_id=user.id, type="COMMENT", payload={})
        mark_all_notifications_read(db, user.id)
        remaining_unread = list_notifications(db, user.id, unread_only=True)
        assert remaining_unread == []

    def test_mark_all_read_only_counts_unread(self, db: Session) -> None:
        user = _make_user(db)
        n1 = create_notification(db, recipient_id=user.id, type="RUN_DONE", payload={})
        create_notification(db, recipient_id=user.id, type="COMMENT", payload={})
        mark_notification_read(db, n1.id)  # mark one read first
        count = mark_all_notifications_read(db, user.id)
        assert count == 1  # only the remaining unread one

    def test_mark_all_read_returns_zero_when_already_all_read(
        self, db: Session
    ) -> None:
        user = _make_user(db)
        n = create_notification(db, recipient_id=user.id, type="RUN_DONE", payload={})
        mark_notification_read(db, n.id)
        count = mark_all_notifications_read(db, user.id)
        assert count == 0

    def test_unread_count_before_marking(self, db: Session) -> None:
        user = _make_user(db)
        create_notification(db, recipient_id=user.id, type="RUN_DONE", payload={})
        create_notification(db, recipient_id=user.id, type="COMMENT", payload={})
        assert unread_count(db, user.id) == 2

    def test_unread_count_after_marking_all_read(self, db: Session) -> None:
        user = _make_user(db)
        create_notification(db, recipient_id=user.id, type="RUN_DONE", payload={})
        create_notification(db, recipient_id=user.id, type="COMMENT", payload={})
        mark_all_notifications_read(db, user.id)
        assert unread_count(db, user.id) == 0

    def test_unread_count_zero_with_no_notifications(self, db: Session) -> None:
        user = _make_user(db)
        assert unread_count(db, user.id) == 0

    def test_notifications_isolated_by_recipient(self, db: Session) -> None:
        alice = _make_user(db, email="alice@ex.com", name="Alice A")
        bob = _make_user(db, email="bob@ex.com", name="Bob B")
        create_notification(db, recipient_id=alice.id, type="COMMENT", payload={})
        create_notification(db, recipient_id=alice.id, type="RUN_DONE", payload={})
        create_notification(db, recipient_id=bob.id, type="COMMENT", payload={})

        assert unread_count(db, alice.id) == 2
        assert unread_count(db, bob.id) == 1
        mark_all_notifications_read(db, alice.id)
        assert unread_count(db, alice.id) == 0
        assert unread_count(db, bob.id) == 1  # bob's unread unaffected


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TestPydanticSchemas:
    def test_user_schema_excludes_password_hash(self, db: Session) -> None:
        user = _make_user(db, email="schema@example.com")
        schema = UserSchema.model_validate(user)
        assert not hasattr(schema, "password_hash")
        schema_dict = schema.model_dump()
        assert "password_hash" not in schema_dict

    def test_user_schema_from_orm_object(self, db: Session) -> None:
        user = _make_user(db, email="orm@example.com", name="ORM User", role="viewer")
        schema = UserSchema.model_validate(user)
        assert schema.email == "orm@example.com"
        assert schema.name == "ORM User"
        assert schema.role == "viewer"
        assert schema.avatar_initials == "OU"

    def test_run_schema_round_trips_result_dict(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        result_dict = {
            "total_return": 0.25,
            "sharpe_ratio": 1.8,
            "metrics": {"num_trades": 10},
        }
        update_run_status(db, run.id, "DONE", result=result_dict)
        run = get_run(db, run.id)
        schema = RunSchema.model_validate(run)
        assert schema.result == result_dict
        assert schema.status == "DONE"
        assert schema.result["sharpe_ratio"] == pytest.approx(1.8)

    def test_run_schema_result_is_none_when_running(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        schema = RunSchema.model_validate(run)
        assert schema.result is None
        assert schema.status == "RUNNING"

    def test_run_schema_error_message_on_failed(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        update_run_status(
            db, run.id, "FAILED", error_message="strategy crashed"
        )
        run = get_run(db, run.id)
        schema = RunSchema.model_validate(run)
        assert schema.status == "FAILED"
        assert schema.error_message == "strategy crashed"

    def test_comment_schema_from_orm_object(self, db: Session) -> None:
        user = _make_user(db)
        run = _make_run(db, owner_id=user.id)
        comment = create_comment(
            db, run_id=run.id, author_id=user.id,
            body="Test body", mentions=[42, 99],
        )
        schema = CommentSchema.model_validate(comment)
        assert schema.body == "Test body"
        assert schema.mentions == [42, 99]
        assert schema.run_id == run.id

    def test_strategy_schema_from_orm_object(self, db: Session) -> None:
        user = _make_user(db)
        strat = create_strategy(
            db, name="TestStrat", description="A test",
            code="class TestStrat: pass", author_id=user.id,
        )
        schema = StrategySchema.model_validate(strat)
        assert schema.name == "TestStrat"
        assert schema.status == "DRAFT"
        assert schema.author_id == user.id

    def test_notification_schema_from_orm_object(self, db: Session) -> None:
        user = _make_user(db)
        n = create_notification(
            db, recipient_id=user.id, type="RUN_DONE",
            payload={"run_id": "xyz-789"},
        )
        schema = NotificationSchema.model_validate(n)
        assert schema.type == "RUN_DONE"
        assert schema.payload == {"run_id": "xyz-789"}
        assert schema.read is False

    def test_notification_schema_read_true_after_mark(self, db: Session) -> None:
        user = _make_user(db)
        n = create_notification(
            db, recipient_id=user.id, type="COMMENT", payload={}
        )
        mark_notification_read(db, n.id)
        # Re-fetch from DB
        notifications = list_notifications(db, user.id)
        schema = NotificationSchema.model_validate(notifications[0])
        assert schema.read is True
