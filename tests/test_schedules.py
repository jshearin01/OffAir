"""Scheduled-recording store: CRUD, atomicity, corruption, expiry."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.storage.schedules import (
    FIRE_GRACE_SECONDS,
    Schedule,
    cancel_schedule,
    create_schedule,
    due_schedules,
    expire_past_due,
    expire_stale_due,
    load_schedules,
    mark_fired,
    mark_skipped,
)


def _future(minutes=60):
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


def _past(minutes=60):
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


def test_create_and_load_round_trip(tmp_path):
    path = tmp_path / "schedules.json"
    created = create_schedule(
        _future(), "ms-teams.exe", subject="Weekly sync",
        attendees=["Priya", "Marcus"], organiser="John",
        created_by="mcp", path=path)
    assert created.status == "pending"
    assert created.id
    loaded = load_schedules(path)
    assert len(loaded) == 1
    assert loaded[0].id == created.id
    assert loaded[0].subject == "Weekly sync"
    assert loaded[0].attendees == ["Priya", "Marcus"]
    assert loaded[0].organiser == "John"
    assert loaded[0].created_by == "mcp"
    assert loaded[0].fire_at.tzinfo is not None
    # The file is plain JSON with the documented shape.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schedules"][0]["target_app"] == "ms-teams.exe"
    # No temp file left behind by the atomic write.
    assert not (tmp_path / "schedules.json.tmp").exists()


def test_create_rejects_past_fire_time(tmp_path):
    path = tmp_path / "schedules.json"
    with pytest.raises(ValueError, match="not in the future"):
        create_schedule(_past(), "ms-teams.exe", path=path)
    assert load_schedules(path) == []


def test_create_rejects_blank_target(tmp_path):
    path = tmp_path / "schedules.json"
    with pytest.raises(ValueError, match="must not be blank"):
        create_schedule(_future(), "   ", path=path)


def test_create_rejects_naive_past_after_localising(tmp_path):
    # A naive wall time in the past is past, even localised.
    naive_past = datetime.now().astimezone().replace(tzinfo=None) \
        - timedelta(minutes=5)
    with pytest.raises(ValueError, match="not in the future"):
        create_schedule(naive_past, "ms-teams.exe",
                        path=tmp_path / "schedules.json")


def test_naive_future_fire_time_is_localised(tmp_path):
    path = tmp_path / "schedules.json"
    naive = datetime.now().astimezone().replace(tzinfo=None) \
        + timedelta(hours=1)
    created = create_schedule(naive, "ms-teams.exe", path=path)
    assert created.fire_at.tzinfo is not None
    assert created.fire_at > datetime.now(timezone.utc)


def test_missing_file_loads_empty(tmp_path):
    assert load_schedules(tmp_path / "nope.json") == []


def test_corrupt_json_loads_empty_without_raising(tmp_path):
    path = tmp_path / "schedules.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_schedules(path) == []


def test_wrong_shape_loads_empty_without_raising(tmp_path):
    path = tmp_path / "schedules.json"
    path.write_text(json.dumps(["not", "an", "object"]),
                    encoding="utf-8")
    assert load_schedules(path) == []
    path.write_text(json.dumps({"schedules": "not-a-list"}),
                    encoding="utf-8")
    assert load_schedules(path) == []


def test_malformed_entries_skipped_valid_kept(tmp_path):
    path = tmp_path / "schedules.json"
    good = create_schedule(_future(), "ms-teams.exe", path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schedules"].extend([
        {"id": "bad-no-fire-time", "target_app": "x.exe"},
        {"fire_at": _future().isoformat(), "target_app": "y.exe"},
        {"id": "bad-status", "fire_at": _future().isoformat(),
         "target_app": "z.exe", "status": "bogus"},
        "not-a-dict",
    ])
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_schedules(path)
    assert [s.id for s in loaded] == [good.id]


def test_cancel_pending_and_errors(tmp_path):
    path = tmp_path / "schedules.json"
    created = create_schedule(_future(), "ms-teams.exe", path=path)
    cancelled = cancel_schedule(created.id, path=path)
    assert cancelled.status == "cancelled"
    assert load_schedules(path)[0].status == "cancelled"
    with pytest.raises(ValueError, match="not pending"):
        cancel_schedule(created.id, path=path)
    with pytest.raises(ValueError, match="No schedule"):
        cancel_schedule("missing-id", path=path)


def test_mark_fired_and_skipped(tmp_path):
    path = tmp_path / "schedules.json"
    created = create_schedule(_future(), "ms-teams.exe", path=path)
    fired = mark_fired(created, session_id="2026-11-22_09-00-00_teams",
                       path=path)
    assert fired.status == "fired"
    assert fired.session_id == "2026-11-22_09-00-00_teams"
    assert fired.fired_at is not None
    other = create_schedule(_future(120), "zoom.exe", path=path)
    skipped = mark_skipped(other, "already recording", path=path)
    assert skipped.status == "skipped"
    assert skipped.note == "already recording"


def test_expire_past_due_on_load(tmp_path):
    # A schedule created in the future, then time passes: startup
    # expiry marks it expired and leaves others alone.
    path = tmp_path / "schedules.json"
    soon = create_schedule(_future(1), "ms-teams.exe", path=path)
    later = create_schedule(_future(120), "zoom.exe", path=path)
    expired = expire_past_due(
        datetime.now(timezone.utc) + timedelta(minutes=5), path=path)
    assert [s.id for s in expired] == [soon.id]
    assert expired[0].status == "expired"
    assert expired[0].note
    statuses = {s.id: s.status for s in load_schedules(path)}
    assert statuses[soon.id] == "expired"
    assert statuses[later.id] == "pending"


def test_due_schedules_grace_boundary(tmp_path):
    path = tmp_path / "schedules.json"
    fire_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    # Write directly (create_schedule rejects the past).
    record = Schedule(id="due1", fire_at=fire_at,
                      target_app="ms-teams.exe")
    path.write_text(json.dumps({"schedules": [record.to_dict()]}),
                    encoding="utf-8")
    assert [s.id for s in due_schedules(path=path)] == ["due1"]
    # Past the grace: no longer due, and expire_stale_due claims it.
    assert due_schedules(
        datetime.now(timezone.utc)
        + timedelta(seconds=FIRE_GRACE_SECONDS), path=path) == []
    stale = expire_stale_due(
        datetime.now(timezone.utc)
        + timedelta(seconds=FIRE_GRACE_SECONDS + 1), path=path)
    assert [s.id for s in stale] == ["due1"]
    assert load_schedules(path)[0].status == "expired"


def test_due_schedules_sorted_earliest_first(tmp_path):
    path = tmp_path / "schedules.json"
    now = datetime.now(timezone.utc)
    for offset in (30, 10, 20):
        record = Schedule(
            id=f"s{offset}", fire_at=now - timedelta(seconds=offset),
            target_app="ms-teams.exe")
        payload = {"schedules": [record.to_dict()]}
        # Append via merge path: write, then next record merges.
        existing = load_schedules(path)
        existing.append(record)
        path.write_text(json.dumps(
            {"schedules": [s.to_dict() for s in existing]}),
            encoding="utf-8")
    assert [s.id for s in due_schedules(path=path)] == ["s30", "s20", "s10"]


def test_merge_preserves_records_from_the_other_side(tmp_path):
    # The app loads, the MCP side creates a schedule, then the app
    # marks its own record fired: the MCP-side record must survive.
    path = tmp_path / "schedules.json"
    first = create_schedule(_future(), "ms-teams.exe", path=path)
    second = create_schedule(_future(30), "zoom.exe", path=path)
    mark_fired(first, path=path)
    ids = [s.id for s in load_schedules(path)]
    assert ids == [first.id, second.id]
    assert load_schedules(path)[0].status == "fired"
    assert load_schedules(path)[1].status == "pending"


def test_from_dict_tolerates_bad_optional_fields(tmp_path):
    schedule = Schedule.from_dict({
        "id": "x1",
        "fire_at": _future().isoformat(),
        "target_app": "ms-teams.exe",
        "attendees": "not-a-list",
        "created_by": "someone-else",
        "subject": 42,
    })
    assert schedule is not None
    assert schedule.attendees == []
    assert schedule.created_by == "gui"
    assert schedule.subject is None
