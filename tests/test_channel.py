"""Tests for Channel ABC, SqliteChannel, and RedisChannel."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import redis

from fanout.channels.redis import RedisChannel
from fanout.channels.sqlite import SqliteChannel
from fanout.db.models import Evaluation, Run, Solution
from fanout.store import Store


def _redis_available() -> bool:
    try:
        r = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
        r.ping()
        return True
    except Exception:
        return False


@pytest.fixture()
def ch(tmp_path: Path) -> SqliteChannel:
    return SqliteChannel(db_path=tmp_path / "test.db")


@pytest.fixture()
def rch() -> RedisChannel:
    prefix = f"fanout:test:{uuid.uuid4().hex}:"
    ch = RedisChannel(prefix=prefix)
    yield ch
    # cleanup: delete all keys with this prefix
    for key in ch.r.scan_iter(f"{prefix}*"):
        ch.r.delete(key)


# ── SqliteChannel basics ──────────────────────────────────


class TestSqliteChannel:
    def test_put_get_roundtrip(self, ch: SqliteChannel) -> None:
        ch.put("t", "k1", {"x": 1})
        assert ch.get("t", "k1") == {"x": 1}

    def test_get_missing_returns_none(self, ch: SqliteChannel) -> None:
        assert ch.get("t", "nope") is None

    def test_put_upserts(self, ch: SqliteChannel) -> None:
        ch.put("t", "k1", {"v": "old"})
        ch.put("t", "k1", {"v": "new"})
        assert ch.get("t", "k1") == {"v": "new"}

    def test_list_no_filters(self, ch: SqliteChannel) -> None:
        ch.put("t", "a", {"n": 1})
        ch.put("t", "b", {"n": 2})
        items = ch.list("t")
        assert len(items) == 2

    def test_list_single_filter(self, ch: SqliteChannel) -> None:
        ch.put("t", "a", {"n": 1}, color="red")
        ch.put("t", "b", {"n": 2}, color="blue")
        ch.put("t", "c", {"n": 3}, color="red")
        items = ch.list("t", color="red")
        assert len(items) == 2
        assert {d["n"] for d in items} == {1, 3}

    def test_list_multiple_filters_intersection(self, ch: SqliteChannel) -> None:
        ch.put("t", "a", {"n": 1}, color="red", size="L")
        ch.put("t", "b", {"n": 2}, color="red", size="S")
        ch.put("t", "c", {"n": 3}, color="blue", size="L")
        items = ch.list("t", color="red", size="L")
        assert len(items) == 1
        assert items[0]["n"] == 1

    def test_list_empty_topic(self, ch: SqliteChannel) -> None:
        assert ch.list("empty") == []

    def test_delete_existing(self, ch: SqliteChannel) -> None:
        ch.put("t", "k1", {"x": 1})
        assert ch.delete("t", "k1") is True
        assert ch.get("t", "k1") is None

    def test_delete_missing(self, ch: SqliteChannel) -> None:
        assert ch.delete("t", "nope") is False

    def test_topic_isolation(self, ch: SqliteChannel) -> None:
        ch.put("a", "k1", {"topic": "a"})
        ch.put("b", "k1", {"topic": "b"})
        assert ch.get("a", "k1") == {"topic": "a"}
        assert ch.get("b", "k1") == {"topic": "b"}
        assert len(ch.list("a")) == 1
        assert len(ch.list("b")) == 1

    def test_upsert_replaces_indexes(self, ch: SqliteChannel) -> None:
        ch.put("t", "k1", {"v": 1}, color="red")
        ch.put("t", "k1", {"v": 2}, color="blue")
        assert ch.list("t", color="red") == []
        assert len(ch.list("t", color="blue")) == 1


# ── RedisChannel basics ──────────────────────────────────


@pytest.mark.skipif(not _redis_available(), reason="Redis not running")
class TestRedisChannel:
    def test_put_get_roundtrip(self, rch: RedisChannel) -> None:
        rch.put("t", "k1", {"x": 1})
        assert rch.get("t", "k1") == {"x": 1}

    def test_get_missing_returns_none(self, rch: RedisChannel) -> None:
        assert rch.get("t", "nope") is None

    def test_put_upserts(self, rch: RedisChannel) -> None:
        rch.put("t", "k1", {"v": "old"})
        rch.put("t", "k1", {"v": "new"})
        assert rch.get("t", "k1") == {"v": "new"}

    def test_list_no_filters(self, rch: RedisChannel) -> None:
        rch.put("t", "a", {"n": 1})
        rch.put("t", "b", {"n": 2})
        items = rch.list("t")
        assert len(items) == 2

    def test_list_single_filter(self, rch: RedisChannel) -> None:
        rch.put("t", "a", {"n": 1}, color="red")
        rch.put("t", "b", {"n": 2}, color="blue")
        rch.put("t", "c", {"n": 3}, color="red")
        items = rch.list("t", color="red")
        assert len(items) == 2
        assert {d["n"] for d in items} == {1, 3}

    def test_list_multiple_filters_intersection(self, rch: RedisChannel) -> None:
        rch.put("t", "a", {"n": 1}, color="red", size="L")
        rch.put("t", "b", {"n": 2}, color="red", size="S")
        rch.put("t", "c", {"n": 3}, color="blue", size="L")
        items = rch.list("t", color="red", size="L")
        assert len(items) == 1
        assert items[0]["n"] == 1

    def test_list_empty_topic(self, rch: RedisChannel) -> None:
        assert rch.list("empty") == []

    def test_delete_existing(self, rch: RedisChannel) -> None:
        rch.put("t", "k1", {"x": 1})
        assert rch.delete("t", "k1") is True
        assert rch.get("t", "k1") is None

    def test_delete_missing(self, rch: RedisChannel) -> None:
        assert rch.delete("t", "nope") is False

    def test_topic_isolation(self, rch: RedisChannel) -> None:
        rch.put("a", "k1", {"topic": "a"})
        rch.put("b", "k1", {"topic": "b"})
        assert rch.get("a", "k1") == {"topic": "a"}
        assert rch.get("b", "k1") == {"topic": "b"}
        assert len(rch.list("a")) == 1
        assert len(rch.list("b")) == 1

    def test_upsert_replaces_indexes(self, rch: RedisChannel) -> None:
        rch.put("t", "k1", {"v": 1}, color="red")
        rch.put("t", "k1", {"v": 2}, color="blue")
        assert rch.list("t", color="red") == []
        assert len(rch.list("t", color="blue")) == 1


# ── Store-on-Channel integration ─────────────────────────


class TestStoreOnSqliteChannel:
    def test_solution_roundtrip(self, ch: SqliteChannel) -> None:
        store = Store(channel=ch)
        sol = Solution(run_id="r1", round_num=0, model="test", output="hello")
        store.save_solution(sol)
        got = store.get_solutions_for_run("r1")
        assert len(got) == 1
        assert got[0].id == sol.id
        assert got[0].output == "hello"

    def test_solution_filter_by_round(self, ch: SqliteChannel) -> None:
        store = Store(channel=ch)
        s0 = Solution(run_id="r1", round_num=0, model="m", output="a")
        s1 = Solution(run_id="r1", round_num=1, model="m", output="b")
        store.save_solution(s0)
        store.save_solution(s1)
        assert len(store.get_solutions_for_run("r1")) == 2
        assert len(store.get_solutions_for_run("r1", round_num=0)) == 1
        assert store.get_solutions_for_run("r1", round_num=0)[0].output == "a"

    def test_run_roundtrip(self, ch: SqliteChannel) -> None:
        store = Store(channel=ch)
        run = Run(prompt="test prompt")
        store.save_run(run)
        got = store.get_run(run.id)
        assert got is not None
        assert got.prompt == "test prompt"

    def test_evaluation_roundtrip(self, ch: SqliteChannel) -> None:
        store = Store(channel=ch)
        ev = Evaluation(solution_id="s1", evaluator="test_eval", score=0.9)
        store.save_evaluation(ev)
        got = store.get_evaluations_for_solution("s1")
        assert len(got) == 1
        assert got[0].score == 0.9


@pytest.mark.skipif(not _redis_available(), reason="Redis not running")
class TestStoreOnRedisChannel:
    def test_solution_roundtrip(self, rch: RedisChannel) -> None:
        store = Store(channel=rch)
        sol = Solution(run_id="r1", round_num=0, model="test", output="hello")
        store.save_solution(sol)
        got = store.get_solutions_for_run("r1")
        assert len(got) == 1
        assert got[0].id == sol.id
        assert got[0].output == "hello"

    def test_solution_filter_by_round(self, rch: RedisChannel) -> None:
        store = Store(channel=rch)
        s0 = Solution(run_id="r1", round_num=0, model="m", output="a")
        s1 = Solution(run_id="r1", round_num=1, model="m", output="b")
        store.save_solution(s0)
        store.save_solution(s1)
        assert len(store.get_solutions_for_run("r1")) == 2
        assert len(store.get_solutions_for_run("r1", round_num=0)) == 1
        assert store.get_solutions_for_run("r1", round_num=0)[0].output == "a"

    def test_run_roundtrip(self, rch: RedisChannel) -> None:
        store = Store(channel=rch)
        run = Run(prompt="test prompt")
        store.save_run(run)
        got = store.get_run(run.id)
        assert got is not None
        assert got.prompt == "test prompt"

    def test_evaluation_roundtrip(self, rch: RedisChannel) -> None:
        store = Store(channel=rch)
        ev = Evaluation(solution_id="s1", evaluator="test_eval", score=0.9)
        store.save_evaluation(ev)
        got = store.get_evaluations_for_solution("s1")
        assert len(got) == 1
        assert got[0].score == 0.9

    def test_default_store_uses_redis(self) -> None:
        store = Store()
        assert isinstance(store.ch, RedisChannel)
