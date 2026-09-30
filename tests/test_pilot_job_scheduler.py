import importlib.util
import json
import queue
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def job(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("pilot_job", ROOT / "kaggle" / "pilot_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processed = tmp_path / "project" / "data" / "processed"
    processed.mkdir(parents=True)
    monkeypatch.setattr(module, "PROJECT", tmp_path / "project")
    monkeypatch.setattr(module, "OUT", tmp_path / "out")
    monkeypatch.setattr(module, "EXPERIMENT", "context2b")
    monkeypatch.setattr(module, "CHUNK_SESSIONS", 3)
    monkeypatch.setattr(module, "DOMAINS", {
        "amazon_film": {"prefix": "amazon_film"}, "amazon_tv": {"prefix": "amazon_tv"}})
    (tmp_path / "out").mkdir()
    # Film has more sessions than TV, as at L4, so a fixed split would idle a GPU.
    for prefix, count in (("amazon_film", 10), ("amazon_tv", 4)):
        with (processed / f"{prefix}_pool_random_pilot.jsonl").open("w") as handle:
            for index in range(count):
                handle.write(json.dumps({"session_id": f"{prefix}-{index:02d}"}) + "\n")
    return module


def _fake_run(job, busy):
    def run_unit(unit, port, minutes):
        busy.append((port, unit["label"], time.monotonic()))
        candidates = job.PROJECT / unit["args"][unit["args"].index("--candidates") + 1]
        records = [json.loads(line) for line in candidates.read_text().splitlines()]
        time.sleep(0.02 * len(records))
        # Write trials in reverse to check that merging restores session order.
        unit["output"].write_text("".join(
            json.dumps({"session_id": r["session_id"], "port": port}) + "\n" for r in reversed(records)))
        unit["log"].write_text(f"{unit['label']}\n")
        return 0
    return run_unit


def _run_all(job, units, failures):
    work = queue.Queue()
    for unit in units:
        work.put(unit)
    workers = [threading.Thread(target=job.gpu_worker, args=(port, work, time.time(), failures))
               for port in job.GPU_PORTS]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)


def test_both_gpus_share_all_units_and_runs_merge_in_session_order(job, monkeypatch):
    busy = []
    monkeypatch.setattr(job, "run_unit", _fake_run(job, busy))
    units = job.plan_units([("random", "shuffled")])
    assert [unit["label"] for unit in units] == (
        [f"amazon_film_random_shuffled_context2b_chunk{i}" for i in range(4)]
        + [f"amazon_tv_random_shuffled_context2b_chunk{i}" for i in range(2)])
    failures = []
    _run_all(job, units, failures)
    assert not failures and all(unit["done"].is_set() for unit in units)
    assert {port for port, _, _ in busy} == set(job.GPU_PORTS)
    for domain, count in (("amazon_film", 10), ("amazon_tv", 4)):
        name = job.run_name(domain, "random", "shuffled")
        job.merge_run(name, [unit for unit in units if unit["name"] == name])
        merged = [json.loads(line) for line in (job.OUT / f"{name}_pilot_trials.jsonl").read_text().splitlines()]
        assert [r["session_id"] for r in merged] == [f"{domain}-{i:02d}" for i in range(count)]
        assert (job.OUT / f"{name}_run.log").is_file()
    assert sorted(path.name for path in job.OUT.iterdir()) == sorted(
        f"{job.run_name(d, 'random', 'shuffled')}_{kind}"
        for d in ("amazon_film", "amazon_tv") for kind in ("pilot_trials.jsonl", "run.log"))


def test_failed_unit_is_reported_and_released(job, monkeypatch):
    def run_unit(unit, port, minutes):
        if unit["label"].endswith("tv_random_shuffled_context2b_chunk1"):
            raise OSError("server gone")
        return 0
    monkeypatch.setattr(job, "run_unit", run_unit)
    units = job.plan_units([("random", "shuffled")])
    failures = []
    _run_all(job, units, failures)
    assert failures == ["amazon_tv_random_shuffled_context2b_chunk1"]
    assert all(unit["done"].is_set() for unit in units)
