"""Kaggle GPU pilot: prompt ensemble on a sample of both domains, then evaluation.

POOLS selects the candidate pools: "stress" is the original top-popular pool
(evaluated with evaluate_ensemble.py and power_analysis.py);
"popularity_matched" and "retrieval" are the repaired benchmark built by
scripts/build_benchmark.py, always run with shuffled candidates and evaluated
with evaluate_debiased.py (validation-fitted calibration, test-split scores).

This file is the job template. kaggle/build_kernel.py appends the project
source as an embedded archive and writes a pushable kernel to kaggle/build/.
Everything the run produces lands in /kaggle/working/pilot, which Kaggle keeps
as the notebook output.
"""

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("PILOT_MODEL", "qwen2.5:3b-instruct")
EXPECTED_DIGEST = "357c53fb659c"  # Ollama ID of the frozen 3B baseline
OLLAMA_VERSION = "0.32.15"  # pinned in EXPERIMENT_SPEC.md Addendum v5
CONTEXT_LENGTH = 32768
PILOT_SESSIONS = int(os.environ.get("PILOT_SESSIONS", "300"))
PLANNED_SESSIONS = 3000
# "fixed" keeps the stored pool order; "shuffled" gives every prompt its own
# seeded candidate order (run_ensemble.py --shuffle-candidates).
CONDITIONS = os.environ.get("PILOT_CONDITIONS", "fixed").split(",")
POOLS = os.environ.get("PILOT_POOLS", "stress").split(",")
# Item context shown to every ensemble member: "title" or "genre" (title plus
# genres; on Amazon the genres include the platform).
CONTEXT = os.environ.get("PILOT_CONTEXT", "title")
WORDINGS = ("baseline_scores_v1", "wording_direct_v1", "wording_preference_v1", "wording_detailed_v1")
BENCHMARK_POOLS = ("popularity_matched", "retrieval")
MAX_RUNTIME_MINUTES = 660  # total budget, inside Kaggle's 12-hour limit

PROJECT = Path("/tmp/project")
OUT = Path("/kaggle/working/pilot")

DOMAINS = {
    "movielens": {
        "port": 11434,
        "prefix": "ml1m",
        "extra": [],
        "popularity": "data/processed/ml1m_item_popularity.json",
        "popularity_domain": "movielens",
    },
    "amazon_games": {
        "port": 11435,
        "prefix": "amazon_games",
        "extra": [
            "--examples", "data/processed/amazon_games_leave_one_out.jsonl",
            "--candidates", "data/processed/amazon_games_candidate_pools.jsonl",
            "--items-file", "data/processed/amazon_games_items.jsonl",
        ],
        "popularity": "data/processed/amazon_games_item_popularity.json",
        "popularity_domain": "amazon-games",
    },
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(cmd, **kwargs):
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    subprocess.run(cmd, check=True, **kwargs)


def extract_project():
    PROJECT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
        tar.extractall(PROJECT, filter="data")
    # Plain import path instead of `pip install -e`: no build dependencies to fetch.
    os.environ["PYTHONPATH"] = str(PROJECT / "src")


def check_internet():
    for url in ("https://ollama.com", "https://files.grouplens.org", "https://huggingface.co"):
        try:
            urllib.request.urlopen(url, timeout=20)
        except OSError as exc:
            raise RuntimeError(
                f"No internet access to {url} ({exc}). Enable Internet for this "
                "notebook; Kaggle requires a phone-verified account."
            ) from exc


def install_ollama():
    run("apt-get install -y -qq zstd pciutils > /dev/null 2>&1 || true", shell=True)
    try:
        run(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh",
            shell=True)
    except subprocess.CalledProcessError:
        log(f"WARNING: pinned Ollama {OLLAMA_VERSION} failed to install; using latest")
        run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
    return subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()


def gpu_names():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def start_server(port, gpu_index):
    env = dict(os.environ,
               OLLAMA_HOST=f"127.0.0.1:{port}",
               OLLAMA_CONTEXT_LENGTH=str(CONTEXT_LENGTH),
               OLLAMA_NUM_PARALLEL="1",
               OLLAMA_KEEP_ALIVE="-1",
               CUDA_VISIBLE_DEVICES=str(gpu_index))
    handle = open(OUT / f"ollama_{port}.log", "w")
    subprocess.Popen(["ollama", "serve"], env=env, stdout=handle, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/version", timeout=2)
            log(f"Ollama up on port {port} (GPU {gpu_index})")
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"Ollama on port {port} did not start")


def model_digest():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                            env=dict(os.environ, OLLAMA_HOST="127.0.0.1:11434"))
    for line in result.stdout.splitlines():
        if line.startswith(MODEL):
            return line.split()[1]
    return None


def warm_up(port):
    """Load the model onto this server's GPU before timing starts."""
    body = json.dumps({"model": MODEL, "prompt": "ok", "stream": False,
                       "options": {"num_predict": 1}}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    urllib.request.urlopen(request, timeout=600).read()


def prepare_data():
    python = sys.executable
    run([python, "scripts/prepare_movielens.py"], cwd=PROJECT)
    run([python, "scripts/prepare_amazon_games.py"], cwd=PROJECT)
    for spec in DOMAINS.values():
        run([python, "scripts/build_item_popularity.py", "--domain",
             spec["popularity_domain"]], cwd=PROJECT)
    if set(POOLS) & set(BENCHMARK_POOLS):
        for domain, spec in DOMAINS.items():
            run([python, "scripts/build_benchmark.py", "--domain", domain], cwd=PROJECT)
            # Rerankers only matter where the retriever found the target; an
            # unretrieved target is a miss end to end whatever the reranker does.
            processed = PROJECT / "data/processed"
            source = processed / f"{spec['prefix']}_pool_retrieval.jsonl"
            with source.open() as handle, (processed / f"{spec['prefix']}_pool_retrieval_found.jsonl").open("w") as out:
                for line in handle:
                    if json.loads(line)["target_retrieved"]:
                        out.write(line)


def pool_args(domain, pool):
    """run_ensemble.py arguments selecting a domain's examples and pool file."""
    spec = DOMAINS[domain]
    if pool == "stress":
        return list(spec["extra"])
    items = ["--items-file", "data/processed/amazon_games_items.jsonl"] if domain == "amazon_games" else []
    pool_file = "retrieval_found" if pool == "retrieval" else pool
    return [
        "--examples", f"data/processed/{spec['prefix']}_clean_examples.jsonl",
        "--candidates", f"data/processed/{spec['prefix']}_pool_{pool_file}.jsonl",
        *items,
    ]


def run_name(domain, pool, condition):
    name = f"{domain}_{condition}" if pool == "stress" else f"{domain}_{pool}_{condition}"
    return name if CONTEXT == "title" else f"{name}_{CONTEXT}"


def member_args():
    return [arg for wording in WORDINGS for arg in ("--member", f"{wording}:context_{CONTEXT}_v1")]


def run_domains(pool, condition, minutes):
    processes = {}
    for domain, spec in DOMAINS.items():
        name = run_name(domain, pool, condition)
        output = OUT / f"{name}_pilot_trials.jsonl"
        cmd = [sys.executable, "scripts/run_ensemble.py",
               "--provider", "chat-completions", "--model", MODEL,
               "--base-url", f"http://127.0.0.1:{spec['port']}/v1",
               "--domain", domain, *pool_args(domain, pool), *member_args(),
               "--sample-size", str(PILOT_SESSIONS), "--sample-seed", "0",
               "--continue-on-error", "--resume",
               "--max-runtime-minutes", str(minutes),
               "--output", str(output)]
        if condition == "shuffled":
            cmd += ["--shuffle-candidates", "--shuffle-seed", "0"]
        log(f"starting {domain}: " + " ".join(cmd))
        processes[domain] = subprocess.Popen(
            cmd, cwd=PROJECT, stdout=open(OUT / f"{name}_run.log", "w"),
            stderr=subprocess.STDOUT)
    for domain, process in processes.items():
        code = process.wait()
        log(f"{domain} finished with exit code {code}")
        if code != 0:
            raise RuntimeError(f"{domain} run failed; see {run_name(domain, pool, condition)}_run.log")


def max_prompt_tokens(path):
    largest = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            for trial in json.loads(line).get("member_trials", []):
                usage = trial.get("usage") or {}
                largest = max(largest, int(usage.get("prompt_tokens") or 0))
    return largest


def analyse_benchmark(pool, condition):
    summary = {}
    for domain, spec in DOMAINS.items():
        name = run_name(domain, pool, condition)
        trials = OUT / f"{name}_pilot_trials.jsonl"
        report_path = OUT / f"{name}_pilot_debiased.json"
        run([sys.executable, "scripts/evaluate_debiased.py", "--trials", str(trials),
             "--pools", f"data/processed/{spec['prefix']}_pool_{pool}.jsonl",
             "--output", str(report_path)], cwd=PROJECT)
        report = json.loads(report_path.read_text())
        hybrid_path = OUT / f"{name}_pilot_hybrid.json"
        run([sys.executable, "scripts/evaluate_hybrid.py", "--domain", domain, "--trials", str(trials),
             "--pools", f"data/processed/{spec['prefix']}_pool_{pool}.jsonl",
             "--output", str(hybrid_path)], cwd=PROJECT)
        hybrid = json.loads(hybrid_path.read_text())
        summary[domain] = {
            "hybrid_mrr": {m: round(v["RR"], 4) for m, v in hybrid["methods"].items()},
            "hybrid_llm_weight": hybrid["fitted_on_validation"]["llm_weight"],
            "hybrid_primary": {k: {"diff": round(v["mean_difference"], 4), "p": round(v["p_value"], 4)}
                               for k, v in hybrid["primary_comparisons"].items()},
            "test_sessions": report["test_sessions_all_members_parsed"],
            "validation_sessions": report["validation_sessions"],
            "preselected_single_prompt": report["preselected_single_prompt"],
            "mrr": {m: round(v["RR"], 4) for m, v in report["methods"].items()},
            "primary": {k: {"diff": round(v["mean_difference"], 4), "p_holm": round(v["p_holm"], 4)}
                        for k, v in report["primary_comparisons"].items()},
            "max_prompt_tokens": max_prompt_tokens(trials),
        }
        if "retrieval_recall_test" in report:
            summary[domain]["retrieval_recall_test"] = round(report["retrieval_recall_test"], 4)
    return summary


def analyse(condition):
    summary = {}
    for domain, spec in DOMAINS.items():
        trials = OUT / f"{domain}_{condition}_pilot_trials.jsonl"
        metrics = OUT / f"{domain}_{condition}_pilot_metrics.json"
        power = OUT / f"{domain}_{condition}_pilot_power.json"
        run([sys.executable, "scripts/evaluate_ensemble.py", "--input", str(trials),
             "--output", str(metrics), "--popularity-file", spec["popularity"]], cwd=PROJECT)
        run([sys.executable, "scripts/power_analysis.py", "--input", str(trials),
             "--output", str(power), "--planned-sessions", str(PLANNED_SESSIONS)], cwd=PROJECT)
        report = json.loads(power.read_text())
        conditions = json.loads(metrics.read_text())["conditions"]
        summary[domain] = {
            "hr_at_10": {name: round(c["metrics"]["HR@10"], 3)
                         for name, c in conditions.items()},
            "pilot_sessions": report["pilot_sessions"],
            "max_prompt_tokens": max_prompt_tokens(trials),
            "verdicts": {name: result["verdict"]
                         for name, result in report["comparisons"].items()},
        }
    return summary


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    extract_project()
    check_internet()
    ollama_version = install_ollama()
    gpus = gpu_names()
    log(f"GPUs: {gpus}")
    for index, spec in enumerate(DOMAINS.values()):
        start_server(spec["port"], index % max(len(gpus), 1))
    run(["ollama", "pull", MODEL], env=dict(os.environ, OLLAMA_HOST="127.0.0.1:11434"))
    digest = model_digest()
    if MODEL == "qwen2.5:3b-instruct" and digest != EXPECTED_DIGEST:
        log(f"WARNING: model digest {digest} differs from frozen {EXPECTED_DIGEST}")
    for spec in DOMAINS.values():
        warm_up(spec["port"])

    manifest = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "model": MODEL,
        "model_digest": digest,
        "expected_digest": EXPECTED_DIGEST,
        "ollama_version": ollama_version,
        "ollama_version_pinned": OLLAMA_VERSION,
        "context_length": CONTEXT_LENGTH,
        "num_parallel": 1,
        "gpus": gpus,
        "pilot_sessions_per_domain": PILOT_SESSIONS,
        "conditions": CONDITIONS,
        "pools": POOLS,
        "context": CONTEXT,
        "sample_seed": 0,
        "code_sha256": hashlib.sha256(PAYLOAD.encode()).hexdigest(),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    prepare_data()
    summary = {}
    # Benchmark pools are only ever run shuffled: calibration needs it.
    plan = [(pool, condition) for pool in POOLS for condition in CONDITIONS
            if pool == "stress" or condition == "shuffled"]
    for index, (pool, condition) in enumerate(plan):
        # Split what is left of the budget evenly over the remaining runs.
        left = MAX_RUNTIME_MINUTES - (time.time() - started) / 60
        run_domains(pool, condition, max(int(left / (len(plan) - index)), 1))
        key = condition if pool == "stress" else f"{pool}_{condition}"
        summary[key] = analyse(condition) if pool == "stress" else analyse_benchmark(pool, condition)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    summary["context_ok"] = all(
        domain["max_prompt_tokens"] < CONTEXT_LENGTH
        for runs in summary.values() for domain in runs.values())
    summary["elapsed_minutes"] = round((time.time() - started) / 60, 1)
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("SUMMARY\n" + json.dumps(summary, indent=2))
