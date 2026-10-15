import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from logagent.service import app

client = TestClient(app)
GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("case", GOLDEN, ids=[g["block_id"] for g in GOLDEN])
def test_golden_examples_match_notebook(case):
    r = client.post("/predict", json={"events": case["events"], "elapsed_seconds": case["elapsed_seconds"]})
    assert r.status_code == 200
    body = r.json()
    assert body["p_anomaly"] == pytest.approx(case["expected_p"], abs=1e-6)
    assert body["alert"] == (body["p_anomaly"] >= body["threshold"])


def test_only_first_8_events_used():
    events = ["E5", "E22", "E5", "E5", "E11", "E9", "E11", "E9"]
    a = client.post("/predict", json={"events": events, "elapsed_seconds": 1.0}).json()
    b = client.post("/predict", json={"events": events + ["E3"] * 50, "elapsed_seconds": 1.0}).json()
    assert a["p_anomaly"] == b["p_anomaly"]


@pytest.mark.parametrize("payload", [
    {"events": ["E99"], "elapsed_seconds": 1.0},          # unknown event id
    {"events": [], "elapsed_seconds": 1.0},               # empty
    {"events": ["E5"], "elapsed_seconds": -1},            # negative time
    {"events": ["E5"] * 1001, "elapsed_seconds": 1.0},    # oversized
    {"elapsed_seconds": 1.0},                             # missing field
])
def test_invalid_input_rejected(payload):
    assert client.post("/predict", json=payload).status_code == 422
