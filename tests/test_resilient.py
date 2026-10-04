"""Failure recovery of the embedding encoder, with fake backends (no models, no downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from coderet.embed.probe import Probe, check_probe, validate_vectors
from coderet.embed.resilient import Candidate, EncoderUnavailable, ResilientEncoder

DIM = 8
PROBE = Probe(
    items=[("query", "q"), ("document", "good"), ("document", "bad")],
    triples=[(0, 1, 2)],
)


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype("float32")


def _vec_for(text: str) -> np.ndarray:
    """Deterministic unit vector per text; the probe texts are placed so the triple holds."""
    fixed = {
        "q": np.eye(DIM)[0],
        "good": _unit(np.eye(DIM)[0] + 0.1 * np.eye(DIM)[1]),
        "bad": np.eye(DIM)[5],
    }
    if text in fixed:
        return fixed[text].astype("float32")
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return _unit(rng.normal(size=DIM))


class Fake:
    """A backend with scriptable failures."""

    def __init__(self, name: str, *, fail_after: int | None = None, nan: bool = False,
                 scale: float = 1.0, wrong_rows: bool = False, garbage: bool = False) -> None:
        self.name, self.calls = name, 0
        self.fail_after, self.nan, self.scale = fail_after, nan, scale
        self.wrong_rows, self.garbage = wrong_rows, garbage

    def embed(self, texts: list[str], role: str) -> np.ndarray:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError(f"{self.name} exploded")
        if self.garbage:  # finite, unit norm, but semantically meaningless (probe triple fails)
            return np.vstack([np.eye(DIM)[3] for _ in texts]).astype("float32")
        out = np.vstack([_vec_for(t) for t in texts]) * self.scale
        if self.nan:
            out[:] = np.nan
        if self.wrong_rows:
            out = out[:-1]
        return out.astype("float32")


def make(*backends: Fake, **kw) -> ResilientEncoder:
    return ResilientEncoder([Candidate(b.name, (lambda b=b: b)) for b in backends], probe=PROBE, dim=DIM, **kw)


def test_healthy_first_backend_is_used_and_nothing_falls_back():
    a, b = Fake("fast"), Fake("ref")
    enc = make(a, b)
    out = enc.embed_queries(["x", "y"])
    assert out.shape == (2, DIM) and enc.active_name == "fast"
    assert b.calls == 0
    assert [e["event"] for e in enc.events] == ["activated"]


def test_backend_that_fails_to_build_is_skipped():
    def boom() -> Fake:
        raise FileNotFoundError("no exported model")

    enc = ResilientEncoder([Candidate("ov", boom), Candidate("ref", lambda: Fake("ref"))], probe=PROBE, dim=DIM)
    enc.embed_documents(["x"])
    assert enc.active_name == "ref"
    assert enc.events[0]["event"] == "rejected" and "no exported model" in enc.events[0]["detail"]


def test_nan_backend_is_rejected_by_the_probe():
    enc = make(Fake("nan", nan=True), Fake("ref"))
    enc.embed_queries(["x"])
    assert enc.active_name == "ref"
    assert "NaN" in enc.events[0]["detail"] or "nan" in enc.events[0]["detail"].lower()


def test_finite_but_meaningless_backend_is_rejected_by_the_semantic_check():
    enc = make(Fake("garbage", garbage=True), Fake("ref"))
    enc.embed_queries(["x"])
    assert enc.active_name == "ref"
    assert "semantic" in enc.events[0]["detail"]


def test_reference_cosine_threshold_rejects_a_drifting_backend():
    ref_vectors = np.vstack([_vec_for(t) for _, t in PROBE.items])
    drifting = Fake("drift")
    # a backend whose vectors are rotated away from the reference but still pass the other checks
    orig = drifting.embed

    def rotated(texts, role):
        v = orig(texts, role)
        return np.vstack([_unit(x + 0.9 * np.eye(DIM)[2]) if t not in ("q", "good", "bad") else x
                          for x, t in zip(v, texts, strict=True)])

    drifting.embed = rotated  # type: ignore[method-assign]
    ok = Fake("ok")
    # probe items themselves are untouched above, so force drift on the probe too
    probe_rot = Fake("drift2")
    probe_rot.embed = lambda texts, role: np.vstack(  # type: ignore[method-assign]
        [_unit(_vec_for(t) + 0.3 * np.eye(DIM)[7]) for t in texts]
    )
    enc = ResilientEncoder(
        [Candidate("drift2", lambda: probe_rot, min_cosine=0.999), Candidate("ok", lambda: ok, min_cosine=0.999)],
        probe=PROBE, reference=ref_vectors.astype("float32"), dim=DIM,
    )
    enc.embed_queries(["x"])
    assert enc.active_name == "ok"
    assert "cosine to reference" in enc.events[0]["detail"]


def test_runtime_exception_demotes_and_retries_the_same_batch_on_the_next_backend():
    flaky, ref = Fake("flaky", fail_after=3), Fake("ref")  # 2 probe calls + 1 good call, then it dies
    enc = make(flaky, ref)
    first = enc.embed_queries(["a"])
    assert enc.active_name == "flaky"
    second = enc.embed_queries(["a", "b"])  # flaky raises here
    assert enc.active_name == "ref"
    assert second.shape == (2, DIM)
    np.testing.assert_allclose(second[0], first[0], atol=1e-6)  # same text, same vector from the fallback
    assert [e["event"] for e in enc.events].count("demoted") == 1


def test_runtime_nan_output_is_caught_and_retried():
    class LateNaN(Fake):
        def embed(self, texts, role):
            self.calls += 1
            out = np.vstack([_vec_for(t) for t in texts]).astype("float32")
            if self.calls > 2:
                out[:] = np.nan
            return out

    enc = make(LateNaN("late"), Fake("ref"))
    out = enc.embed_documents(["x"])
    assert np.isfinite(out).all() and enc.active_name == "ref"


def test_wrong_row_count_is_caught():
    class Short(Fake):
        def embed(self, texts, role):
            self.calls += 1
            out = np.vstack([_vec_for(t) for t in texts]).astype("float32")
            return out[:-1] if self.calls > 2 and len(texts) > 1 else out

    enc = make(Short("short"), Fake("ref"))
    assert enc.embed_documents(["a", "b", "c"]).shape == (3, DIM)
    assert enc.active_name == "ref"


def test_demotion_is_permanent_no_flapping():
    flaky, ref = Fake("flaky", fail_after=2), Fake("ref")  # dies on the first real call
    enc = make(flaky, ref)
    enc.embed_queries(["a"])  # flaky dies here, ref takes over
    calls_before = flaky.calls
    for _ in range(3):
        enc.embed_queries(["a"])
    assert flaky.calls == calls_before and enc.active_name == "ref"


def test_all_backends_failing_raises_with_every_reason():
    enc = make(Fake("a", nan=True), Fake("b", garbage=True))
    with pytest.raises(EncoderUnavailable) as err:
        enc.embed_queries(["x"])
    msg = str(err.value)
    assert "a:" in msg and "b:" in msg


def test_last_backend_failing_at_runtime_raises_cleanly():
    enc = make(Fake("only", fail_after=3))  # 2 probe calls + 1 good call, then it dies
    enc.embed_queries(["x"])
    with pytest.raises(EncoderUnavailable):
        enc.embed_queries(["y"])


def test_empty_input_returns_empty_array_without_touching_backends():
    a = Fake("a")
    enc = make(a)
    out = enc.embed_queries([])
    assert out.shape == (0, DIM) and a.calls == 0


def test_validate_vectors():
    good = np.eye(3, dtype="float32")
    assert validate_vectors(good) is None
    assert "NaN" in (validate_vectors(np.array([[np.nan, 1.0]], dtype="float32")) or "")
    assert "zero-norm" in (validate_vectors(np.zeros((1, 3), dtype="float32")) or "")
    assert "unit norm" in (validate_vectors(np.array([[2.0, 0.0]], dtype="float32")) or "")
    assert "dimension" in (validate_vectors(good, dim=4) or "")
    assert validate_vectors([[1.0]]) is not None  # type: ignore[arg-type]


def test_check_probe_semantic_failure_message():
    vecs = np.vstack([np.eye(DIM)[0], np.eye(DIM)[5], np.eye(DIM)[0]]).astype("float32")  # positive is far
    res = check_probe(vecs, PROBE)
    assert not res.ok and "semantic" in res.reason
