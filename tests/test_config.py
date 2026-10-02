import re

from coderet.config import MODELS, RunConfig, get_model


def test_registry_pins_full_revisions():
    for spec in MODELS.values():
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), spec.key


def test_fingerprint_changes_with_config():
    base = RunConfig(get_model("qwen3-0.6b"))
    assert base.fingerprint() == RunConfig(get_model("qwen3-0.6b")).fingerprint()
    assert base.fingerprint() != RunConfig(get_model("jina-code-0.5b")).fingerprint()
    assert base.fingerprint() != RunConfig(get_model("qwen3-0.6b"), extra={"max_len": 512}).fingerprint()
    assert base.cache_dir().name.startswith("qwen3-0.6b-")
