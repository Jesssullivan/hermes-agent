"""Route custody over MoA reference slots (agent/route_hooks.py F12, phase "moa"). A slot's provider
is explicit, so no fallback walk would ever ask the gate. Each test counts the calls that reach
``call_llm``: a refused slot must send nothing. The plugin side is a fake."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import moa_loop, route_hooks
from hermes_cli.moa_config import normalize_moa_config
from hermes_cli.plugins import PluginManager

OPUS = {"provider": "anthropic", "model": "claude-opus-5-5", "enabled": True}
KIMI = {"provider": "kimi-coding", "model": "k3.1-pro", "enabled": True, "route_custody": "required"}
SOL = {"provider": "cade-siwc", "model": "gpt-6.1-sol", "enabled": True, "route_custody": "required"}


@pytest.fixture
def hooks(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: mgr)
    return mgr._hooks


@pytest.fixture
def sent():
    """Every reference call that left, by provider. A fake call_llm answers each one."""
    calls = []

    def fake_call_llm(*, task, provider=None, **_kwargs):
        calls.append((task, provider))
        message = SimpleNamespace(content="advice", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None, model="m")

    with (patch.object(moa_loop, "call_llm", side_effect=fake_call_llm),
          patch.object(moa_loop, "_slot_runtime", side_effect=lambda s: {"provider": s["provider"], "model": s["model"]}),
          patch.object(moa_loop, "_price_reference_response", return_value=(moa_loop.CanonicalUsage(), None))):
        yield calls


def fan_out(*slots, agent=None):
    agent = agent if agent is not None else SimpleNamespace(session_id="s1", provider="anthropic", model="claude-opus-5-5")
    return moa_loop._run_references_parallel(list(slots), [{"role": "user", "content": "hi"}], agent=agent)


def outputs(results):
    return {label.split(":")[0]: text for label, text, _ in results}


def client_raw_gate(*, candidate, labels, **_kw):
    """Refuses Kimi and Sol on a session labelled as holding a client raw export: the SPR-102 half."""
    return {"action": "skip"} if "cade:client-raw" in labels and candidate["provider"] != "anthropic" else {"action": "allow"}


def test_no_gate_and_no_key_changes_nothing(hooks, sent):
    plain = {k: v for k, v in KIMI.items() if k != "route_custody"}
    results = fan_out(OPUS, plain)
    assert sorted(p for _, p in sent) == ["anthropic", "kimi-coding"]
    assert [text for _, text, _ in results] == ["advice", "advice"]


def test_required_slot_with_no_gate_loaded_sends_nothing(hooks, sent):
    results = fan_out(OPUS, KIMI, SOL)
    assert [p for _, p in sent] == ["anthropic"]
    assert outputs(results)["kimi-coding"] == moa_loop._ROUTE_CUSTODY_REFERENCE_NOTE
    assert outputs(results)["cade-siwc"] == moa_loop._ROUTE_CUSTODY_REFERENCE_NOTE


def test_gate_refuses_a_labelled_turn_and_the_others_still_run(hooks, sent):
    hooks["route_custody"] = [client_raw_gate]
    with patch.object(route_hooks, "_session_labels", return_value=frozenset({"cade:client-raw"})):
        results = fan_out(OPUS, KIMI, SOL)
    assert [p for _, p in sent] == ["anthropic"]
    assert len(results) == 3


def test_gate_admits_a_clean_turn(hooks, sent):
    hooks["route_custody"] = [client_raw_gate]
    with patch.object(route_hooks, "_session_labels", return_value=frozenset()):
        fan_out(OPUS, KIMI, SOL)
    assert sorted(p for _, p in sent) == ["anthropic", "cade-siwc", "kimi-coding"]


def test_gate_sees_the_moa_phase_and_the_slot(hooks, sent):
    seen = []

    def gate(**kwargs):
        seen.append((kwargs["phase"], kwargs["task"], kwargs["candidate"]))
        return {"action": "allow"}

    hooks["route_custody"] = [gate]
    fan_out(KIMI)
    assert seen == [("moa", "moa_reference", {"provider": "kimi-coding", "model": "k3.1-pro"})]


def test_a_gate_that_raises_refuses(hooks, sent):
    def broken(**_kw):
        raise RuntimeError("gate down")

    hooks["route_custody"] = [broken]
    fan_out(KIMI)
    assert sent == []


def test_a_gate_that_abstains_refuses_a_required_slot_only(hooks, sent):
    hooks["route_custody"] = [lambda **_kw: None]
    fan_out(OPUS, KIMI)
    assert [p for _, p in sent] == ["anthropic"]


def test_unknown_labels_refuse_without_asking(hooks, sent):
    hooks["route_custody"] = [client_raw_gate]
    with patch.object(route_hooks, "_labels_known", return_value=False):
        fan_out(OPUS)
    assert sent == []


def test_no_agent_and_no_turn_in_scope_refuses_a_gated_slot(hooks, sent):
    assert route_hooks.moa_reference_refuses(None, KIMI) is True
    assert route_hooks.moa_reference_refuses(None, {"provider": "anthropic", "model": "m"}) is False


def test_a_check_that_raises_refuses(hooks, sent):
    with patch.object(route_hooks, "fallback_custody_refuses", side_effect=RuntimeError("broken")):
        fan_out(KIMI)
    assert sent == []


def test_every_refused_reference_leaves_the_aggregator_alone_with_a_notice(hooks, sent):
    results = fan_out(KIMI, SOL)
    assert sent == []
    agg_refs, degraded, all_failed = moa_loop._guidance_inputs(results, False, "loud")
    assert agg_refs == [] and all_failed
    assert "kimi-coding" in degraded and "cade-siwc" in degraded


def test_normalize_keeps_the_key_on_a_slot():
    normalized = normalize_moa_config({"default_preset": "home", "presets": {"home": {
        "reference_models": [KIMI], "aggregator": {"provider": "anthropic", "model": "claude-opus-5-5"}}}})
    assert normalized["presets"]["home"]["reference_models"][0]["route_custody"] == "required"
