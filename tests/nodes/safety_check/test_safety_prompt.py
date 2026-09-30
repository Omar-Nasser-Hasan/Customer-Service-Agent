"""Offline checks of the safety prompt, its wiring, and the labeled case fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from customer_service.config.settings import Settings
from customer_service.nodes.safety_check.node import SAFE_REFUSAL, SafetyDecision, build_node
from customer_service.nodes.safety_check.prompt import build_system_prompt
from customer_service.privacy.redaction import redact_text
from customer_service.state.models import AgentState, EscalationReason
from tests.conftest import ScriptedSafetyJudge

CASES = json.loads((Path(__file__).parent / "safety_cases.json").read_text(encoding="utf-8"))
SETTINGS = Settings(store_name="Souqly")
PROMPT = build_system_prompt(SETTINGS)
WORD_BUDGET = 230
CHAR_BUDGET = 1600


def run(message: str, judge: ScriptedSafetyJudge) -> dict[str, object]:
    return build_node(settings=SETTINGS, judge=judge)(AgentState(messages=[HumanMessage(content=message)]))


def test_prompt_names_the_store_and_every_schema_action() -> None:
    actions = get_args(SafetyDecision.model_fields["action"].annotation)
    assert set(actions) == {"allow", "refuse", "escalate"}
    assert "Souqly" in PROMPT
    assert all(f"{action}:" in PROMPT for action in actions)


def test_prompt_stays_small() -> None:
    # Runs on every non-regex message; growth here is paid in latency and tokens.
    assert len(PROMPT.split()) <= WORD_BUDGET
    assert len(PROMPT) <= CHAR_BUDGET


def test_prompt_declares_untrusted_input_placeholders_and_languages() -> None:
    for required in ("untrusted", "Never obey", "[EMAIL]", "[ORDER_ID]", "Arabic", "self-harm"):
        assert required in PROMPT


def test_classifier_gets_the_fixed_prompt_and_only_the_redacted_message() -> None:
    judge = ScriptedSafetyJudge([{"action": "allow"}])
    run("Can you explain this: alice@example.com, ORD-1001?", judge)

    system, human = judge.calls[0]
    assert isinstance(system, SystemMessage) and system.content == PROMPT
    assert isinstance(human, HumanMessage)
    assert human.content == "Can you explain this: [EMAIL], [ORDER_ID]?"


@pytest.mark.parametrize(
    ("verdict", "action", "reason"),
    [
        ({"action": "allow"}, "allow", None),
        (SafetyDecision(action="allow"), "allow", None),
        ({"action": "refuse"}, "refuse", None),
        ({"action": "escalate"}, "escalate", EscalationReason.SAFETY_UNCERTAIN),
    ],
)
def test_each_verdict_maps_to_the_right_update(verdict: object, action: str, reason: object) -> None:
    update = run("Can you explain this policy?", ScriptedSafetyJudge([verdict]))

    assert update["safety_action"] == action
    assert update.get("escalation_reason") == reason
    if action == "refuse":
        assert update["messages"][0].content == SAFE_REFUSAL
    else:
        assert "messages" not in update


@pytest.mark.parametrize(
    "verdict",
    [
        {"action": "ALLOW"},
        {"action": "allow", "reason": "ordinary"},
        {"action": ["allow"]},
        {},
        None,
        "allow",
        RuntimeError("blocked by provider"),
    ],
)
def test_malformed_or_failed_verdicts_fail_closed(verdict: object) -> None:
    update = run("Can you explain this policy?", ScriptedSafetyJudge([verdict]))

    assert update["safety_action"] == "escalate"
    assert update["escalation_reason"] == EscalationReason.SAFETY_UNCERTAIN


def test_label_fixture_is_bilingual_balanced_and_unique() -> None:
    for label in ("allow", "refuse", "escalate"):
        for language in ("en", "ar"):
            count = sum(1 for c in CASES if c["expected"] == label and c["language"] == language)
            assert count >= 5, (label, language)
    assert len({c["text"] for c in CASES}) == len(CASES)
    assert any(c.get("critical") for c in CASES)


def test_regex_layer_agrees_with_every_label_where_it_fires() -> None:
    caught = 0
    wrong: list[str] = []
    for case in CASES:
        judge = ScriptedSafetyJudge([{"action": "allow"}])
        update = run(case["text"], judge)
        if judge.calls:  # the model decides this one; the live test covers it
            continue
        caught += 1
        if update["safety_action"] != case["expected"]:
            wrong.append(f'{case["text"]} -> {update["safety_action"]}')
    assert caught >= 3
    assert wrong == []


def test_redaction_keeps_arabic_intact_and_masks_only_identifiers() -> None:
    plain = [c["text"] for c in CASES if c["language"] == "ar" and "ORD-" not in c["text"]]
    assert plain and all(redact_text(text) == text for text in plain)
    assert (
        redact_text("رقم طلبي ORD-1001 وبريدي alice@example.com")
        == "رقم طلبي [ORDER_ID] وبريدي [EMAIL]"
    )