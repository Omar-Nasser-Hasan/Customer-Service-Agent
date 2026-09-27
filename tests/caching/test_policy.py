from customer_service.caching.policy import decide_cache_policy, static_context_fingerprint


def test_cache_fingerprint_is_stable_and_changes_with_static_context() -> None:
    baseline = static_context_fingerprint(
        prompt_revision="assistant/v1", model="gemini", tool_schemas=[{"name": "faq_lookup"}]
    )
    assert baseline == static_context_fingerprint(
        prompt_revision="assistant/v1", model="gemini", tool_schemas=[{"name": "faq_lookup"}]
    )
    assert baseline != static_context_fingerprint(
        prompt_revision="assistant/v2", model="gemini", tool_schemas=[{"name": "faq_lookup"}]
    )


def test_tool_calling_assistant_explicitly_bypasses_gemini_cache() -> None:
    decision = decide_cache_policy(
        node_name="assistant",
        prompt_revision="assistant/v1",
        model="gemini",
        tool_schemas=[{"name": "faq_lookup"}],
    )

    assert decision.status == "bypass"
    assert decision.reason == "langchain_tool_binding_incompatible"
