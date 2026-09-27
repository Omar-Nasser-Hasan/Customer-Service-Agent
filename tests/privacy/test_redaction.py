from customer_service.privacy.redaction import redact_text, sanitize_for_trace, stable_thread_reference


def test_redact_text_removes_direct_identifiers_and_secret_values() -> None:
    text = "ORD-1001 alice@example.com +20 123 456 7890 4111 1111 1111 1111 token=abc123"

    redacted = redact_text(text)

    assert "ORD-1001" not in redacted
    assert "alice@example.com" not in redacted
    assert "4111" not in redacted
    assert "abc123" not in redacted
    assert "[ORDER_ID]" in redacted


def test_trace_sanitizer_redacts_recursively_and_hashes_thread_id() -> None:
    payload = {
        "thread_id": "customer-visible-thread",
        "authorization": "Bearer top-secret",
        "nested": ["alice@example.com", {"order": "ORD-1001"}],
    }

    sanitized = sanitize_for_trace(payload, thread_id_salt="test-salt")

    assert sanitized["thread_id"] == stable_thread_reference("customer-visible-thread", "test-salt")
    assert sanitized["authorization"] == "[SECRET]"
    assert sanitized["nested"] == ["[EMAIL]", {"order": "[ORDER_ID]"}]
