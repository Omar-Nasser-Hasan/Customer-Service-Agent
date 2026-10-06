from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from customer_service.verification.claims import (
    find_claim,
    latest_customer_text,
    normalize_order_id,
    partial_fields,
)


def human(text: str, message_id: str) -> HumanMessage:
    return HumanMessage(content=text, id=message_id)


def test_complete_pair_in_one_message_is_normalized() -> None:
    claim = find_claim([human("Where is ord-1001? I used Alice@Example.com", "m1")], None)

    assert claim is not None
    assert (claim.order_id, claim.email, claim.message_id) == (
        "ORD-1001",
        "alice@example.com",
        "m1",
    )


def test_pair_can_be_completed_by_the_message_just_before() -> None:
    claim = find_claim(
        [
            human("My order is ORD-1001", "m1"),
            AIMessage(content="Please also send your email."),
            human("alice@example.com", "m2"),
        ],
        None,
    )

    assert claim is not None and claim.message_id == "m2"
    assert (claim.order_id, claim.email) == ("ORD-1001", "alice@example.com")


@pytest.mark.parametrize("text", ["alice@example.com", "ORD-1001", "hello", ""])
def test_a_lone_field_is_never_a_claim(text: str) -> None:
    assert find_claim([human(text, "m1")], None) is None


def test_the_latest_message_must_contribute_a_field() -> None:
    messages = [
        human("ORD-1001 alice@example.com", "m1"),
        AIMessage(content="Let me check."),
        human("thanks", "m2"),
    ]

    assert find_claim(messages, None) is None


def test_spent_messages_cannot_complete_a_new_claim() -> None:
    messages = [
        human("ORD-1001 wrong@example.com", "m1"),
        AIMessage(content="I couldn't verify that."),
        human("sorry, it is alice@example.com", "m2"),
    ]

    assert find_claim(messages, "m1") is None
    # Without the spent marker the earlier order ID would have been reused.
    assert find_claim(messages, None) is not None


def test_a_spent_latest_message_is_not_a_claim_again() -> None:
    messages = [human("ORD-1001 alice@example.com", "m1")]

    assert find_claim(messages, "m1") is None


@pytest.mark.parametrize(
    "text",
    [
        "ORD-1001",
        "ord 1001",
        "ORD1001",
        "ORD-١٠٠١",
        "ORD-۱۰۰۱",
        "رقم طلبيORD-1001",
        "رقم الطلب: ORD-1001، شكرا",
    ],
)
def test_order_id_variants_normalize(text: str) -> None:
    assert normalize_order_id(text) == "ORD-1001"


@pytest.mark.parametrize("text", ["1001", "order 1001", "ORD-", "", "رقم الطلب ١٠٠١"])
def test_non_order_ids_do_not_normalize(text: str) -> None:
    assert normalize_order_id(text) is None


def test_email_next_to_arabic_text_is_extracted_cleanly() -> None:
    claim = find_claim([human("وبريديalice@example.com، ورقم طلبي ORD-١٠٠١", "m1")], None)

    assert claim is not None
    assert (claim.order_id, claim.email) == ("ORD-1001", "alice@example.com")


def test_partial_fields_report_what_unused_messages_contain() -> None:
    assert partial_fields([human("Track ORD-1001", "m1")], None) == (True, False)
    assert partial_fields([human("my email is a@b.co", "m1")], None) == (False, True)
    assert partial_fields([human("hello", "m1")], None) == (False, False)
    # A spent message no longer counts as partial information.
    assert partial_fields([human("Track ORD-1001", "m1")], "m1") == (False, False)


def test_latest_customer_text_ignores_assistant_messages() -> None:
    messages = [human("first", "m1"), AIMessage(content="reply"), human("second", "m2"), AIMessage(content="r")]

    assert latest_customer_text(messages) == "second"
    assert latest_customer_text([]) == ""
