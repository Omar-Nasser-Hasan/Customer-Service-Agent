"""Fixed customer-facing wording for identity checks and human handoff.

The wording is deterministic on purpose: a failed check must never reveal which
field was wrong, and the model must not improvise security messages. The Arabic
texts are committed development content and need business-language review
before a real customer launch.
"""

from __future__ import annotations

import re
from typing import Literal

Language = Literal["en", "ar"]

_ARABIC_SCRIPT = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")

CATALOG: dict[str, dict[Language, str]] = {
    "ask_both": {
        "en": "To protect your account, please send both your order ID and the email used for the order.",
        "ar": "لحماية حسابك، يرجى إرسال رقم طلبك والبريد الإلكتروني المستخدم في الطلب.",
    },
    "ask_email": {
        "en": "To protect your account, please also send the email used for the order.",
        "ar": "لحماية حسابك، يرجى إرسال البريد الإلكتروني المستخدم في الطلب أيضًا.",
    },
    "ask_order_id": {
        "en": "To protect your account, please also send your order ID.",
        "ar": "لحماية حسابك، يرجى إرسال رقم طلبك أيضًا.",
    },
    "expired": {
        "en": "For your security, please send your order ID and the email used for the order again to continue.",
        "ar": "حفاظًا على أمان حسابك، يرجى إعادة إرسال رقم طلبك والبريد الإلكتروني المستخدم في الطلب للمتابعة.",
    },
    # Deliberately silent about which field was wrong.
    "claim_failed": {
        "en": "I couldn't verify that order ID and email combination. Please check both and try again.",
        "ar": "لم أتمكن من التحقق من رقم الطلب والبريد الإلكتروني المُدخلين. يرجى التأكد من كليهما والمحاولة مرة أخرى.",
    },
    # Identical for an order that does not exist and one that belongs to someone else.
    "order_not_found": {
        "en": "I couldn't find that order on your account. Please check the order ID and try again.",
        "ar": "لم أتمكن من العثور على هذا الطلب في حسابك. يرجى التأكد من رقم الطلب والمحاولة مرة أخرى.",
    },
    "handoff_ack": {
        "en": "I'm connecting you with a support specialist. They'll continue here shortly.",
        "ar": "سأوصلك بأحد مختصي الدعم، وسيتابع معك هنا قريبًا.",
    },
}


def detect_language(text: str) -> Language:
    """Arabic when the text contains any Arabic-script character, otherwise English."""

    return "ar" if _ARABIC_SCRIPT.search(text) else "en"


def message(key: str, language: Language) -> str:
    return CATALOG[key][language]
