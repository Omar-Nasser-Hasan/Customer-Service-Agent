"""Prompt owned by the evaluation subsystem, not a graph node."""

from customer_service.config.settings import Settings


def build_judge_prompt(settings: Settings, *, customer_message: str, reply: str) -> str:
    return f"""You are a strict quality evaluator for {settings.store_name} support.
Score the completed customer-facing answer from 1 to 5. Consider correctness,
helpfulness, concise tone, safety, and whether any optional promotion is clearly
separate and appropriate. Do not reward invented policy or hidden reasoning.
Return only the required structured result.

Customer message: {customer_message}
Support reply: {reply}"""
