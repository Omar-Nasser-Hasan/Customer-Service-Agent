# Identity verification

How a customer proves who they are before the bot discusses their orders.

## What "verified" means

A customer is verified when they sent an **order ID and the email on that order**
and the pair matched. That proves control of the email, so a verified customer may
ask about **any order placed under that email** with just the order ID. Nothing
else in the database is reachable.

State holds `verified_email` and `verified_at`. A grant is valid for
`VERIFICATION_TTL_SECONDS` (default 2 hours) from `verified_at`; it is fixed, not
sliding. After that the customer is asked for the pair again. One helper,
`VerificationPolicy`, decides validity and ownership for the claim node, the gate,
the tools, the assistant context, the promo prefilter, and the handoff summary, so
they can never disagree about the clock or the TTL.

## Flow

```
safety_check ──► verify_identity ──► assistant ──► account_gate ──► tools ──► assistant
                 (only if a fresh          │         (only for account
                  full pair exists)        │          tool calls)
                                           └──► promo_prefilter ...
```

1. **`verify_identity` (before the assistant).** Runs only when the latest customer
   message, optionally completed by the one customer message before it, contains a
   complete order ID + email pair that has not been used yet. Success sets the
   identity (replacing any earlier one). A mismatch counts one failure and ends the
   turn with a fixed message. An existing verification is left untouched. A lone
   email or lone order ID is never a claim, so a stray email cannot switch identity.
2. **`account_gate` (after the assistant, before tools).** Every `order_status`,
   `returns`, and `billing` call needs a valid grant **and** an order owned by the
   verified email. If any call fails, every call in that AI message gets a
   "not executed" tool response (no dangling calls) and the turn ends with a fixed
   message.
3. **Tools.** Defence in depth: the tools receive `verified_email` / `verified_at`
   from graph state (injected, invisible to the model) and apply the same check.
   Expired, unverified, someone else's, and nonexistent orders all return the same
   "not found" payload.

FAQ-only chat never touches steps 1 or 2. The happy path is two model calls (tool
call, answer).

## Failure handling

| Event | Counted? | Customer sees |
| --- | --- | --- |
| Pair doesn't match | yes | "I couldn't verify that order ID and email combination…" (never says which field) |
| Verified, asks for an order that isn't theirs or doesn't exist | yes | "I couldn't find that order on your account…" (identical for both cases) |
| Not verified / expired | no | asks for the missing field(s), or for the pair again |
| Order ID the model could not even parse | no | same "couldn't find" message |
| Lookup or limiter infrastructure error | no | handoff (`tool_failure`) |

* **Per thread:** the 3rd counted failure inside 24 hours opens a human handoff with
  reason `verification_failed`. Counters clear only when a human resolves the
  handoff, never on a successful verification.
* **Across threads and channels:** failures are also counted per **hashed email** and
  per **hashed order ID** in the `verification_failures` table. After 5 per key in
  24 hours, further claims for that key go to a human, even with a correct pair. A
  denial for a verified customer is recorded under their email only, so a verified
  user cannot lock the real owner of an order they probe.
* Emails and order IDs are stored only as HMAC-SHA256 digests keyed by
  `VERIFICATION_HASH_KEY`. Logs carry outcomes and counters only.

## Language

Replies use Arabic when the customer's latest message contains Arabic script,
otherwise English. Arabic-Indic digits and `ORD1001` / `ord 1001` forms are
normalized. All wording lives in `verification/messages.py`. **The Arabic text needs
business-language review before launch.**

## Promotions

A promotion is considered only when the reply is backed by a **found** tool result
from the same turn (a found FAQ or a found order). Greetings, requests for
information, refusals, and "I can't help" replies never get one. This reads tool
output, not keywords, so it works in Arabic.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `BOT_API_KEY` | unset | `X-API-Key` for `/conversations/...`. Required when `APP_ENVIRONMENT=production`. |
| `VERIFICATION_HASH_KEY` | dev constant | Required in production. **API and worker must share it.** |
| `VERIFICATION_TTL_SECONDS` | 7200 | Fixed lifetime of a verification. |
| `VERIFICATION_MAX_THREAD_FAILURES` | 3 | Per-thread failures in 24h before handoff. |
| `VERIFICATION_KEY_FAILURE_LIMIT` | 5 | Per-email / per-order failures in 24h before handoff. |

## Known limits

* A spray of many different emails against many different orders is not caught by
  per-key counters. Put a network-level rate limit in front of the API and Meta's
  webhook.
* Blocking is per key, so someone can deliberately burn 5 guesses against a real
  customer's email to send that customer to a human for 24 hours. They get a human,
  not a lockout from support.
* The safety refusal and the worker's "please send your question as text" reply are
  still English-only.
