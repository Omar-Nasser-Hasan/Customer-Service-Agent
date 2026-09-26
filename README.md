# WhatsApp customer-service agent — Iteration 4

This iteration adds an optional promotion branch after a completed support
answer. It retains public FAQ lookup and verified account support for order
status, returns, and billing, and adds a hybrid safety gate plus durable human
handoff. It still deliberately excludes WhatsApp/Meta integration, admin
routes/UI, authentication, production promotions, vector retrieval, tracing,
and load testing.

## Supported support flows

- `faq_lookup` answers public shipping, return-policy, and payment-method FAQs
  without identity verification.
- `order_status`, `returns`, and `billing` require an order ID plus the email
  used for that order. A successful match remains valid for the current
  durable conversation thread.
- The return rule is deterministic: an order must be delivered and is eligible
  through 14 calendar days after its recorded delivery date. This release only
  reports eligibility; it does not create a return.
- Once a support response is complete, the bot may add at most one optional
  promotion per conversation. A deterministic prefilter rejects negative
  English-language sentiment, inactive/expired offers, duplicate promotions,
  and unmatched offers before the separate `promo_judge` model is called.
- `safety_check` runs before the assistant. It refuses clear prompt-injection
  and abusive messages, and opens a durable human handoff for an explicit
  human request, defined return exceptions, billing disputes, safety
  uncertainty, or model/tool failure.
- A newly escalated conversation receives one fixed acknowledgement. While its
  handoff remains open, later customer messages receive `202 handoff_active`
  with no bot reply. The future authenticated admin surface will resolve the
  LangGraph interrupt through `HandoffService`.

## Promotion prototype warning

`src/customer_service/data/promotions.json` contains **synthetic development
offers only**. The catalog text visibly identifies itself as a development
sample and must not be enabled for live customers. Topic tags and the English
negative-sentiment word list are prototype heuristics: they do not reliably
handle misspellings, synonyms, multilingual messages, nuanced dissatisfaction,
or real offer eligibility.

Before launch, replace the catalog with approved live promotions, tune the
branch using anonymized real customer transcripts, and replace local matching
with the planned semantic/vector retrieval implementation. The promo branch
uses only conversation topic plus derived order/delivery state; it never sends
raw customer email, name, billing amount, or payment data to matching or the
promo-judge model.

The local fixture identities are intentionally non-production:

| Order ID | Email |
| --- | --- |
| `ORD-1001` | `alice@example.com` |
| `ORD-1002` | `ben@example.com` |
| `ORD-1003` | `carla@example.com` |

## Run locally

1. Create and activate a Python 3.12+ virtual environment.
2. Install the application and test dependencies:

   ```powershell
   python -m pip install -e ".[dev]"
   ```

3. Start local Postgres for durable checkpoints:

   ```powershell
   docker compose up -d
   ```

4. Use `.env.example` as a template, then set `GOOGLE_API_KEY`. The selected
   model is configured per node through `NODE_MODELS__<NODE>__MODEL`; the
   assistant defaults to `gemini-3.1-flash-lite`. Settings load from the
   process environment and, for local development, an optional `.env` file.
5. Start the service:

   ```powershell
   python -m uvicorn customer_service.api.app:app --reload
   ```

6. Send a message using a stable, caller-chosen conversation ID:

   ```powershell
   Invoke-RestMethod -Method Post `
     -Uri http://127.0.0.1:8000/conversations/demo-1/messages `
     -ContentType 'application/json' `
     -Body '{"message":"Where is order ORD-1001?"}'
   ```

Run the offline test suite with:

```powershell
python -m pytest -q -p no:cacheprovider
```

## API

`POST /conversations/{thread_id}/messages`

Request:

```json
{"message": "Where is order ORD-1001? My email is alice@example.com."}
```

Response:

```json
{"thread_id": "demo-1", "status": "completed", "reply": "..."}
```

An initial escalation returns `202` with `status: "handoff_open"` and the fixed
specialist acknowledgement. Later messages on that open thread return `202`
with `status: "handoff_active"` and `reply: null`.

Conversation history, verification, and handoff state are stored in local
Postgres through `AsyncPostgresSaver`. `LANGGRAPH_STRICT_MSGPACK=true` is
enforced at runtime to restrict checkpoint deserialization. Run the manual
retention command to remove resolved/inactive threads older than 90 days:

```powershell
python -m customer_service.retention
```

Open and claimed handoffs are never pruned. Scheduling retention, staff auth,
admin case actions, audit logs, and WhatsApp/Meta transport are intentionally
deferred to later iterations.
