"""Synthetic-only main-branch latency gate; never use real customer identifiers."""

from __future__ import annotations

from itertools import count

from locust import HttpUser, between, task

_SEQUENCE = count()


class SupportFlowUser(HttpUser):
    wait_time = between(0.05, 0.25)

    def on_start(self) -> None:
        self.sequence = next(_SEQUENCE)

    @task(4)
    def public_support(self) -> None:
        self._send("What payment methods do you accept?")

    @task(1)
    def verified_account(self) -> None:
        self._send("My order is ORD-1001 and my email is alice@example.com. Where is it?")

    def _send(self, message: str) -> None:
        thread_id = f"load-{self.sequence}-{next(_SEQUENCE)}"
        with self.client.post(
            f"/conversations/{thread_id}/messages",
            json={"message": message},
            name="/conversations/[thread_id]/messages",
            catch_response=True,
            timeout=30,
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected HTTP {response.status_code}")
                return
            payload = response.json()
            if payload.get("status") != "completed" or not payload.get("reply"):
                response.failure("expected a completed support reply")
