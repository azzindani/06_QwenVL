"""Webhook integration for event notifications."""

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import httpx


class EventType(str, Enum):
    """Webhook event types."""
    EXTRACTION_STARTED = "extraction.started"
    EXTRACTION_COMPLETED = "extraction.completed"
    EXTRACTION_FAILED = "extraction.failed"
    BATCH_STARTED = "batch.started"
    BATCH_PROGRESS = "batch.progress"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"


@dataclass
class WebhookConfig:
    """Webhook endpoint configuration."""
    webhook_id: str
    url: str
    events: List[EventType]
    secret: Optional[str] = None
    active: bool = True
    retry_count: int = 3
    retry_delay_seconds: int = 5
    timeout_seconds: int = 30
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    delivery_id: str
    webhook_id: str
    event_type: EventType
    payload: Dict[str, Any]
    status: str  # pending, success, failed
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None


class WebhookManager:
    """Manage webhook subscriptions and deliveries."""

    def __init__(self):
        """Initialize webhook manager."""
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._deliveries: List[WebhookDelivery] = []
        self._client = httpx.AsyncClient()

    def register_webhook(
        self,
        webhook_id: str,
        url: str,
        events: List[EventType],
        secret: Optional[str] = None,
        **kwargs,
    ) -> WebhookConfig:
        """
        Register a new webhook endpoint.

        Args:
            webhook_id: Unique webhook identifier
            url: Endpoint URL
            events: List of events to subscribe to
            secret: Secret for signature verification
            **kwargs: Additional configuration

        Returns:
            Created webhook configuration
        """
        config = WebhookConfig(
            webhook_id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            **kwargs,
        )
        self._webhooks[webhook_id] = config
        return config

    def unregister_webhook(self, webhook_id: str) -> bool:
        """
        Unregister a webhook.

        Args:
            webhook_id: Webhook to remove

        Returns:
            True if removed
        """
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            return True
        return False

    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook configuration."""
        return self._webhooks.get(webhook_id)

    def list_webhooks(self) -> List[WebhookConfig]:
        """List all registered webhooks."""
        return list(self._webhooks.values())

    async def trigger_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[WebhookDelivery]:
        """
        Trigger a webhook event.

        Args:
            event_type: Type of event
            data: Event data
            job_id: Associated job ID
            document_id: Associated document ID

        Returns:
            List of delivery records
        """
        payload = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "document_id": document_id,
            "data": data,
        }

        # Find subscribed webhooks
        webhooks = [
            w for w in self._webhooks.values()
            if w.active and event_type in w.events
        ]

        # Send to all subscribed webhooks
        deliveries = []
        for webhook in webhooks:
            delivery = await self._send_webhook(webhook, payload)
            deliveries.append(delivery)
            self._deliveries.append(delivery)

        return deliveries

    async def _send_webhook(
        self,
        webhook: WebhookConfig,
        payload: Dict[str, Any],
    ) -> WebhookDelivery:
        """Send webhook with retry logic."""
        import uuid

        delivery = WebhookDelivery(
            delivery_id=str(uuid.uuid4()),
            webhook_id=webhook.webhook_id,
            event_type=EventType(payload["event_type"]),
            payload=payload,
            status="pending",
        )

        body = json.dumps(payload)
        headers = {
            "Content-Type": "application/json",
            **webhook.headers,
        }

        # Add signature if secret is configured
        if webhook.secret:
            signature = self._generate_signature(body, webhook.secret)
            headers["X-Webhook-Signature"] = signature

        # Retry loop
        for attempt in range(webhook.retry_count):
            delivery.attempts = attempt + 1
            delivery.last_attempt = datetime.utcnow()

            try:
                response = await self._client.post(
                    webhook.url,
                    content=body,
                    headers=headers,
                    timeout=webhook.timeout_seconds,
                )

                delivery.response_code = response.status_code
                delivery.response_body = response.text[:1000]

                if response.is_success:
                    delivery.status = "success"
                    return delivery

            except Exception as e:
                delivery.error = str(e)

            # Wait before retry
            if attempt < webhook.retry_count - 1:
                await asyncio.sleep(webhook.retry_delay_seconds * (attempt + 1))

        delivery.status = "failed"
        return delivery

    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for payload."""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    def get_deliveries(
        self,
        webhook_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[WebhookDelivery]:
        """
        Get webhook delivery history.

        Args:
            webhook_id: Filter by webhook
            status: Filter by status
            limit: Maximum results

        Returns:
            List of deliveries
        """
        deliveries = self._deliveries

        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        if status:
            deliveries = [d for d in deliveries if d.status == status]

        return deliveries[-limit:]

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()


# Global webhook manager
_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get or create the global webhook manager."""
    global _manager
    if _manager is None:
        _manager = WebhookManager()
    return _manager


if __name__ == "__main__":
    print("=" * 60)
    print("WEBHOOK MANAGER TEST")
    print("=" * 60)

    manager = WebhookManager()

    # Register a webhook
    webhook = manager.register_webhook(
        webhook_id="test-webhook",
        url="https://example.com/webhook",
        events=[EventType.EXTRACTION_COMPLETED, EventType.BATCH_COMPLETED],
        secret="my-secret-key",
    )

    print(f"  Webhook ID: {webhook.webhook_id}")
    print(f"  URL: {webhook.url}")
    print(f"  Events: {[e.value for e in webhook.events]}")

    # List webhooks
    webhooks = manager.list_webhooks()
    print(f"  Total webhooks: {len(webhooks)}")

    print("=" * 60)
