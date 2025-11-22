"""Unit tests for webhook integration."""

import pytest

from qwen_vl.api.webhooks import (
    WebhookManager,
    WebhookConfig,
    EventType,
    get_webhook_manager,
)


@pytest.mark.unit
class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_create_config(self):
        """Test creating webhook configuration."""
        config = WebhookConfig(
            webhook_id="test-webhook",
            url="https://example.com/webhook",
            events=[EventType.EXTRACTION_COMPLETED],
        )

        assert config.webhook_id == "test-webhook"
        assert config.active is True
        assert config.retry_count == 3

    def test_config_with_secret(self):
        """Test configuration with secret."""
        config = WebhookConfig(
            webhook_id="test",
            url="https://example.com/webhook",
            events=[EventType.BATCH_COMPLETED],
            secret="my-secret",
        )

        assert config.secret == "my-secret"


@pytest.mark.unit
class TestWebhookManager:
    """Tests for WebhookManager."""

    def test_register_webhook(self):
        """Test registering a webhook."""
        manager = WebhookManager()

        config = manager.register_webhook(
            webhook_id="test-1",
            url="https://example.com/hook",
            events=[EventType.EXTRACTION_COMPLETED],
        )

        assert config.webhook_id == "test-1"
        assert manager.get_webhook("test-1") is not None

    def test_unregister_webhook(self):
        """Test unregistering a webhook."""
        manager = WebhookManager()

        manager.register_webhook(
            webhook_id="test-1",
            url="https://example.com/hook",
            events=[EventType.EXTRACTION_COMPLETED],
        )

        assert manager.unregister_webhook("test-1") is True
        assert manager.get_webhook("test-1") is None

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent webhook."""
        manager = WebhookManager()
        assert manager.unregister_webhook("nonexistent") is False

    def test_list_webhooks(self):
        """Test listing all webhooks."""
        manager = WebhookManager()

        manager.register_webhook(
            webhook_id="test-1",
            url="https://example.com/hook1",
            events=[EventType.EXTRACTION_COMPLETED],
        )
        manager.register_webhook(
            webhook_id="test-2",
            url="https://example.com/hook2",
            events=[EventType.BATCH_COMPLETED],
        )

        webhooks = manager.list_webhooks()
        assert len(webhooks) == 2

    def test_signature_generation(self):
        """Test HMAC signature generation."""
        manager = WebhookManager()

        signature = manager._generate_signature(
            '{"event": "test"}',
            "secret-key",
        )

        assert len(signature) == 64  # SHA256 hex
        assert signature.isalnum()

    def test_get_deliveries_empty(self):
        """Test getting deliveries when empty."""
        manager = WebhookManager()
        deliveries = manager.get_deliveries()
        assert deliveries == []


@pytest.mark.unit
class TestEventTypes:
    """Tests for EventType enum."""

    def test_event_types_exist(self):
        """Test all event types are defined."""
        assert EventType.EXTRACTION_STARTED
        assert EventType.EXTRACTION_COMPLETED
        assert EventType.EXTRACTION_FAILED
        assert EventType.BATCH_STARTED
        assert EventType.BATCH_PROGRESS
        assert EventType.BATCH_COMPLETED
        assert EventType.BATCH_FAILED

    def test_event_type_values(self):
        """Test event type string values."""
        assert EventType.EXTRACTION_COMPLETED.value == "extraction.completed"
        assert EventType.BATCH_COMPLETED.value == "batch.completed"


@pytest.mark.unit
class TestGlobalManager:
    """Tests for global webhook manager."""

    def test_get_webhook_manager_singleton(self):
        """Test singleton behavior."""
        import qwen_vl.api.webhooks as webhook_module
        webhook_module._manager = None

        m1 = get_webhook_manager()
        m2 = get_webhook_manager()

        assert m1 is m2

        webhook_module._manager = None
