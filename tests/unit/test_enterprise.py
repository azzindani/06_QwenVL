"""Unit tests for enterprise features."""

import pytest
from datetime import datetime, timedelta

from qwen_vl.enterprise.multitenancy import (
    TenantManager,
    TenantTier,
    ResourceQuota,
    TIER_QUOTAS,
)
from qwen_vl.enterprise.monitoring import (
    MetricsCollector,
    RequestTimer,
)
from qwen_vl.enterprise.audit import (
    AuditLogger,
    AuditAction,
)
from qwen_vl.enterprise.auth import (
    AuthManager,
    Role,
    Permission,
    ROLE_PERMISSIONS,
)


@pytest.mark.unit
class TestMultitenancy:
    """Tests for multi-tenant architecture."""

    def test_create_tenant(self):
        """Test creating a tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Test Corp", TenantTier.PROFESSIONAL)

        assert tenant.name == "Test Corp"
        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.is_active

    def test_tier_quotas(self):
        """Test tier-specific quotas."""
        manager = TenantManager()

        free = manager.create_tenant("Free", TenantTier.FREE)
        pro = manager.create_tenant("Pro", TenantTier.PROFESSIONAL)

        assert free.quota.max_requests_per_day < pro.quota.max_requests_per_day
        assert len(free.quota.allowed_tasks) < len(pro.quota.allowed_tasks)

    def test_create_workspace(self):
        """Test creating workspaces."""
        manager = TenantManager()
        tenant = manager.create_tenant("Test Corp")

        workspace = manager.create_workspace(tenant.tenant_id, "Development")

        workspaces = manager.get_workspaces(tenant.tenant_id)
        assert len(workspaces) == 2  # Default + Development

    def test_check_quota_allowed(self):
        """Test quota check when allowed."""
        manager = TenantManager()
        tenant = manager.create_tenant("Test", TenantTier.PROFESSIONAL)

        result = manager.check_quota(tenant.tenant_id, "ocr", 5)
        assert result["allowed"]

    def test_check_quota_task_not_allowed(self):
        """Test quota check for disallowed task."""
        manager = TenantManager()
        tenant = manager.create_tenant("Test", TenantTier.FREE)

        result = manager.check_quota(tenant.tenant_id, "invoice", 1)
        assert not result["allowed"]
        assert "not allowed" in result["reason"]

    def test_check_quota_batch_exceeded(self):
        """Test quota check for exceeded batch size."""
        manager = TenantManager()
        tenant = manager.create_tenant("Test", TenantTier.FREE)

        result = manager.check_quota(tenant.tenant_id, "ocr", 100)
        assert not result["allowed"]
        assert "exceeds" in result["reason"]

    def test_record_usage(self):
        """Test usage recording."""
        manager = TenantManager()
        tenant = manager.create_tenant("Test")

        manager.record_usage(tenant.tenant_id, request_count=10, document_count=5)

        today = datetime.utcnow().strftime("%Y-%m-%d")
        usage = manager.get_usage(tenant.tenant_id, today)

        assert usage.request_count == 10
        assert usage.document_count == 5

    def test_update_tier(self):
        """Test updating tenant tier."""
        manager = TenantManager()
        tenant = manager.create_tenant("Test", TenantTier.FREE)

        assert manager.update_tier(tenant.tenant_id, TenantTier.ENTERPRISE)
        assert tenant.tier == TenantTier.ENTERPRISE

    def test_deactivate_tenant(self):
        """Test deactivating tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Test")

        manager.deactivate_tenant(tenant.tenant_id)

        result = manager.check_quota(tenant.tenant_id, "ocr", 1)
        assert not result["allowed"]


@pytest.mark.unit
class TestMonitoring:
    """Tests for monitoring metrics."""

    def test_counter(self):
        """Test counter metric."""
        collector = MetricsCollector()

        collector.increment_counter("requests")
        collector.increment_counter("requests", 5)

        assert collector.get_counter("requests") == 6

    def test_counter_with_labels(self):
        """Test counter with labels."""
        collector = MetricsCollector()

        collector.increment_counter("requests", labels={"task": "ocr"})
        collector.increment_counter("requests", labels={"task": "table"})

        assert collector.get_counter("requests", {"task": "ocr"}) == 1
        assert collector.get_counter("requests", {"task": "table"}) == 1

    def test_gauge(self):
        """Test gauge metric."""
        collector = MetricsCollector()

        collector.set_gauge("active_jobs", 5)
        assert collector.get_gauge("active_jobs") == 5

        collector.set_gauge("active_jobs", 3)
        assert collector.get_gauge("active_jobs") == 3

    def test_histogram(self):
        """Test histogram metric."""
        collector = MetricsCollector()

        for value in [10, 20, 30, 40, 50]:
            collector.observe_histogram("latency", value)

        stats = collector.get_histogram_stats("latency")

        assert stats["count"] == 5
        assert stats["sum"] == 150
        assert stats["avg"] == 30
        assert stats["min"] == 10
        assert stats["max"] == 50

    def test_prometheus_export(self):
        """Test Prometheus format export."""
        collector = MetricsCollector()

        collector.increment_counter("requests")
        collector.set_gauge("jobs", 3)

        output = collector.export_prometheus()

        assert "requests" in output
        assert "jobs" in output

    def test_request_timer(self):
        """Test request timer context manager."""
        import time

        collector = MetricsCollector()

        with RequestTimer(collector, "test_request"):
            time.sleep(0.01)

        stats = collector.get_histogram_stats("test_request")
        assert stats["count"] == 1
        assert stats["avg"] > 0


@pytest.mark.unit
class TestAuditLogging:
    """Tests for audit logging."""

    def test_log_entry(self):
        """Test logging an audit entry."""
        logger = AuditLogger()

        entry = logger.log(
            action=AuditAction.DOCUMENT_UPLOADED,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        assert entry.action == AuditAction.DOCUMENT_UPLOADED
        assert entry.tenant_id == "tenant-1"

    def test_query_by_tenant(self):
        """Test querying entries by tenant."""
        logger = AuditLogger()

        logger.log(AuditAction.DOCUMENT_UPLOADED, tenant_id="tenant-1")
        logger.log(AuditAction.DOCUMENT_UPLOADED, tenant_id="tenant-2")
        logger.log(AuditAction.DOCUMENT_UPLOADED, tenant_id="tenant-1")

        entries = logger.query(tenant_id="tenant-1")
        assert len(entries) == 2

    def test_query_by_action(self):
        """Test querying entries by action."""
        logger = AuditLogger()

        logger.log(AuditAction.DOCUMENT_UPLOADED)
        logger.log(AuditAction.DOCUMENT_PROCESSED)
        logger.log(AuditAction.DOCUMENT_UPLOADED)

        entries = logger.query(action=AuditAction.DOCUMENT_UPLOADED)
        assert len(entries) == 2

    def test_compliance_report(self):
        """Test generating compliance report."""
        logger = AuditLogger()

        logger.log(AuditAction.DOCUMENT_UPLOADED, tenant_id="tenant-1", user_id="user-1")
        logger.log(AuditAction.DOCUMENT_PROCESSED, tenant_id="tenant-1", user_id="user-1")
        logger.log(AuditAction.DOCUMENT_EXPORTED, tenant_id="tenant-1", user_id="user-2")

        report = logger.get_compliance_report(
            tenant_id="tenant-1",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2025, 12, 31),
        )

        assert report["total_events"] == 3
        assert report["unique_users"] == 2

    def test_export_json(self):
        """Test exporting to JSON."""
        logger = AuditLogger()

        logger.log(AuditAction.DOCUMENT_UPLOADED)

        output = logger.export_entries(format="json")
        assert "document.uploaded" in output


@pytest.mark.unit
class TestAuth:
    """Tests for authentication and authorization."""

    def test_create_api_key(self):
        """Test creating an API key."""
        manager = AuthManager()

        raw_key, api_key = manager.create_api_key(
            tenant_id="tenant-1",
            name="Test Key",
        )

        assert raw_key.startswith("qwvl_")
        assert api_key.is_active

    def test_validate_api_key(self):
        """Test validating an API key."""
        manager = AuthManager()

        raw_key, _ = manager.create_api_key("tenant-1", "Test")

        validated = manager.validate_api_key(raw_key)
        assert validated is not None

        invalid = manager.validate_api_key("invalid_key")
        assert invalid is None

    def test_revoke_api_key(self):
        """Test revoking an API key."""
        manager = AuthManager()

        raw_key, api_key = manager.create_api_key("tenant-1", "Test")

        manager.revoke_api_key(api_key.key_id)

        validated = manager.validate_api_key(raw_key)
        assert validated is None

    def test_create_user(self):
        """Test creating a user."""
        manager = AuthManager()

        user = manager.create_user(
            tenant_id="tenant-1",
            email="test@example.com",
            password="secret",
            role=Role.USER,
        )

        assert user.email == "test@example.com"
        assert user.role == Role.USER

    def test_authenticate_user(self):
        """Test user authentication."""
        manager = AuthManager()

        manager.create_user("tenant-1", "test@example.com", "secret")

        user = manager.authenticate_user("test@example.com", "secret")
        assert user is not None

        invalid = manager.authenticate_user("test@example.com", "wrong")
        assert invalid is None

    def test_role_permissions(self):
        """Test role-based permissions."""
        manager = AuthManager()

        admin = manager.create_user("t1", "admin@test.com", "pw", Role.ADMIN)
        user = manager.create_user("t1", "user@test.com", "pw", Role.USER)
        readonly = manager.create_user("t1", "ro@test.com", "pw", Role.READONLY)

        # Admin has all permissions
        assert manager.check_permission(admin.user_id, Permission.TENANT_MANAGE)

        # User has basic permissions
        assert manager.check_permission(user.user_id, Permission.DOCUMENT_WRITE)
        assert not manager.check_permission(user.user_id, Permission.USER_MANAGE)

        # Readonly has limited permissions
        assert manager.check_permission(readonly.user_id, Permission.DOCUMENT_READ)
        assert not manager.check_permission(readonly.user_id, Permission.DOCUMENT_WRITE)

    def test_rate_limiting(self):
        """Test rate limiting."""
        manager = AuthManager()

        # First request allowed
        result = manager.check_rate_limit("key", limit=2)
        assert result["allowed"]
        assert result["remaining"] == 1

        # Second request allowed
        result = manager.check_rate_limit("key", limit=2)
        assert result["allowed"]
        assert result["remaining"] == 0

        # Third request blocked
        result = manager.check_rate_limit("key", limit=2)
        assert not result["allowed"]

    def test_expired_api_key(self):
        """Test expired API key validation."""
        manager = AuthManager()

        raw_key, api_key = manager.create_api_key(
            "tenant-1",
            "Test",
            expires_in_days=1,
        )

        # Manually expire
        api_key.expires_at = datetime.utcnow() - timedelta(hours=1)

        validated = manager.validate_api_key(raw_key)
        assert validated is None
