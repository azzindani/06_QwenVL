"""Multi-tenant architecture for document processing."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class TenantTier(str, Enum):
    """Tenant subscription tiers."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class ResourceQuota:
    """Resource quotas for a tenant."""
    max_requests_per_day: int = 100
    max_documents_per_batch: int = 10
    max_storage_mb: int = 100
    max_concurrent_jobs: int = 1
    allowed_tasks: List[str] = field(default_factory=lambda: ["ocr", "layout"])
    retention_days: int = 7


@dataclass
class Tenant:
    """Tenant/organization entity."""
    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.FREE
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workspace:
    """Workspace within a tenant."""
    workspace_id: str
    tenant_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_default: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Usage tracking for a tenant."""
    tenant_id: str
    date: str  # YYYY-MM-DD
    request_count: int = 0
    document_count: int = 0
    storage_used_mb: float = 0.0
    processing_time_ms: int = 0
    tokens_used: int = 0
    cost_cents: int = 0


# Default quotas by tier
TIER_QUOTAS = {
    TenantTier.FREE: ResourceQuota(
        max_requests_per_day=100,
        max_documents_per_batch=5,
        max_storage_mb=100,
        max_concurrent_jobs=1,
        allowed_tasks=["ocr", "layout"],
        retention_days=1,
    ),
    TenantTier.STARTER: ResourceQuota(
        max_requests_per_day=1000,
        max_documents_per_batch=20,
        max_storage_mb=1000,
        max_concurrent_jobs=2,
        allowed_tasks=["ocr", "layout", "table", "ner"],
        retention_days=7,
    ),
    TenantTier.PROFESSIONAL: ResourceQuota(
        max_requests_per_day=10000,
        max_documents_per_batch=100,
        max_storage_mb=10000,
        max_concurrent_jobs=5,
        allowed_tasks=["ocr", "layout", "table", "ner", "form", "invoice", "contract", "field_extraction"],
        retention_days=30,
    ),
    TenantTier.ENTERPRISE: ResourceQuota(
        max_requests_per_day=100000,
        max_documents_per_batch=1000,
        max_storage_mb=100000,
        max_concurrent_jobs=20,
        allowed_tasks=["ocr", "layout", "table", "ner", "form", "invoice", "contract", "field_extraction"],
        retention_days=365,
    ),
}


class TenantManager:
    """Manage tenants and their resources."""

    def __init__(self):
        """Initialize tenant manager."""
        self._tenants: Dict[str, Tenant] = {}
        self._workspaces: Dict[str, Workspace] = {}
        self._usage: Dict[str, Dict[str, Usage]] = {}  # tenant_id -> date -> usage

    def create_tenant(
        self,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        tenant_id: Optional[str] = None,
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            name: Tenant name
            tier: Subscription tier
            tenant_id: Optional specific ID

        Returns:
            Created tenant
        """
        tenant_id = tenant_id or str(uuid.uuid4())

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            quota=TIER_QUOTAS.get(tier, ResourceQuota()),
        )

        self._tenants[tenant_id] = tenant

        # Create default workspace
        self.create_workspace(tenant_id, "Default", is_default=True)

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)

    def update_tier(self, tenant_id: str, tier: TenantTier) -> bool:
        """
        Update tenant tier.

        Args:
            tenant_id: Tenant ID
            tier: New tier

        Returns:
            True if updated
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        tenant.tier = tier
        tenant.quota = TIER_QUOTAS.get(tier, ResourceQuota())
        return True

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate a tenant."""
        tenant = self.get_tenant(tenant_id)
        if tenant:
            tenant.is_active = False
            return True
        return False

    def create_workspace(
        self,
        tenant_id: str,
        name: str,
        is_default: bool = False,
    ) -> Optional[Workspace]:
        """
        Create a workspace for a tenant.

        Args:
            tenant_id: Tenant ID
            name: Workspace name
            is_default: Whether this is the default workspace

        Returns:
            Created workspace
        """
        if tenant_id not in self._tenants:
            return None

        workspace_id = str(uuid.uuid4())
        workspace = Workspace(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            name=name,
            is_default=is_default,
        )

        self._workspaces[workspace_id] = workspace
        return workspace

    def get_workspaces(self, tenant_id: str) -> List[Workspace]:
        """Get all workspaces for a tenant."""
        return [w for w in self._workspaces.values() if w.tenant_id == tenant_id]

    def check_quota(
        self,
        tenant_id: str,
        task_type: str,
        document_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Check if request is within quota.

        Args:
            tenant_id: Tenant ID
            task_type: Task being requested
            document_count: Number of documents

        Returns:
            Dict with allowed status and reason
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return {"allowed": False, "reason": "Tenant not found"}

        if not tenant.is_active:
            return {"allowed": False, "reason": "Tenant is deactivated"}

        # Check task allowed
        if task_type not in tenant.quota.allowed_tasks:
            return {
                "allowed": False,
                "reason": f"Task '{task_type}' not allowed in {tenant.tier.value} tier",
            }

        # Check document count
        if document_count > tenant.quota.max_documents_per_batch:
            return {
                "allowed": False,
                "reason": f"Batch size {document_count} exceeds limit {tenant.quota.max_documents_per_batch}",
            }

        # Check daily requests
        today = datetime.utcnow().strftime("%Y-%m-%d")
        usage = self.get_usage(tenant_id, today)
        if usage.request_count >= tenant.quota.max_requests_per_day:
            return {
                "allowed": False,
                "reason": f"Daily request limit ({tenant.quota.max_requests_per_day}) reached",
            }

        return {"allowed": True, "reason": None}

    def record_usage(
        self,
        tenant_id: str,
        request_count: int = 1,
        document_count: int = 0,
        processing_time_ms: int = 0,
        tokens_used: int = 0,
    ) -> None:
        """
        Record usage for a tenant.

        Args:
            tenant_id: Tenant ID
            request_count: Number of requests
            document_count: Number of documents processed
            processing_time_ms: Processing time
            tokens_used: Tokens consumed
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        usage = self.get_usage(tenant_id, today)

        usage.request_count += request_count
        usage.document_count += document_count
        usage.processing_time_ms += processing_time_ms
        usage.tokens_used += tokens_used

    def get_usage(self, tenant_id: str, date: str) -> Usage:
        """
        Get usage for a tenant on a date.

        Args:
            tenant_id: Tenant ID
            date: Date string (YYYY-MM-DD)

        Returns:
            Usage record
        """
        if tenant_id not in self._usage:
            self._usage[tenant_id] = {}

        if date not in self._usage[tenant_id]:
            self._usage[tenant_id][date] = Usage(tenant_id=tenant_id, date=date)

        return self._usage[tenant_id][date]

    def get_usage_summary(
        self,
        tenant_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get usage summary for a tenant.

        Args:
            tenant_id: Tenant ID
            days: Number of days to summarize

        Returns:
            Usage summary
        """
        if tenant_id not in self._usage:
            return {
                "total_requests": 0,
                "total_documents": 0,
                "total_processing_ms": 0,
            }

        total_requests = 0
        total_documents = 0
        total_processing = 0

        for usage in self._usage[tenant_id].values():
            total_requests += usage.request_count
            total_documents += usage.document_count
            total_processing += usage.processing_time_ms

        return {
            "total_requests": total_requests,
            "total_documents": total_documents,
            "total_processing_ms": total_processing,
            "avg_processing_ms": total_processing // max(total_requests, 1),
        }


# Global tenant manager
_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get or create global tenant manager."""
    global _manager
    if _manager is None:
        _manager = TenantManager()
    return _manager


if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-TENANCY TEST")
    print("=" * 60)

    manager = TenantManager()

    # Create tenant
    tenant = manager.create_tenant("Acme Corp", TenantTier.PROFESSIONAL)
    print(f"  Tenant: {tenant.name} ({tenant.tier.value})")
    print(f"  Max requests/day: {tenant.quota.max_requests_per_day}")

    # Check quota
    result = manager.check_quota(tenant.tenant_id, "ocr", 5)
    print(f"  OCR allowed: {result['allowed']}")

    # Record usage
    manager.record_usage(tenant.tenant_id, request_count=10)
    usage = manager.get_usage(tenant.tenant_id, datetime.utcnow().strftime("%Y-%m-%d"))
    print(f"  Requests today: {usage.request_count}")

    print("=" * 60)
