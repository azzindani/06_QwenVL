"""Enterprise features module."""

# Lazy imports
def __getattr__(name):
    if name in ("TenantManager", "get_tenant_manager", "Tenant", "TenantTier", "ResourceQuota"):
        from .multitenancy import TenantManager, get_tenant_manager, Tenant, TenantTier, ResourceQuota
        return locals()[name]
    elif name in ("MetricsCollector", "get_metrics_collector", "RequestTimer"):
        from .monitoring import MetricsCollector, get_metrics_collector, RequestTimer
        return locals()[name]
    elif name in ("AuditLogger", "get_audit_logger", "AuditAction", "AuditEntry"):
        from .audit import AuditLogger, get_audit_logger, AuditAction, AuditEntry
        return locals()[name]
    elif name in ("AuthManager", "get_auth_manager", "Role", "Permission", "APIKey", "User"):
        from .auth import AuthManager, get_auth_manager, Role, Permission, APIKey, User
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Multitenancy
    "TenantManager",
    "get_tenant_manager",
    "Tenant",
    "TenantTier",
    "ResourceQuota",
    # Monitoring
    "MetricsCollector",
    "get_metrics_collector",
    "RequestTimer",
    # Audit
    "AuditLogger",
    "get_audit_logger",
    "AuditAction",
    "AuditEntry",
    # Auth
    "AuthManager",
    "get_auth_manager",
    "Role",
    "Permission",
    "APIKey",
    "User",
]
