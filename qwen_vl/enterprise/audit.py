"""Audit logging for compliance and tracking."""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AuditAction(str, Enum):
    """Types of auditable actions."""
    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    TOKEN_CREATED = "auth.token_created"
    TOKEN_REVOKED = "auth.token_revoked"

    # Document operations
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_EXPORTED = "document.exported"

    # Configuration
    CONFIG_CHANGED = "config.changed"
    QUOTA_UPDATED = "quota.updated"

    # Administration
    USER_CREATED = "admin.user_created"
    USER_UPDATED = "admin.user_updated"
    USER_DELETED = "admin.user_deleted"
    TENANT_CREATED = "admin.tenant_created"
    TENANT_UPDATED = "admin.tenant_updated"


@dataclass
class AuditEntry:
    """Single audit log entry."""
    entry_id: str
    timestamp: datetime
    action: AuditAction
    tenant_id: Optional[str]
    user_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["action"] = self.action.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """Log and query audit events."""

    def __init__(self, max_entries: int = 10000):
        """
        Initialize audit logger.

        Args:
            max_entries: Maximum entries to keep in memory
        """
        self._entries: List[AuditEntry] = []
        self._max_entries = max_entries
        self._handlers: List[callable] = []

    def log(
        self,
        action: AuditAction,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditEntry:
        """
        Log an audit event.

        Args:
            action: Type of action
            tenant_id: Tenant ID
            user_id: User ID
            resource_type: Type of resource
            resource_id: Resource ID
            ip_address: Client IP
            user_agent: Client user agent
            details: Additional details
            success: Whether action succeeded
            error_message: Error message if failed

        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            action=action,
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            success=success,
            error_message=error_message,
        )

        self._entries.append(entry)

        # Trim if over limit
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception:
                pass

        return entry

    def add_handler(self, handler: callable) -> None:
        """Add a handler to be called for each audit entry."""
        self._handlers.append(handler)

    def query(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Query audit entries.

        Args:
            tenant_id: Filter by tenant
            user_id: Filter by user
            action: Filter by action
            resource_type: Filter by resource type
            start_time: Filter by start time
            end_time: Filter by end time
            success_only: Only successful actions
            limit: Maximum results

        Returns:
            List of matching entries
        """
        results = []

        for entry in reversed(self._entries):
            if tenant_id and entry.tenant_id != tenant_id:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if action and entry.action != action:
                continue
            if resource_type and entry.resource_type != resource_type:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if success_only and not entry.success:
                continue

            results.append(entry)

            if len(results) >= limit:
                break

        return results

    def get_compliance_report(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """
        Generate compliance report for a tenant.

        Args:
            tenant_id: Tenant ID
            start_time: Report start
            end_time: Report end

        Returns:
            Compliance report dict
        """
        entries = self.query(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        # Aggregate by action
        action_counts: Dict[str, int] = {}
        for entry in entries:
            action_counts[entry.action.value] = action_counts.get(entry.action.value, 0) + 1

        # Count users
        users = set(e.user_id for e in entries if e.user_id)

        # Count failures
        failures = sum(1 for e in entries if not e.success)

        return {
            "tenant_id": tenant_id,
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "total_events": len(entries),
            "unique_users": len(users),
            "failed_actions": failures,
            "action_breakdown": action_counts,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def export_entries(
        self,
        entries: Optional[List[AuditEntry]] = None,
        format: str = "json",
    ) -> str:
        """
        Export audit entries.

        Args:
            entries: Entries to export (default: all)
            format: Export format (json, csv)

        Returns:
            Exported data string
        """
        entries = entries or self._entries

        if format == "json":
            return json.dumps([e.to_dict() for e in entries], indent=2)

        elif format == "csv":
            if not entries:
                return ""

            lines = ["timestamp,action,tenant_id,user_id,resource_type,resource_id,success"]
            for entry in entries:
                lines.append(
                    f"{entry.timestamp.isoformat()},{entry.action.value},"
                    f"{entry.tenant_id or ''},{entry.user_id or ''},"
                    f"{entry.resource_type or ''},{entry.resource_id or ''},"
                    f"{entry.success}"
                )
            return "\n".join(lines)

        raise ValueError(f"Unknown format: {format}")

    def clear(self) -> None:
        """Clear all audit entries."""
        self._entries.clear()


# Global audit logger
_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger."""
    global _logger
    if _logger is None:
        _logger = AuditLogger()
    return _logger


if __name__ == "__main__":
    print("=" * 60)
    print("AUDIT LOGGING TEST")
    print("=" * 60)

    logger = AuditLogger()

    # Log some events
    logger.log(
        action=AuditAction.DOCUMENT_UPLOADED,
        tenant_id="tenant-1",
        user_id="user-1",
        resource_type="document",
        resource_id="doc-123",
        details={"filename": "invoice.pdf", "size_kb": 512},
    )

    logger.log(
        action=AuditAction.DOCUMENT_PROCESSED,
        tenant_id="tenant-1",
        user_id="user-1",
        resource_type="document",
        resource_id="doc-123",
        details={"task": "invoice", "processing_ms": 1234},
    )

    # Query
    entries = logger.query(tenant_id="tenant-1")
    print(f"  Entries for tenant-1: {len(entries)}")

    # Generate report
    report = logger.get_compliance_report(
        tenant_id="tenant-1",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2025, 12, 31),
    )
    print(f"  Total events: {report['total_events']}")

    print("=" * 60)
