"""Authentication and authorization for document processing service."""

import hashlib
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class Role(str, Enum):
    """User roles."""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    READONLY = "readonly"


class Permission(str, Enum):
    """Available permissions."""
    # Document operations
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_EXPORT = "document:export"

    # Batch operations
    BATCH_CREATE = "batch:create"
    BATCH_READ = "batch:read"
    BATCH_CANCEL = "batch:cancel"

    # Admin operations
    USER_MANAGE = "user:manage"
    TENANT_MANAGE = "tenant:manage"
    CONFIG_MANAGE = "config:manage"
    AUDIT_READ = "audit:read"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.MANAGER: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.DOCUMENT_DELETE,
        Permission.DOCUMENT_EXPORT,
        Permission.BATCH_CREATE,
        Permission.BATCH_READ,
        Permission.BATCH_CANCEL,
        Permission.USER_MANAGE,
        Permission.AUDIT_READ,
    },
    Role.USER: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.DOCUMENT_EXPORT,
        Permission.BATCH_CREATE,
        Permission.BATCH_READ,
    },
    Role.READONLY: {
        Permission.DOCUMENT_READ,
        Permission.BATCH_READ,
    },
}


@dataclass
class APIKey:
    """API key for authentication."""
    key_id: str
    key_hash: str  # Hashed key value
    tenant_id: str
    user_id: Optional[str]
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    scopes: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60


@dataclass
class User:
    """User entity."""
    user_id: str
    tenant_id: str
    email: str
    password_hash: str
    role: Role = Role.USER
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    permissions: Set[Permission] = field(default_factory=set)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a permission."""
        # Check explicit permissions
        if permission in self.permissions:
            return True
        # Check role permissions
        return permission in ROLE_PERMISSIONS.get(self.role, set())


@dataclass
class RateLimitInfo:
    """Rate limit tracking info."""
    key: str
    window_start: datetime
    request_count: int


class AuthManager:
    """Manage authentication and authorization."""

    def __init__(self):
        """Initialize auth manager."""
        self._api_keys: Dict[str, APIKey] = {}
        self._users: Dict[str, User] = {}
        self._rate_limits: Dict[str, RateLimitInfo] = {}

    # API Key Management

    def create_api_key(
        self,
        tenant_id: str,
        name: str,
        user_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        scopes: Optional[List[str]] = None,
        rate_limit: int = 60,
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Args:
            tenant_id: Tenant ID
            name: Key name
            user_id: Associated user
            expires_in_days: Expiration in days
            scopes: Allowed scopes
            rate_limit: Requests per minute

        Returns:
            Tuple of (raw key, key object)
        """
        # Generate key
        raw_key = f"qwvl_{secrets.token_urlsafe(32)}"
        key_id = str(uuid.uuid4())
        key_hash = self._hash_key(raw_key)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            tenant_id=tenant_id,
            user_id=user_id,
            name=name,
            expires_at=expires_at,
            scopes=scopes or [],
            rate_limit_per_minute=rate_limit,
        )

        self._api_keys[key_id] = api_key

        return raw_key, api_key

    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key.

        Args:
            raw_key: Raw API key string

        Returns:
            APIKey if valid, None otherwise
        """
        key_hash = self._hash_key(raw_key)

        for api_key in self._api_keys.values():
            if api_key.key_hash == key_hash:
                # Check active
                if not api_key.is_active:
                    return None

                # Check expiration
                if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                    return None

                # Update last used
                api_key.last_used = datetime.utcnow()

                return api_key

        return None

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._api_keys:
            self._api_keys[key_id].is_active = False
            return True
        return False

    def list_api_keys(self, tenant_id: str) -> List[APIKey]:
        """List API keys for a tenant."""
        return [k for k in self._api_keys.values() if k.tenant_id == tenant_id]

    # User Management

    def create_user(
        self,
        tenant_id: str,
        email: str,
        password: str,
        role: Role = Role.USER,
    ) -> User:
        """
        Create a new user.

        Args:
            tenant_id: Tenant ID
            email: User email
            password: Plain text password
            role: User role

        Returns:
            Created user
        """
        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(password)

        user = User(
            user_id=user_id,
            tenant_id=tenant_id,
            email=email,
            password_hash=password_hash,
            role=role,
        )

        self._users[user_id] = user
        return user

    def authenticate_user(
        self,
        email: str,
        password: str,
    ) -> Optional[User]:
        """
        Authenticate user by email/password.

        Args:
            email: User email
            password: Plain text password

        Returns:
            User if authenticated, None otherwise
        """
        password_hash = self._hash_password(password)

        for user in self._users.values():
            if user.email == email and user.password_hash == password_hash:
                if not user.is_active:
                    return None
                return user

        return None

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    def check_permission(
        self,
        user_id: str,
        permission: Permission,
    ) -> bool:
        """
        Check if user has permission.

        Args:
            user_id: User ID
            permission: Required permission

        Returns:
            True if permitted
        """
        user = self.get_user(user_id)
        if not user:
            return False
        return user.has_permission(permission)

    # Rate Limiting

    def check_rate_limit(
        self,
        key: str,
        limit: int = 60,
        window_seconds: int = 60,
    ) -> Dict[str, Any]:
        """
        Check and update rate limit.

        Args:
            key: Rate limit key
            limit: Max requests per window
            window_seconds: Window duration

        Returns:
            Dict with allowed, remaining, reset_at
        """
        now = datetime.utcnow()

        if key not in self._rate_limits:
            self._rate_limits[key] = RateLimitInfo(
                key=key,
                window_start=now,
                request_count=0,
            )

        info = self._rate_limits[key]

        # Check if window expired
        window_end = info.window_start + timedelta(seconds=window_seconds)
        if now > window_end:
            info.window_start = now
            info.request_count = 0

        # Check limit
        if info.request_count >= limit:
            reset_at = info.window_start + timedelta(seconds=window_seconds)
            return {
                "allowed": False,
                "remaining": 0,
                "reset_at": reset_at.isoformat(),
            }

        # Increment
        info.request_count += 1

        return {
            "allowed": True,
            "remaining": limit - info.request_count,
            "reset_at": (info.window_start + timedelta(seconds=window_seconds)).isoformat(),
        }

    # Helper methods

    def _hash_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        # In production, use bcrypt or argon2
        return hashlib.sha256(password.encode()).hexdigest()


# Global auth manager
_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get or create global auth manager."""
    global _manager
    if _manager is None:
        _manager = AuthManager()
    return _manager


if __name__ == "__main__":
    print("=" * 60)
    print("AUTHENTICATION TEST")
    print("=" * 60)

    manager = AuthManager()

    # Create user
    user = manager.create_user(
        tenant_id="tenant-1",
        email="admin@example.com",
        password="secret123",
        role=Role.ADMIN,
    )
    print(f"  Created user: {user.email} ({user.role.value})")

    # Create API key
    raw_key, api_key = manager.create_api_key(
        tenant_id="tenant-1",
        name="Production Key",
        user_id=user.user_id,
        expires_in_days=30,
    )
    print(f"  API Key: {raw_key[:20]}...")

    # Validate
    validated = manager.validate_api_key(raw_key)
    print(f"  Key valid: {validated is not None}")

    # Check permission
    has_perm = manager.check_permission(user.user_id, Permission.TENANT_MANAGE)
    print(f"  Has TENANT_MANAGE: {has_perm}")

    # Rate limit
    for i in range(3):
        result = manager.check_rate_limit("test-key", limit=2)
        print(f"  Request {i+1}: allowed={result['allowed']}, remaining={result['remaining']}")

    print("=" * 60)
