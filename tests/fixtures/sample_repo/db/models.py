"""ORM-style data models for the application database."""
from dataclasses import dataclass, field
from typing import List, Optional
import uuid


@dataclass
class User:
    """Represents a registered user account."""

    email: str
    hashed_password: str
    role: str = "viewer"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_active: bool = True

    def deactivate(self) -> None:
        """Mark this user as inactive."""
        self.is_active = False


@dataclass
class Session:
    """Tracks an authenticated user session."""

    user_id: str
    token: str
    expires_at: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def is_expired(self, now: float) -> bool:
        """Return True if the session has expired."""
        return now >= self.expires_at


@dataclass
class Post:
    """A content post authored by a user."""

    title: str
    body: str
    author_id: str
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_tag(self, tag: str) -> None:
        """Append a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)
