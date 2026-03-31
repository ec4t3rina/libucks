"""HTTP route handlers for the REST API."""
from typing import Any, Dict


def handle_login(request: Dict[str, Any]) -> Dict[str, Any]:
    """Authenticate a user and return a session token.

    Expects ``request`` to contain ``email`` and ``password`` keys.
    Returns ``{"token": <str>, "expires_at": <float>}``.
    """
    email = request.get("email", "")
    password = request.get("password", "")
    if not email or not password:
        return {"error": "email and password are required", "status": 400}
    # Stub: real implementation would look up the DB.
    return {"token": "stub-token", "expires_at": 9999999999.0, "status": 200}


def handle_get_post(request: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a single post by ID.

    Expects ``request["post_id"]``.  Returns the post dict or a 404.
    """
    post_id = request.get("post_id")
    if not post_id:
        return {"error": "post_id is required", "status": 400}
    # Stub: real implementation would query the DB.
    return {"id": post_id, "title": "Stub Post", "body": "...", "status": 200}


def handle_create_post(request: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new post.

    Expects ``title``, ``body``, and ``author_id`` in ``request``.
    """
    for field in ("title", "body", "author_id"):
        if not request.get(field):
            return {"error": f"{field} is required", "status": 400}
    return {"id": "new-post-id", "status": 201}


def handle_delete_post(request: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a post by ID (admin only)."""
    post_id = request.get("post_id")
    if not post_id:
        return {"error": "post_id is required", "status": 400}
    return {"deleted": post_id, "status": 200}
