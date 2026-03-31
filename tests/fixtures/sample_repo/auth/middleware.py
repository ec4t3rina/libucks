"""JWT authentication middleware."""
import hashlib
import hmac


def validate_token(token: str, secret: str) -> bool:
    """Return True if the JWT token signature is valid."""
    if not token or not secret:
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    header_payload = f"{parts[0]}.{parts[1]}"
    expected = hmac.new(secret.encode(), header_payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, parts[2])


def require_role(role: str):
    """Decorator that gates a route to users with the given role."""
    def decorator(fn):
        def wrapper(request, *args, **kwargs):
            user_role = getattr(request, "role", None)
            if user_role != role:
                raise PermissionError(f"Requires role '{role}', got '{user_role}'")
            return fn(request, *args, **kwargs)
        return wrapper
    return decorator


def extract_bearer(header: str) -> str:
    """Extract the token string from an Authorization: Bearer <token> header."""
    if not header or not header.startswith("Bearer "):
        raise ValueError("Missing or malformed Authorization header")
    return header[len("Bearer "):]
