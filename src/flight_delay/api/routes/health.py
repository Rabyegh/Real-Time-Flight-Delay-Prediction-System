"""Process health route."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Report that the API process is responsive."""
    return {"status": "ok"}
