"""
Security utilities for path sanitization, input validation, and safe file handling.

Use these helpers to prevent path traversal, injection, and unsafe deserialization.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Optional

# Maximum filename length; prevent excessively long paths
MAX_FILENAME_LEN = 255

# Safe character set for filenames (alphanumeric, dash, underscore, dot)
SAFE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]")

# Control characters and other unsafe characters in user text
CONTROL_AND_UNSAFE = re.compile(r"[\x00-\x1f\x7f-\x9f]")

# Allowed image extensions for uploads (lowercase)
ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})

# Content-Type to extension mapping for uploads
CONTENT_TYPE_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


def sanitize_filename(name: str) -> str:
    """
    Sanitize a filename to prevent path traversal and unsafe characters.

    - Removes path components (dirs).
    - Replaces unsafe characters with underscore.
    - Truncates to MAX_FILENAME_LEN.
    - Strips null bytes and leading/trailing dots/spaces.
    """
    if not name or not isinstance(name, str):
        return "unnamed"
    # Remove null bytes
    name = name.replace("\x00", "")
    # Keep only the base name
    name = os.path.basename(name)
    # Replace unsafe chars
    name = SAFE_FILENAME_PATTERN.sub("_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("._ ")
    if not name:
        return "unnamed"
    return name[:MAX_FILENAME_LEN]


def safe_join_path(base: Path, *parts: str) -> Path:
    """
    Join path parts to base and resolve, ensuring result is under base.

    Raises ValueError if the resolved path is outside base (path traversal).
    """
    base = Path(base).resolve()
    combined = base
    for part in parts:
        part = part.replace("\x00", "")
        combined = (combined / part).resolve()
    try:
        combined.relative_to(base)
    except ValueError:
        raise ValueError(f"Path would escape base directory: {combined}")
    return combined


def validate_user_text(text: str, max_length: int = 10000) -> str:
    """
    Validate and normalize user-provided text (prompts, queries, descriptions).

    - Strips leading/trailing whitespace.
    - Removes null bytes and control characters.
    - Raises ValueError if length exceeds max_length after strip.
    """
    if not isinstance(text, str):
        raise ValueError("Expected string input")
    text = text.replace("\x00", "").strip()
    text = CONTROL_AND_UNSAFE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        raise ValueError(f"Text exceeds maximum length of {max_length}")
    return text


def safe_upload_filename(content_type: Optional[str] = None, original_filename: Optional[str] = None) -> str:
    """
    Generate a safe storage filename for an upload (UUID + allowed extension).

    Prefers extension from content_type allowlist; falls back to sanitized original extension.
    """
    ext = ".bin"
    if content_type and content_type.split(";")[0].strip().lower() in CONTENT_TYPE_TO_EXT:
        ext = CONTENT_TYPE_TO_EXT[content_type.split(";")[0].strip().lower()]
    elif original_filename:
        base = sanitize_filename(original_filename)
        _, e = os.path.splitext(base)
        if e.lower() in ALLOWED_IMAGE_EXTENSIONS:
            ext = e.lower()
    return f"{uuid.uuid4().hex}{ext}"


def is_allowed_image_extension(ext: str) -> bool:
    """Return True if extension is in the allowed image list."""
    return ext.lower() in ALLOWED_IMAGE_EXTENSIONS
