# Security

This document summarizes security measures and practices used in this project.

## Secure Coding Practices

### Input validation
- **API requests**: Search query, prompts, and other text inputs are validated and sanitized via Pydantic schemas (`src/api/schemas.py`). Control characters and null bytes are stripped; length limits apply (e.g. query 500 chars, prompt 1000 chars).
- **User text**: `src/utils/security.py` provides `validate_user_text()` for normalizing and length-checking user-provided strings.

### Path and file safety
- **Path traversal**: User-controlled paths are not trusted. `safe_join_path()` ensures resolved paths stay under a base directory. `sanitize_filename()` removes path components and unsafe characters.
- **Uploads**: File uploads (`POST /api/v1/upload`) are saved under a fixed `uploads/` directory with UUID-based filenames; the client-provided filename is never used for storage and is only stored sanitized in metadata.
- **Index paths**: FAISS index save/load rejects paths containing `..` or null bytes; metadata is stored as JSON (not pickle).

### Serialization
- **No unsafe deserialization**: Redis cache uses JSON only; pickle is not used for cache values so untrusted data in Redis cannot trigger code execution.
- **Embedding metadata**: FAISS index metadata is persisted as JSON instead of pickle to avoid deserialization risks from tampered files.

### API and error handling
- **Rate limiting**: Applied per client (e.g. 60 requests/minute) to reduce abuse.
- **Error responses**: In production (`DEBUG=false`), 500 responses return a generic message; exception details are not exposed to clients. Full details are logged server-side only.
- **Secrets**: Database URL and API keys use `SecretStr` / env vars; avoid logging connection strings or secrets.

### Database
- **Queries**: Parameterized queries (e.g. SQLAlchemy `text()` with `:param`) are used; user input is not interpolated into SQL.

## Configuration

- Set `ENVIRONMENT=production` and `DEBUG=false` in production.
- Use strong, unique values for `DB_URL` and any API keys; keep `.env` out of version control (see `.gitignore`).
- Restrict `CORS_ORIGINS` to trusted origins in production instead of `["*"]`.

## Reporting vulnerabilities

If you discover a security issue, please report it responsibly (e.g. via a private channel or issue as appropriate to the project’s policy).
