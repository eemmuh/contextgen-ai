# contextgen-ai Documentation

This `docs/` folder contains the project’s core documentation. Links below only point to files that exist in this repository.

## Contents

- **[Quick Start](quickstart.md)**: Minimal setup + first run
- **[Installation](installation.md)**: Full setup guide
- **[Database Integration](database/README.md)**: pgvector schema, indexing, and operations
- **[Database Setup (root)](../DATABASE_SETUP.md)**: Docker + local Postgres setup notes

## API

FastAPI serves interactive documentation at:
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI**: `/openapi.json`

### Common endpoints

- **Core**
  - `GET /` – service info
  - `GET /health` – basic health check
- **API v1** (prefixed with `/api/v1`)
  - `GET /api/v1/health`
  - `POST /api/v1/search`
  - `POST /api/v1/generate`
  - `POST /api/v1/rag/generate`
  - `GET /api/v1/rag/search`
  - `POST /api/v1/upload`
  - `GET /api/v1/images`
  - `GET /api/v1/images/{image_id}`
  - `DELETE /api/v1/images/{image_id}`
  - `GET /api/v1/stats`

## Configuration

- **Example env file**: start from `../env.example` and copy to `.env`
- **Settings code**: `config/settings.py`

## Support

- **Issues**: [GitHub Issues](https://github.com/eemmuh/contextgen-ai/issues)

## License

MIT — see `../LICENSE`.

---

**Last updated**: 2026-01-24