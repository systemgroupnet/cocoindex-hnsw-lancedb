# Builds cocoindex-code from the LOCAL source tree (the build context) rather
# than from PyPI. Use this when you have local changes that aren't released yet
# (e.g. a feature branch). For the PyPI-based release image, see Dockerfile.external.
#
# Layer ordering still favours cache reuse: the heavy, slow-changing layers
# (gosu/uv/sentence-transformers/baked model) come first and are digest-stable
# across builds, so only the final source-install layer invalidates when the
# source tree changes.
#
# `cocoindex` itself (the upstream dependency) still resolves from PyPI — it
# ships pre-built Rust wheels. What's built locally here is THIS repo's
# `cocoindex-code` package, via a PEP 517 build from the bind-mounted tree.
#
# Two variants, matching Dockerfile.external:
#   - slim (default) — ~450 MB. cocoindex-code's LiteLLM deps install in the
#     final layer.
#   - full           — ~5 GB. Bundles torch + sentence-transformers + a baked
#     default embedding model for offline-ready local embeddings.
#
# Use slim (glibc-based) — cocoindex ships pre-built Rust wheels that need glibc.
# Alpine / musl-libc would require building from source.
#
# `--system` tells uv to install into the base Python at
# /usr/local/lib/python3.12/... since there's no virtualenv in the image.

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 coco \
    && useradd -u 1000 -g 1000 -m coco

RUN --mount=type=cache,target=/root/.cache/pip pip install --quiet uv

# Heavy, stable deps for full variant. Layer digest is reproducible across
# builds (RUN command string is constant), so this layer stays cached.
ARG CCC_VARIANT=slim
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$CCC_VARIANT" = "full" ]; then \
        uv pip install --system sentence-transformers; \
    fi

ENV HF_HOME=/var/cocoindex/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/var/cocoindex/cache/sentence-transformers

# Pre-bake the default embedding model (full only). For slim, just create
# empty cache dirs so the cocoindex-data named volume mounts cleanly.
RUN mkdir -p /var/cocoindex/cache/huggingface /var/cocoindex/cache/sentence-transformers \
    && if [ "$CCC_VARIANT" = "full" ]; then \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Snowflake/snowflake-arctic-embed-xs'); print('Model cached.')"; \
    fi

# Writable paths the daemon needs, pre-chowned to coco. Under PUID/PGID the
# entrypoint re-chowns to the host user; under root (Docker Desktop
# default) coco-ownership is harmless since processes run as root and can
# write anywhere.
RUN mkdir -p /var/cocoindex/db /var/run/cocoindex_code \
    && chown -R coco:coco /var/cocoindex /var/run/cocoindex_code

WORKDIR /workspace

# Runtime defaults — see the spec for what each does. All overridable at
# `docker run -e ...` time.
#
# The daemon serves a streamable-HTTP MCP server in-process when
# COCOINDEX_CODE_MCP_PORT is set (bound to 0.0.0.0 so it's reachable from
# outside the container). No external proxy is involved — the MCP `search` tool
# queries the daemon's project registry directly. Endpoint: http://<host>:8001/mcp
# Set COCOINDEX_CODE_MCP_DISABLE=1 at `docker run` time to turn it off without
# unsetting the port baked in here.
#
# Behind a reverse proxy (the Host header is the public hostname, not
# localhost), the MCP SDK's DNS-rebinding protection returns `421 Invalid Host
# header`. Allow the proxied host with COCOINDEX_CODE_MCP_ALLOWED_HOSTS, e.g.
#   -e COCOINDEX_CODE_MCP_ALLOWED_HOSTS=code.example.com
# (comma-separated; `host:*` matches any port). Use the literal `*` to disable
# the Host check entirely when a trusted proxy already controls access.
# COCOINDEX_CODE_MCP_ALLOWED_ORIGINS does the same for the Origin header
# (only needed for browser-based clients).
ENV COCOINDEX_CODE_DIR=/workspace/.cocoindex_code \
    COCOINDEX_CODE_RUNTIME_DIR=/var/run/cocoindex_code \
    COCOINDEX_CODE_DB_PATH_MAPPING=/workspace=/var/cocoindex/db \
    COCOINDEX_CODE_DAEMON_SUPERVISED=1 \
    COCOINDEX_CODE_MCP_HOST=0.0.0.0 \
    COCOINDEX_CODE_MCP_PORT=8001

EXPOSE 8001

COPY docker/entrypoint.sh /entrypoint.sh
# Strip any CR before chmod: a Windows build context (core.autocrlf=true) can
# COPY the script in with CRLF, and a `#!/bin/sh\r` shebang fails to exec.
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# ─── Per-release layer (last so only this one invalidates per build) ───────
#
# Install cocoindex-code from the bind-mounted local source tree. The extras
# match CCC_VARIANT (slim → no extras, full → [full]). `rw=true` on the bind
# mount gives hatch-vcs a writable overlay for `_version.py` during the
# PEP 517 build; the overlay is discarded after the RUN, so the source tree
# doesn't persist as a layer in the final image.
RUN --mount=type=bind,source=.,target=/ccc-src,rw=true \
    --mount=type=cache,target=/root/.cache/uv \
    if [ "$CCC_VARIANT" = "full" ]; then \
        uv pip install --system "/ccc-src[full]"; \
    else \
        uv pip install --system "/ccc-src"; \
    fi