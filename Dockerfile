FROM 127.0.0.1:5000/luxeillum/codex-executor-runtime:aa0d84548a5d03c66dd110be6455ada2ce837537

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Pin the Codex CLI to a release whose bundled model metadata and ChatGPT
# account support include the current Luna generation. The base image ships an
# older CLI that rejects GPT-6 Luna with "model is not supported when using
# Codex with a ChatGPT account". Model routing stays in this gateway; only the
# CLI version is pinned here.
ARG CODEX_CLI_VERSION=0.156.1
RUN npm install -g "@openai/codex@${CODEX_CLI_VERSION}" \
    && codex --version | grep -Fx "codex-cli ${CODEX_CLI_VERSION}"

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY runtime-entrypoint.sh /usr/local/bin/luna-runtime-entrypoint
RUN groupadd --system --gid 10001 luna-runtime \
    && useradd --system --uid 10001 --gid 10001 --home-dir /home/luna-runtime --create-home --shell /usr/sbin/nologin luna-runtime \
    && mkdir -p /home/luna-runtime /tmp/luna-codex-home \
    && chown -R 10001:10001 /home/luna-runtime /tmp/luna-codex-home \
    && chown 10001:0 /tmp/luna-codex-home \
    && chmod 0770 /tmp/luna-codex-home \
    && chmod 0555 /usr/local/bin/luna-runtime-entrypoint

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=8s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"
ENTRYPOINT ["tini", "--", "/usr/local/bin/luna-runtime-entrypoint"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
