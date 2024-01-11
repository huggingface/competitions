FROM huggingface/autotrain-advanced:latest

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

WORKDIR /app
RUN mkdir -p /app/.cache
ENV HF_HOME="/app/.cache"
RUN chown -R 1000:1000 /app
USER 1000
ENV HOME=/app

SHELL ["conda", "run","--no-capture-output", "-p","/app/env", "/bin/bash", "-c"]

COPY --chown=1000:1000 . /app/
RUN make socket-kit.so

ENV PATH="/app:${PATH}"

RUN pip install -e .