FROM python:3.8.9

RUN pip install pip==22.3.1

WORKDIR /app
RUN mkdir -p /app/.cache
ENV HF_HOME="/app/.cache"
RUN chown -R 1000:1000 /app
USER 1000
ENV HOME=/app

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && sh Miniconda3-latest-Linux-x86_64.sh -b -p /app/miniconda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH /app/miniconda/bin:$PATH

RUN conda create -p /app/env -y python=3.8

SHELL ["conda", "run","--no-capture-output", "-p","/app/env", "/bin/bash", "-c"]

COPY --chown=1000:1000 requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY --chown=1000:1000 *.py /app/
COPY --chown=1000:1000 pages/ /app/pages/

CMD streamlit run Overview.py \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.fileWatcherType none