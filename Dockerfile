# Docker image for the NBM WBGT forecast app.
# Uses micromamba so the ecCodes/GDAL/GEOS/PROJ native stack comes from
# conda-forge (pip cannot reliably install ecCodes for GRIB2 decoding).
FROM mambaorg/micromamba:1.5.8

# Install the conda environment first (cached layer).
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean --all --yes

# Activate the base env for the build and for CMD.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Render injects $PORT (defaults to 10000 here for local `docker run`).
ENV PORT=10000 PYTHONUNBUFFERED=1
EXPOSE 10000

# 1 worker so the single background build thread + in-memory state are shared.
CMD gunicorn app:app --workers 1 --threads 8 --timeout 180 --bind 0.0.0.0:${PORT}
