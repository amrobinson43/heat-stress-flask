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
# MALLOC_ARENA_MAX caps glibc's per-thread heap arenas, which otherwise inflate
# RSS a lot with numpy/GDAL + threads -> important to fit the 512 MB free tier.
ENV PORT=10000 PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1
EXPOSE 10000

# 1 worker so the single background build thread + in-memory state are shared.
# --timeout 0 disables the worker timeout: the forecast build runs in a daemon
# thread inside this worker and can take many minutes on a small/free instance,
# and we must NOT let gunicorn kill+restart the worker (which would abort the
# build and reset state). Requests themselves are fast (the build is off-thread).
CMD gunicorn app:app --workers 1 --threads 2 --timeout 0 --bind 0.0.0.0:${PORT}
