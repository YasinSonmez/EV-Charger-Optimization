FROM continuumio/miniconda3@sha256:eca594d684f495c1a02beff33a9fab53aec8c5830eaf431bb149912dc6c9e4c1 AS libsp-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ git file && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY build_liblsp.sh .
RUN ./build_liblsp.sh

FROM continuumio/miniconda3@sha256:eca594d684f495c1a02beff33a9fab53aec8c5830eaf431bb149912dc6c9e4c1

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

ARG VCS_REF=unknown

LABEL description="EV Charger Optimization — full pipeline (CG + queue simulation) on Linux." \
      org.opencontainers.image.source="https://github.com/YasinSonmez/EV-Charger-Optimization" \
      org.opencontainers.image.revision="${VCS_REF}" \
      evopt.libsp.commit="475298f4570109378a57b4e592f01b8a26fe0c90"

COPY . .

# Keep the native library outside /app so mounting a Git checkout never hides
# the Linux binary supplied by the image.
COPY --from=libsp-builder /build/dlls/liblsp.so /opt/evopt/lib/liblsp.so

ENV CONDA_DEFAULT_ENV=evopt \
    EVOPT_CONTAINERIZED=1 \
    EVOPT_EXECUTION_MODE=image \
    EVOPT_CODE_COMMIT=${VCS_REF} \
    EVOPT_LIBLSP_PATH=/opt/evopt/lib/liblsp.so \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# Verify the library loads
RUN conda run -n evopt python -c "from queue_sim import Runner; print('Queue sim library OK')"
RUN conda run -n evopt python run_suite.py --manifest configs/rebuttal/suite.json --validate-only

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "evopt", "python", "pipeline.py"]
