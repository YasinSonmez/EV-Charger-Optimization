FROM continuumio/miniconda3:latest

LABEL description="EV Charger Optimization — full pipeline (CG + queue simulation) on Linux." \
      org.opencontainers.image.source="https://github.com/YasinSonmez/EV-Charger-Optimization" \
      evopt.libsp.commit="475298f4570109378a57b4e592f01b8a26fe0c90"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ git file && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

COPY . .

# Build the C++ shortest-path library for Linux
RUN ./build_liblsp.sh

# Verify the library loads
RUN conda run -n evopt python -c "from queue_sim import Runner; print('Queue sim library OK')"
RUN conda run -n evopt python run_suite.py --manifest configs/rebuttal/suite.json --validate-only

ENV CONDA_DEFAULT_ENV=evopt
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib

ENTRYPOINT ["conda", "run", "-n", "evopt", "python", "pipeline.py"]
