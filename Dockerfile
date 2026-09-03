FROM continuumio/miniconda3:latest

LABEL description="EV Charger Optimization — full pipeline (CG + queue simulation) on Linux."

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ git file && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

COPY . .

# Build the C++ shortest-path library for Linux
RUN ./build_liblsp.sh

# Install pytest
RUN conda run -n evopt pip install pytest

# Verify the library loads
RUN conda run -n evopt python -c "from queue_sim import Runner; print('Queue sim library OK')"

ENV CONDA_DEFAULT_ENV=evopt

ENTRYPOINT ["conda", "run", "-n", "evopt", "python", "pipeline.py"]
