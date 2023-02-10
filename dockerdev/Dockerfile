# BASE: image that other versions are built upon; only include things common to all versions
FROM continuumio/miniconda3

# Install necessary system packages and clean-up
RUN chmod 1777 /tmp \
  && apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends  \
      curl build-essential graphviz \
  && apt-get clean  \
  && rm -rf /var/tmp /tmp /var/lib/apt/lists/* \
  && mkdir -p /var/tmp /tmp

# Install poetry and tox
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN ln -s /root/.local/bin/poetry /usr/local/bin/poetry
RUN pip3 install tox

# used as a volume (connects to the host machine's disk at runtime)
RUN mkdir -p /data

WORKDIR /workdir
