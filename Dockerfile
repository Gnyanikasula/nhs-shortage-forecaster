# NOTE: running Spark inside a LINUX container means the entire
# winutils.exe / hadoop.dll / HADOOP_HOME saga from local Windows
# development simply doesn't apply here -- Spark's local filesystem
# writes work natively on Linux with no native Windows shim needed.
# One of the real, concrete benefits of containerizing this pipeline,
# not just a portability nicety.
#
# Pinned to slim-bookworm explicitly (not just "slim") -- python:3.12-slim
# floats to whatever Debian release is currently tagged "stable", which
# just moved to trixie (Debian 13) and doesn't carry openjdk-17-jdk-headless
# under that package name. bookworm is the well-documented, known-good base.

FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/