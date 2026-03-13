# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libnetcdf-dev \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install core repositories
RUN pip install --no-cache-dir git+https://github.com/bbakernoaa/monetio.git@develop
RUN pip install --no-cache-dir git+https://github.com/bbakernoaa/monet.git@feature/interp_improvements
RUN pip install --no-cache-dir git+https://github.com/bbakernoaa/monet-stats.git@dev
RUN pip install --no-cache-dir git+https://github.com/bbakernoaa/monet-plots.git@main

# Copy the MELODIES-MONET source code
COPY . .

# Install MELODIES-MONET in editable mode
RUN pip install --no-cache-dir -e .

# Install orchestration and testing dependencies
RUN pip install --no-cache-dir \
    prefect \
    prefect-dask \
    dask-jobqueue \
    dask-cloudprovider \
    networkx \
    pytest \
    pytest-mpl \
    pooch

# Set the default command
CMD ["pytest", "melodies_monet/tests/test_regression.py"]
