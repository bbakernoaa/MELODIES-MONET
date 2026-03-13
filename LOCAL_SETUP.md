# Local Setup Instructions for MELODIES-MONET Validation

To run the validation and deployment environment locally, follow these steps:

## Prerequisites
- Python 3.9+
- git
- (Optional) Docker

## Option 1: Using Docker (Recommended)
1. Build the Docker image:
   ```bash
   docker build -t melodies-monet-regression .
   ```
2. Run the regression tests:
   ```bash
   docker run --rm melodies-monet-regression
   ```

## Option 2: Manual Installation
1. Create a fresh virtual environment:
   ```bash
   python -m venv mm_env
   source mm_env/bin/activate  # On Windows: mm_env\Scripts\activate
   ```
2. Install system dependencies (e.g., `libnetcdf-dev` on Ubuntu).
3. Install the 5 core repositories:
   ```bash
   pip install git+https://github.com/NCAR/monetio.git@develop
   pip install git+https://github.com/NCAR/monet.git@master
   pip install git+https://github.com/NCAR/monet-stats.git@main || pip install git+https://github.com/NCAR/monet-stats.git@master
   pip install git+https://github.com/NCAR/monet-plots.git@main || pip install git+https://github.com/NCAR/monet-plots.git@master
   ```
4. Install MELODIES-MONET and testing dependencies:
   ```bash
   pip install -e .
   pip install prefect prefect-dask dask-jobqueue dask-cloudprovider networkx pooch pytest pytest-mpl
   ```
5. Run the regression tests:
   ```bash
   python -m pytest melodies_monet/tests/test_regression.py
   ```

## Regression Test Suite
The regression test suite (`melodies_monet/tests/test_regression.py`) verifies:
- Bit-for-bit identity between sequential and Prefect-based execution using mock data.
- Placeholder for plotting regression using `pytest-mpl`.
