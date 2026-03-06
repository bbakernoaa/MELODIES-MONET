# monet-stats PR: Implement Paired Data Contract

## Description
This PR refactors `monet-stats` to accept the MELODIES-MONET "Paired Data Contract" (standardized Xarray Dataset) as an input, removing internal data-loading logic and ensuring the library functions as a "pure" consumer.

## Changes
- Added `monet_stats.proc_stats` for legacy MM interface compatibility.
- Standardized all metrics to work directly with Xarray Datasets.
