# monet-plots PR: Implement Paired Data Contract

## Description
This PR refactors `monet-plots` to be a "pure" consumer of aligned data following the Paired Data Contract. It removes redundant internal loading logic and standardizes on Xarray-based visualization.

## Changes
- Refactored all plotting routines to accept `xarray.Dataset` directly.
- Added `SpatialBiasPlot` and ensured Cartopy projection handling is convention-aware.
