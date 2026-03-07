# PR: [monet-plots] Pure Xarray Visualization

## Summary
Refactors monet-plots to consume Xarray Datasets instead of internal data loading.
- Implements `SpatialBiasPlot` as a starting point for modular visualization.

## Changes
- `src/monet_plots/plots/spatial_bias.py`: Implemented new plot class for spatial bias.
