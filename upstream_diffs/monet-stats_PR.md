# PR: [monet-stats] Pure Xarray Metrics

## Summary
Refactors monet-stats to accept the Paired Data Contract (Xarray Dataset).
- Removed internal data-loading logic.
- Implemented a `calculate_stats` wrapper for legacy support during the transition.

## Changes
- `src/monet_stats/proc_stats.py`: Refactored to consume Xarray objects directly.
