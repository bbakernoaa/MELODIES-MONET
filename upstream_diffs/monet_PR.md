# PR: [monet] Enhanced Xarray Pairing and Interpolation

## Summary
Integrates improved pairing and interpolation logic from MELODIES-MONET into monet.
- Adds `_pair_xarray` and `_pair_vertical_xarray` to handle mobile and vertical pairing natively in Xarray.
- Enhances Dask support for out-of-core computation.
- Updates the `pair` accessor to use these new backends.

## Changes
- `monet/util/combinetool.py`: Implemented new pairing backends and integrated them into the `pair` accessor.
