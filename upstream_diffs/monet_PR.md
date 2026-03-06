# monet PR: Enhance Dask Support and Leverage monet.pair

## Description
This PR refactors interpolation and pairing logic to leverage the new `monet.pair` accessor. It enhances out-of-core performance using Dask and integrates MM-specific vertical and mobile pairing requirements into the core library.

## Changes
- Refactored `mobile_and_ground_pair` to delegate to `monet.pair`.
- Integrated optimized `vert_interp` (using `resample_stratify`) into `monet.util.combinetool`.
- Improved auto-detection of pairing modes (vertical vs spatial) within `monet.pair`.
