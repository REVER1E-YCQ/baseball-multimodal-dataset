# Third-party components

This baseline does not vendor third-party source code or model weights.

## M2D

- Upstream repository: <https://github.com/nttcslab/m2d>
- Tested commit: `3d0c4de9447c404a8d3f9f37e04f53bc902e09b3`
- Model: `m2d_vit_base-80x200p16x4-230529`
- Loader used at runtime: `examples/portable_m2d.py`
- Upstream license: follow the `LICENSE.pdf` distributed by the M2D project.

The checkpoint is approximately 1.68 GB and is intentionally excluded from this
package. Download it from the release link published by the M2D authors and
verify the SHA256 recorded in `protocol.json`.

PyTorch checkpoints may contain Python pickle data. Only load a checkpoint from
the trusted upstream release.

## Project license

This candidate package does not add or choose a license for the parent dataset
repository. The repository owner must decide and add the project license before
publishing the code as an open-source deliverable.

