# Torchbp

![SAR image](https://github.com/Ttl/torchbp/blob/master/docs/img/07_19_1_autofocus_sigma0_pol_cal_pauli.png?raw=true)

Differentiable synthetic aperture radar image formation and autofocus library
written in pure PyTorch.

All operators are implemented with standard PyTorch tensor operations, there
is no compiled extension. The library runs on any device supported by
PyTorch (CPU, CUDA, MPS, ...) and every operation is available on every
device.

Geometry and phase computations use float64 by default for accuracy. On GPUs
with slow double precision they can be switched to float32 with
`torchbp.set_geometry_dtype(torch.float32)`. The peak memory of the
vectorized kernels is bounded by chunking, tune it with
`torchbp.set_chunk_elements(n)`.

## Installation

The only dependencies are PyTorch, NumPy and SciPy.

```bash
pip install torch
git clone https://github.com/Ttl/torchbp.git
cd torchbp
pip install -e .
```

## Tests

Install test dependencies, then run pytest:

```bash
pip install -e .[test]
pytest tests/
```

On a CPU-only machine the CUDA tests are skipped rather than failing.

## Documentation

Latest documentation can be viewed at: https://ttl.github.io/torchbp/

API documentation and examples can be built with sphinx.

```bash
pip install .[docs]
cd docs
make html
```

Open `docs/build/html/index.html`.
