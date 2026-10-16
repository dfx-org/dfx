# dfx

dfx is a tensor library for F# with support for differentiable programming. It is designed for use in machine learning, probabilistic programming, optimization and other domains.

- Tensors with forward, reverse and nested (higher-order) automatic differentiation
- Models, optimizers and differentiable probability distributions
- A Torch backend using LibTorch, the C++/CUDA tensor library that powers PyTorch, on CPUs and NVIDIA GPUs
- A Reference backend written in F#, which runs anywhere .NET runs

## Which package to use

| Package | Contents |
|---|---|
| `dfx-cpu` | dfx with the LibTorch CPU binaries for Windows (x64), Linux (x64) and macOS (Apple Silicon) |
| `dfx-cuda-windows` | dfx with the LibTorch CUDA 12.8 binaries for Windows (x64). Large download |
| `dfx-cuda-linux` | dfx with the LibTorch CUDA 12.8 binaries for Linux (x64). Large download |
| `dfx-cuda` | Both `dfx-cuda-windows` and `dfx-cuda-linux` |
| `dfx-lite` | dfx without LibTorch binaries, for use with a LibTorch installed by other means, for example with PyTorch |

The `dfx-Dfx.*` packages contain the individual libraries that these packages reference.

## Example

```fsharp
open Dfx

// The gradient of sum(x^2) is 2x
let f (x: Tensor) = (x * x).sum()
let g = dfx.grad f (dfx.tensor [1.0; 2.0; 3.0])   // tensor([2., 4., 6.])
```

## Requirements

- .NET 10 or later
- CUDA packages: an NVIDIA GPU with a recent driver. The CUDA runtime is included
- macOS: the LibTorch build for Apple Silicon needs OpenMP, which you can install with `brew install libomp`

## Links

- Documentation: https://docs.dfx.ac/
- Source code: https://github.com/dfx-org/dfx
- Issues: https://github.com/dfx-org/dfx/issues

## Credits

dfx is the continuation of [DiffSharp](https://github.com/DiffSharp/DiffSharp). Its lead creator is Atılım Güneş Baydin, and it was developed with Don Syme, Barak A. Pearlmutter, Jeffrey Mark Siskind and many contributors.

If you use dfx in your work, please cite: Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul and Jeffrey Mark Siskind. Automatic Differentiation in Machine Learning: a Survey. Journal of Machine Learning Research, 18(153):1–43, 2018.

dfx is licensed under the BSD 2-Clause License.
