(*** condition: prepare ***)
#I "../tests/Dfx.Tests/bin/Debug/net10.0"
#r "Dfx.Core.dll"
#r "Dfx.Data.dll"
#r "Dfx.Backends.Reference.dll"
#r "Dfx.Backends.Torch.dll"
// These are needed to make fsdocs --eval work. If we don't select a backend like this in the beginning, we get erratic behavior.
Dfx.dfx.config(backend=Dfx.Backend.Reference)
Dfx.dfx.seed(123)

(*** condition: fsx ***)
#if FSX
#r "nuget: dfx-lite,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
// Google Colab only: uncomment and run the following to install dotnet and the F# kernel
#endif // IPYNB
(*** condition: ipynb ***)
#if IPYNB
// Import Dfx package
#r "nuget: dfx-lite,{{fsdocs-package-version}}"

// Set dotnet interactive formatter to plaintext
Formatter.SetPreferredMimeTypesFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dfx-org/dfx/blob/gh-pages/{{fsdocs-source-basename}}.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/dfx-org/dfx/gh-pages?filepath={{fsdocs-source-basename}}.ipynb)&emsp;
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)&emsp;
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb)

# Dfx: Differentiable Tensor Programming Made Simple

Dfx is a tensor library with support for [differentiable programming](https://en.wikipedia.org/wiki/Differentiable_programming).
It is designed for use in machine learning, probabilistic programming, optimization and other domains.

## Continuing DiffSharp

Dfx is the continuation of [DiffSharp](https://github.com/DiffSharp/DiffSharp), carrying its work forward with deep respect for the ideas, care, and contributions that established its foundation.

We offer our sincere thanks to [Dr. Atılım Güneş Baydin](https://www.cs.ox.ac.uk/people/atilim.gunes.baydin/) for creating and leading DiffSharp and for his foundational contributions to differentiable programming in F#.

<button class="button" style="vertical-align:middle" onclick="window.location.href='{{root}}install.html'"><span>Install »</span></button>

## Key Features

🗹 Nested and mixed-mode differentiation

🗹 Common optimizers, model elements, differentiable probability distributions

🗹 F# for robust functional programming

🗹 PyTorch familiar naming and idioms, efficient LibTorch CUDA/C++ tensors with GPU support

🗹 Linux, macOS, Windows supported

🗹 Use interactive notebooks in Jupyter and Visual Studio Code

🗹 100% open source


## Differentiable Programming

Dfx provides world-leading automatic differentiation capabilities for tensor code, including composable gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector products over arbitrary user code. This goes beyond conventional tensor libraries such as PyTorch and TensorFlow, allowing the use of nested forward and reverse differentiation up to any level.

With Dfx, you can compute higher-order derivatives efficiently and differentiate functions that are internally making use of differentiation and gradient-based optimization.

</br>
<img src="img/anim-intro-2.gif" width="85%" />

## Practical, Familiar and Efficient

Dfx comes with a [LibTorch](https://pytorch.org/cppdocs/) backend, using the same C++ and CUDA implementations for tensor computations that power [PyTorch](https://pytorch.org/). On top of these raw tensors (LibTorch's ATen, excluding autograd), Dfx implements its own computation graph and differentiation capabilities. It is tested on Linux, macOS, and Windows, and it supports CUDA and GPUs.

The Dfx API is designed to be similar to [the PyTorch Python API](https://pytorch.org/docs/stable/index.html) through very similar naming and idioms, and where elements have similar names the PyTorch documentation can generally be used as a guide.

Dfx uses [the incredible F# programming language](https://dot.net/fsharp) for tensor programming. F# code is generally faster and more robust than equivalent Python code, while still being succinct and compact like Python, making it an ideal modern AI and machine learning implementation language. This allows fluent and productive code for tensor programming.

</br>
<iframe width="85%" src="https://www.youtube.com/embed/_QnbV6CAWXc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Interactive Notebooks

All documentation pages in this website are interactive notebooks which you can execute directly in your browser without installing anything in your local machine.

Using the buttons [![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dfx-org/dfx/blob/gh-pages/{{fsdocs-source-basename}}.ipynb) [![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/dfx-org/dfx/gh-pages?filepath={{fsdocs-source-basename}}.ipynb) on the top of each page, you can execute the page as an interactive notebook running on cloud servers provided by [Google Colab](https://colab.research.google.com/) and [Binder](https://mybinder.org/).

Using the buttons [![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb) you can also download a page as a script or an interactive notebook, which you can execute locally in [Jupyter](https://jupyter.org/) or [Visual Studio Code](https://code.visualstudio.com/) using [dotnet interactive](https://github.com/dotnet/interactive).

## Example

Define and add two tensors:
*)

open Dfx

let t1 = dfx.tensor [ 0.0 ..0.2.. 1.0 ] // Gives [0., 0.2, 0.4, 0.6, 0.8, 1.]
let t2 = dfx.tensor [ 1, 2, 3, 4, 5, 6 ]

t1 + t2
(*** include-it ***)

(**
Compute a convolution:
*)
let t3 = dfx.tensor [[[[0.0 .. 10.0]]]]
let t4 = dfx.tensor [[[[0.0 ..0.1.. 1.0]]]]

t3.conv2d(t4)
(*** include-it ***)

(**
Take the gradient of a vector-to-scalar function:
*)

let f (x: Tensor) = x.exp().sum()

dfx.grad f (dfx.tensor([1.8, 2.5]))
(*** include-it ***)

(**
Compute a nested derivative (checking for [perturbation confusion](https://doi.org/10.1007/s10990-008-9037-1)):
*)

let x0 = dfx.tensor(1.)
let y0 = dfx.tensor(2.)
dfx.diff (fun x -> x * dfx.diff (fun y -> x * y) y0) x0
(*** include-it ***)


(**
Define a model and optimize it:
*)
(*** do-not-eval-file ***)
open Dfx
open Dfx.Data
open Dfx.Model
open Dfx.Compose
open Dfx.Util
open Dfx.Optim

let epochs = 2
let batchSize = 32
let numBatches = 5

let trainSet = MNIST("../data", train=true, transform=id)
let trainLoader = trainSet.loader(batchSize=batchSize, shuffle=true)

let validSet = MNIST("../data", train=false, transform=id)
let validLoader = validSet.loader(batchSize=batchSize, shuffle=false)

let encoder =
    Conv2d(1, 32, 4, 2)
    --> dfx.relu
    --> Conv2d(32, 64, 4, 2)
    --> dfx.relu
    --> Conv2d(64, 128, 4, 2)
    --> dfx.flatten(1)

let decoder =
    dfx.unflatten(1, [128;1;1])
    --> ConvTranspose2d(128, 64, 4, 2)
    --> dfx.relu
    --> ConvTranspose2d(64, 32, 4, 3)
    --> dfx.relu
    --> ConvTranspose2d(32, 1, 4, 2)
    --> dfx.sigmoid

let model = VAE([1;28;28], 64, encoder, decoder)

let lr = dfx.tensor(0.001)
let optimizer = Adam(model, lr=lr)

for epoch = 1 to epochs do
    let batches = trainLoader.epoch(numBatches)
    for i, x, _ in batches do
        model.reverseDiff()
        let l = model.loss(x)
        l.reverse()
        optimizer.step()
        print $"Epoch: {epoch} minibatch: {i} loss: {l}"

let validLoss =
    validLoader.epoch()
    |> Seq.sumBy (fun (_, x, _) -> model.loss(x, normalize=false))

print $"Validation loss: {validLoss/validSet.length}"

(**

Numerous other model definition, differentiation, and training patterns are supported. See the tutorials in the left-hand menu and [examples](https://github.com/dfx-org/dfx/tree/main/examples) on GitHub.

## More Information

Dfx carries forward [DiffSharp](https://github.com/DiffSharp/DiffSharp), which was developed by [Dr. Atılım Güneş Baydin](https://www.cs.ox.ac.uk/people/atilim.gunes.baydin/), [Don Syme](https://www.microsoft.com/en-us/research/people/dsyme/)
and other contributors, having started as a project supervised by the automatic differentiation wizards [Barak Pearlmutter](https://scholar.google.com/citations?user=AxFrw0sAAAAJ&hl=en) and [Jeffrey Siskind](https://scholar.google.com/citations?user=CgSBtPYAAAAJ&hl=en).

Please join us [on GitHub](https://github.com/dfx-org/dfx)!

*)
