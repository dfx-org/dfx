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
// Import dfx package
#r "nuget: dfx-lite,{{fsdocs-package-version}}"

// Set dotnet interactive formatter to plaintext
Formatter.SetPreferredMimeTypesFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)&emsp;
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb)

* The `cref:T:Dfx.dfx` API

* The `cref:T:Dfx.Tensor` type

Saving tensors as image and loading images as tensors


## Converting between Tensors and arrays

System.Array and F# arrays

*)

open Dfx

// Tensor
let t1 = dfx.tensor [ 0.0 .. 0.2 .. 1.0 ]

// System.Array
let a1 = t1.toArray()

// []<float32>
let a1b = t1.toArray() :?> float32[]

// Tensor
let t2 = dfx.randn([3;3;3])

// [,,]<float32>
let a2 = t2.toArray() :?> float32[,,]