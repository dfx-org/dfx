#!/usr/bin/env -S dotnet fsi

#I "../tests/Dfx.Tests/bin/Debug/net10.0"
#r "Dfx.Core.dll"
#r "Dfx.Data.dll"
#r "Dfx.Backends.Torch.dll"

// Libtorch binaries
// Option A: you can use a platform-specific nuget package
#r "nuget: TorchSharp-cpu, 0.96.5"
// #r "nuget: TorchSharp-cuda-linux, 0.96.5"
// #r "nuget: TorchSharp-cuda-windows, 0.96.5"
// Option B: you can use a local libtorch installation
// System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open Dfx
open Dfx.Compose
open Dfx.Model
open Dfx.Data
open Dfx.Optim
open Dfx.Util
open Dfx.Distributions

open System.IO

dfx.config(backend=Backend.Torch, device=Device.CPU)
dfx.seed(1)

type Model<'In, 'Out> with
    member m.run = m.forward
type DiffProg<'In, 'Out> = Model<'In, 'Out>


let diffprog parameters (f:'In->'Out) : DiffProg<'In, 'Out>=
    DiffProg<'In, 'Out>.create [] parameters [] f

let param (x:Tensor) = Parameter(x)

// Learn a differentiable program given an objective
// DiffProg<'a,'b> -> (DiffProg<'a,'b> -> Tensor) -> DiffProg<'a,'b>
let learn (diffprog:DiffProg<_,_>) loss =
    let lr = 0.001
    for i=0 to 10 do
        diffprog.reverseDiff()
        let l:Tensor = loss diffprog
        l.reverse()
        let p = diffprog.parametersVector
        diffprog.parametersVector <- p.primal - lr * p.derivative
        printfn "iteration %A, loss %A" i (float l)
    diffprog

// A linear model as a differentiable program
// DiffProg<Tensor,Tensor>
let dp =
    let w = param (dfx.randn([5; 1]))
    diffprog [w] 
        (fun (x:Tensor)  -> x.matmul(w.value))

// Data
let x = dfx.randn([1024; 5])
let y = dfx.randn([1024; 1])

// let a = diffprog.run x
// printfn "%A %A %A " a.shape y.shape (dfx.mseLoss(a, y))

// Objective
// DiffProg<Tensor,Tensor> -> Tensor
let loss (diffprog:DiffProg<Tensor, Tensor>) = dfx.mseLoss(diffprog.run x, y)

// Learned diferentiable program
// DiffProg<Tensor,Tensor>
let dpLearned = learn dp loss

// Function that runs the differentiable program with new data
// Tensor -> Tensor
dpLearned.run 