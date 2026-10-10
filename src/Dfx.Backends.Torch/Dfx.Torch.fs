namespace Dfx

open Dfx
open Dfx.Backends.Torch
open TorchSharp

[<AutoOpen>]
module TorchExtensions =

    type dfx with

        /// <summary>
        /// Creates a new Dfx tensor from the torch tensor.
        /// </summary>
        static member fromTorch(tt: torch.Tensor) =
            Tensor.ofRawTensor(TorchRawTensor(tt))

    type Tensor with
        /// <summary>
        /// Converts the primal of a tensor to a torch tensor.
        /// </summary>
        /// <remarks>
        /// If the tensor does not use the Torch backend an exception is raised.
        ///
        /// Note that this operation takes the primal of the tensor. This means
        /// code that converts to Torch tensors will not be differentiable using
        /// Dfx differentiation capabilities.
        /// </remarks>
        member t.toTorch() =
            match t.primalRaw with
            | :? TorchRawTensor as trt -> trt.TorchTensor
            | _ -> failwith $"toTorch: the input is not a Dfx.Backends.Torch tensor, its backend is {t.backend}"

