// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

module Dfx.Numerical

// Functional numerical differentiation API
type dfx with

    /// <summary>TBD</summary>
    static member numdiff (epsilon:float) (f:Tensor->Tensor) (x:Tensor) =
        if x.dim <> 0 then failwithf "f must be a function of a scalar"
        ((f (x + epsilon)) - (f (x - epsilon))) / (2.*epsilon)

    /// <summary>TBD</summary>
    static member numfdiff epsilon f x = f x, dfx.numdiff epsilon f x

    /// <summary>TBD</summary>
    static member numfdiff2 (epsilon:float) (f:Tensor->Tensor) (x:Tensor) =
        if x.dim <> 0 then failwithf "f must be a function of a scalar"
        let fx = f x
        fx, ((f (x + epsilon)) - 2. * fx + (f (x - epsilon))) / (epsilon * epsilon)

    /// <summary>TBD</summary>
    static member numdiff2 epsilon f x = dfx.numfdiff2 epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numjacobianv (epsilon:float) (f:Tensor->Tensor) (x:Tensor) (v:Tensor) =
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let veps = v * epsilon
        let fxa, fxb = f (x+veps), f (x-veps)
        if x.dim <> 1 || fxa.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fxa.shape
        (fxa - fxb) / (2.*epsilon)

    /// <summary>TBD</summary>
    static member numfjacobianv epsilon f x v = f x, dfx.numjacobianv epsilon f x v

    /// <summary>TBD</summary>
    static member numfjacobian (epsilon:float) (f:Tensor->Tensor) (x:Tensor) =
        let fx = f x
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        let j = fx.expand([x.nelement; fx.nelement])
        let jj = dfx.stack(Array.init x.nelement (fun i -> f (x + dfx.onehot(x.nelement, i)*epsilon)))
        fx, (jj - j).transpose() / epsilon

    /// <summary>TBD</summary>
    static member numjacobian epsilon f x = dfx.numfjacobian epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numgradv (epsilon:float) (f:Tensor->Tensor) (x:Tensor) (v:Tensor) =
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let veps = v * epsilon
        let fxa, fxb = f (x + veps), f (x - veps)
        if x.dim <> 1 || fxa.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fxa.shape
        (fxa - fxb) / (2.*epsilon)

    /// <summary>TBD</summary>
    static member numfgradv epsilon f x v = f x, dfx.numgradv epsilon f x v

    /// <summary>TBD</summary>
    static member numfgrad (epsilon:float) (f:Tensor->Tensor) (x:Tensor) =
        if x.dim = 0 then
            dfx.numfdiff epsilon f x
        else
            let fx = f x
            if x.dim > 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector or scalar, encountered f:%A->%A" x.shape fx.shape
            let gg = dfx.stack(Array.init x.nelement (fun i -> let h = dfx.onehot(x.nelement, i)*epsilon in f (x + h) - f (x - h)))
            fx, gg/(2.*epsilon)

    /// <summary>TBD</summary>
    static member numgrad epsilon f x = dfx.numfgrad epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfgradhessian (epsilon:float) (f:Tensor->Tensor) (x:Tensor) =
        let fx, g = dfx.numfgrad epsilon f x
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        let h = g.expand([x.nelement; x.nelement])
        let hh = dfx.stack(Array.init x.nelement (fun i -> dfx.numgrad epsilon f (x + dfx.onehot(x.nelement, i)*epsilon)))
        fx, g, (hh - h) / epsilon

    /// <summary>TBD</summary>
    static member numgradhessian epsilon f x = let _, g, h = dfx.numfgradhessian epsilon f x in g, h

    /// <summary>TBD</summary>
    static member numfhessian epsilon f x = let fx, _, h = dfx.numfgradhessian epsilon f x in fx, h

    /// <summary>TBD</summary>
    static member numhessian epsilon f x = dfx.numfhessian epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfhessianv (epsilon:float) (f:Tensor->Tensor) (x:Tensor) (v:Tensor) =
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let veps = v*epsilon
        let fx, g = dfx.numfgrad epsilon f x
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        let gg = dfx.numgrad epsilon f (x + veps)
        fx, (gg-g)/epsilon

    /// <summary>TBD</summary>
    static member numhessianv epsilon f x v = dfx.numfhessianv epsilon f x v |> snd

    /// <summary>TBD</summary>
    static member numflaplacian epsilon f x =
        let fx, h = dfx.numfhessian epsilon f x
        fx, h.trace()

    /// <summary>TBD</summary>
    static member numlaplacian epsilon f x = dfx.numflaplacian epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfcurl epsilon f x =
        let fx, j = dfx.numfjacobian epsilon f x
        if j.shape <> [|3; 3|] then failwithf "f must be a function with a three-by-three Jacobian"
        fx, dfx.stack([j[2, 1] - j[1, 2]; j[0, 2] - j[2, 0]; j[1, 0] - j[0, 1]])

    /// <summary>TBD</summary>
    static member numcurl epsilon f x = dfx.numfcurl epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfdivergence epsilon f x =
        let fx, j = dfx.numfjacobian epsilon f x
        if j.shape[0] <> j.shape[1] then failwithf "f must have a square Jacobian"
        fx, j.trace()

    /// <summary>TBD</summary>
    static member numdivergence epsilon f x = dfx.numfdivergence epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfcurldivergence epsilon f x =
        let fx, j = dfx.numfjacobian epsilon f x
        if j.shape <> [|3; 3|] then failwithf "f must be a function with a three-by-three Jacobian"
        fx, dfx.stack([j[2, 1] - j[1, 2]; j[0, 2] - j[2, 0]; j[1, 0] - j[0, 1]]), j.trace()

    /// <summary>TBD</summary>
    static member numcurldivergence epsilon f x = let _, c, d = dfx.numfcurldivergence epsilon f x in c, d


module Shorten =
    // Functional numerical differentiation API shorthand names
    type dfx with

        /// <summary>TBD</summary>
        static member numgvp f x v = dfx.numgradv f x v

        /// <summary>TBD</summary>
        static member numg f x = dfx.numgrad f x

        /// <summary>TBD</summary>
        static member numhvp f x v = dfx.numhessianv f x v

        /// <summary>TBD</summary>
        static member numh f x = dfx.numhessian f x

        /// <summary>TBD</summary>
        static member numgh f x = dfx.numgradhessian f x

        /// <summary>TBD</summary>
        static member numjvp f x v = dfx.numjacobianv f x v

        /// <summary>TBD</summary>
        static member numj f x = dfx.numjacobian f x

        /// <summary>TBD</summary>
        static member numfgvp f x v = dfx.numfgradv f x v

        /// <summary>TBD</summary>
        static member numfg f x = dfx.numfgrad f x

        /// <summary>TBD</summary>
        static member numfhvp f x v = dfx.numfhessianv f x v

        /// <summary>TBD</summary>
        static member numfh f x = dfx.numfhessian f x

        /// <summary>TBD</summary>
        static member numfgh f x = dfx.numfgradhessian f x

        /// <summary>TBD</summary>
        static member numfjvp f x v = dfx.numfjacobianv f x v

        /// <summary>TBD</summary>
        static member numfj f x = dfx.numfjacobian f x