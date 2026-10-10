// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

module Dfx.Shorten

// Functional automatic differentiation API shorthand names
type dfx with

    /// <summary>TBD</summary>
    static member gvp f x v = dfx.gradv f x v

    /// <summary>TBD</summary>
    static member g f x = dfx.grad f x

    /// <summary>TBD</summary>
    static member hvp f x v = dfx.hessianv f x v

    /// <summary>TBD</summary>
    static member h f x = dfx.hessian f x

    /// <summary>TBD</summary>
    static member gh f x = dfx.gradhessian f x

    /// <summary>TBD</summary>
    static member ghvp f x v = dfx.gradhessianv f x v

    /// <summary>TBD</summary>
    static member jvp f x v = dfx.jacobianv f x v

    /// <summary>TBD</summary>
    static member vjp f x v = dfx.jacobianTv f x v

    /// <summary>TBD</summary>
    static member j f x = dfx.jacobian f x

    /// <summary>TBD</summary>
    static member fgvp f x v = dfx.fgradv f x v

    /// <summary>TBD</summary>
    static member fg f x = dfx.fgrad f x

    /// <summary>TBD</summary>
    static member fgh f x = dfx.fgradhessian f x

    /// <summary>TBD</summary>
    static member fhvp f x v = dfx.fhessianv f x v

    /// <summary>TBD</summary>
    static member fh f x = dfx.fhessian f x

    /// <summary>TBD</summary>
    static member fghvp f x v = dfx.fgradhessianv f x v

    /// <summary>TBD</summary>
    static member fjvp f x v = dfx.fjacobianv f x v

    /// <summary>TBD</summary>
    static member fvjp f x v = dfx.fjacobianTv f x v

    /// <summary>TBD</summary>
    static member fj f x = dfx.fjacobian f x

