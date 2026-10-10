(*** condition: prepare ***)
#I "../tests/Dfx.Tests/bin/Debug/net10.0"
#r "Dfx.Core.dll"
#r "Dfx.Data.dll"
#r "Dfx.Backends.Reference.dll"
#r "Dfx.Backends.Torch.dll"
// These are needed to make fsdocs --eval work. If we don't select a backend like this in the beginning, we get erratic behavior.
Dfx.dfx.config(backend=Dfx.Backend.Reference)
Dfx.dfx.seed(123)

(**
Test 
*)

open Dfx

dfx.config(backend=Backend.Reference)

let a = dfx.tensor([1,2,3])
printfn "%A" a
(*** include-fsi-output ***)