# Generating docs


To iterate on docs (requires evaluation off since DLLs get locked)

    dotnet fsdocs watch 

To use a local build of FSharp.Formatting:

       git clone https://github.com/fsprojects/FSharp.Formatting  ../FSharp.Formatting
       pushd ..\FSharp.Formatting
       .\build
       popd

Then:

       ..\FSharp.Formatting\src\FSharp.Formatting.CommandTool\bin\Debug\net10.0\fsdocs.exe watch
       ..\FSharp.Formatting\src\FSharp.Formatting.CommandTool\bin\Debug\net10.0\fsdocs.exe build --clean --eval

## Generated Notebooks

Notebooks are generated for all .md and .fsx files under docs as part of the build.

