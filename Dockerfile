FROM mcr.microsoft.com/dotnet/sdk:10.0
WORKDIR /code/dfx
COPY . /code/dfx
RUN dotnet build
RUN dotnet test
