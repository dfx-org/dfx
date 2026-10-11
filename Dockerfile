FROM mcr.microsoft.com/dotnet/sdk:6.0
WORKDIR /code/dfx
COPY . /code/dfx
RUN dotnet build
RUN dotnet test
