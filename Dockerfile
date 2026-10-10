FROM mcr.microsoft.com/dotnet/sdk:6.0
WORKDIR /code/Dfx
COPY . /code/Dfx
RUN dotnet build
RUN dotnet test
