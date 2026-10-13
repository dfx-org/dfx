# Security Policy

## Reporting a vulnerability

Please don't report security vulnerabilities in public issues. Report them privately through GitHub instead: on the repository's **Security** tab, choose **Report a vulnerability**, or go directly to https://github.com/dfx-org/dfx/security/advisories/new.

The maintainers will respond as soon as possible and coordinate a fix and its disclosure with you.

## Supported versions

Security fixes are made in the latest release.

## Loading files from untrusted sources

`dfx.load` and `Tensor.load` read files with .NET's `BinaryFormatter`, which can execute code embedded in a crafted file. Only load files from sources you trust.
