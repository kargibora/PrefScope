# Security policy

PrefScope loads third-party model and lens artifacts. Checkpoints are opened with
PyTorch's restricted `weights_only=True` loader; do not weaken that boundary or load an
untrusted pickle manually.

Report a suspected vulnerability through GitHub's private vulnerability reporting for
this repository. Do not post credentials, exploit details, private model outputs, or
restricted datasets in a public issue.

Only the latest released alpha receives security fixes. Because the API is pre-1.0,
security-related hardening may require a breaking change documented in the changelog.
