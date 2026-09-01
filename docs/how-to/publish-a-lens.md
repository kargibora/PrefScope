# Publish a lens on the Hugging Face Hub

Publish an **inference-only** lens, not the training directory. The inference artifact
contains the SAE checkpoint, embedding/representation manifest, optional whitening
transform, and interpretation tables. It omits corpus text, training logs, and
corpus-aligned `z_*.npy` arrays.

## 1. Package and validate

```bash
prefscope package-lens \
  --lens-dir lenses/completion \
  --annotations interpret/completion \
  --model-card model-card.md \
  --out release/completion
```

For a repository containing prompt and response lenses, package them separately:

```bash
prefscope package-lens \
  --lens-dir lenses/prompt \
  --annotations interpret/prompt \
  --out release/prompt-m256

prefscope package-lens \
  --lens-dir lenses/completion \
  --annotations interpret/completion \
  --out release/completion-m2048

# Hugging Face renders the repository-root card, not a card inside a lens subfolder.
cp model-card.md release/README.md
```

Packaging uses a staged whole-directory replacement and also migrates the manifest to
the current schema. The packaged
manifest records the omitted arrays under `source_output_arrays`, declares
`artifact_scope: inference`, and declares no missing output arrays. Do not edit the JSON
by hand.

## 2. Write a model card

At minimum, document:

- the embedding model and revision;
- prompt/response lens subfolders, width, sparsity, representation, and SAE type;
- source datasets and licenses, without uploading private text;
- naming and verifier models, evidence counts, and number of fidelity-passing axes;
- whether semantic-presence calibration is complete;
- that these are output-embedding features, not causal internal-model features;
- the license and intended/unsupported uses.

The card must say when a signed legacy lens names only its positive pole. If calibration
is incomplete, examples should use `mixed` only with the `presence_basis` field visible.

## 3. Upload

Authenticate through the Hugging Face CLI; never put access tokens in a command file,
configuration, model card, or Git commit:

```bash
hf auth login
hf repo create owner/repository --type model --public --exist-ok
hf upload owner/repository release . \
  --commit-message "Publish PrefScope inference lenses"
```

Use `--private` instead of `--public` for a restricted team artifact. Consumers with
access authenticate once with `hf auth login`; PrefScope then uses the standard Hugging
Face cache and credentials. Pin a Hub commit or tag with `--revision` for reproducible
analysis.

## 4. Consumer smoke test

First test artifact loading without materializing the embedding model:

```python
from prefscope import Lens

prompt = Lens.from_pretrained(
    "owner/repository", subfolder="prompt-m256", device="cpu")
response = Lens.from_pretrained(
    "owner/repository", subfolder="completion-m2048", device="cpu")
print(len(prompt.feature_table), len(response.feature_table))
```

Then run one real embedding on appropriate hardware:

```bash
prefscope extract-concepts \
  --repo owner/repository \
  --prompt-subfolder prompt-m256 \
  --completion-subfolder completion-m2048 \
  --prompt "Explain why the sky is blue." \
  --completion "Shorter wavelengths scatter more strongly." \
  --device cuda
```
