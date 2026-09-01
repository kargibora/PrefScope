# Examples

The installed CLI is the primary interface; run `prefscope --help` for supported
operations. These files support source-checkout tutorials or demonstrate composition of
the public Python API:

- `sample_corpus.parquet`, `quickstart.yaml`: tiny end-to-end smoke test. A pip-only
  installation can generate the same workspace with `prefscope init-demo --out demo`.
- `research.yaml`: higher-budget naming and held-out verification profile.
- `pipeline.yaml`: annotated configuration template for an existing lens.
- `compare_model_stages.py`: label-free pre/post-training comparison using
  `compare_encoded_responses`.
- `make_sample_corpus.py`: regenerate the tracked synthetic corpus.

No file here contains real model output, credentials, or private data.
