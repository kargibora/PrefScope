# Feature bundle reader

`FeatureBundleReader` is the Torch-free, bounded reporting source for an existing
**schema-2** encoded feature bundle. It validates the complete bundle and exposes its
arrays as live, read-only memory maps.

```python
from prefscope.reporting import FeatureBundleReader

source = FeatureBundleReader.open("encoded/dataset")
for chunk in source.iter_chunks(4096, views=("z_a", "z_b")):
    use(chunk.row_ids, chunk.array("z_a"))
```

`open(...)` has conservative positive-integer limits. Current defaults are:

| limit | default |
|---|---:|
| rows / feature width / elements per view | 1,000,000 / 1,000,000 / 1,000,000,000 |
| manifest bytes | 8 MiB |
| each metadata twin | 512 MiB |
| both metadata twins, compressed | 1 GiB |
| metadata columns / cells / row groups | 512 / 8,000,000 / 4,096 |
| both metadata twins, declared uncompressed | 1 GiB |

Use the corresponding `max_rows`, `max_features`, `max_elements`,
`max_manifest_bytes`, `max_metadata_file_bytes`, `max_metadata_compressed_bytes`,
`max_metadata_columns`, `max_metadata_cells`, `max_metadata_row_groups`, and
`max_metadata_uncompressed_bytes` arguments to set smaller trust-boundary budgets.
`max_elements` is per view. The reader currently has no independent view-count or
aggregate feature-array-element budget; `max_manifest_bytes` only indirectly limits view
count.

## Open and memory semantics

`open(...)` is lazy with respect to materializing feature arrays, but it is not an
unverified or constant-time open. It:

- requires the exact schema-2 file inventory from `manifest.json`;
- requires `meta.parquet` and `battles.parquet` to be byte-for-byte identical, then
  bounds their Parquet footer shape before decoding only `meta.parquet`;
- accepts only NPY v1/v2 views with the exact declared two-dimensional shape,
  canonical `float32`/`bool` dtype, C order, bounded header, and no trailing bytes;
- verifies metadata alignment, per-view role/orientation/polarity/semantics, portable
  credential-free provenance, finite float values, and the ordered `dataset_hash`;
- scans arrays in bounded row chunks during validation, without building a whole-array
  finiteness mask or a `FeatureBatch`;
- detects file replacement or mutation during open.

`array(view)` then returns the complete live, read-only `numpy.memmap`. Read-only means
callers cannot write through that array. It does **not** freeze the backing file. Another
process can change bytes that an existing reader observes after `open()` returns. Do not
use a directory with concurrent writers. Publish it first, then open it read-only.

Metadata and identity are exposed through `row_ids`, `feature_ids`, `view_names`,
`roles`, `orientations`, `activation_polarities`, `code_semantics_by_view`, `metadata`,
`provenance`, `n_rows`, and `n_features`. The `FeatureSource` protocol uses the canonical
names `view_names` and `code_semantics_by_view`. `FeatureBundleReader` alone also exposes
`views` and `code_semantics` as compatibility aliases; both semantics properties return
the resolved per-view mapping.

## Alignment and selection

There is no implicit join or reorder:

- `assert_row_ids(ids)` requires the complete row-ID sequence in exact source order.
- `row_positions(ids)` requires unique known IDs and returns their source positions in
  requested order.
- `iter_chunks(chunk_size)` yields contiguous source-order slices for every view.
- `iter_chunks(..., views=(...))` selects unique declared views in the requested order.
- `iter_chunks(..., row_ids=(...))` preserves the explicit requested row order. Because
  NumPy fancy indexing is needed, each selected chunk is a bounded, read-only copy rather
  than a memory-map slice.

Every `FeatureChunk` keeps aligned row IDs, source positions, arrays, and metadata.
Contiguous chunks expose `row_slice`; noncontiguous selections do not.

## Schema 1 is an explicit migration

`FeatureBundleReader` does not infer or adapt schema 1. Use the compatibility loader,
then write a new schema-2 directory:

```python
from prefscope import load_feature_batch, save_feature_batch
from prefscope.reporting import FeatureBundleReader

legacy = load_feature_batch("encoded/schema1")  # explicit schema-1 compatibility path
save_feature_batch(legacy, "encoded/schema2")
source = FeatureBundleReader.open("encoded/schema2")
```

`load_feature_batch(...)` supports schemas 1 and 2 and materializes a validated
`FeatureBatch`. It is the eager compatibility path: it validates and loads **all declared
arrays**, even when `arrays=` returns only selected views. Its fixed preflight budgets are
1,000,000 rows, width 1,000,000, 128 arrays, and 100,000,000 aggregate declared array
elements. Its manifest is limited to 8 MiB and depth 64. Each metadata Parquet is limited
to 512 MiB compressed, 512 columns, 4,096 row groups, and 1 GiB declared uncompressed;
the two twins also share a 1 GiB aggregate uncompressed budget.

`save_feature_batch(...)` publishes schema 2. It stages and validates a complete bundle.
The writer enforces the same 1,000,000-row and 100,000,000 aggregate declared-array-element
limits. These are hard safety limits, not a claim that an in-memory workflow is practical
at either bound.

A saved `FeatureBatch` retains row IDs, requested feature arrays, and every supplied
metadata column. `Lens.featurize(...)` outputs include prompt/response text and label
fields, so bundles from that path must be treated as **local, private artifacts**. They
are not shareable or redacted report bundles. Store them in an access-controlled location
and do not upload them merely because their arrays are numerical.

For a new destination on Darwin and Linux, the final no-replace rename preserves a racing
occupant. For overwrite, the writer validates and rechecks the existing managed
destination before a recoverable multi-rename transaction. This does not guarantee
preservation of an uncooperative replacement made after the final identity check.
Replacement is not a linearizable atomic directory exchange for readers or process death.
The persistent `.lock` file uses advisory `flock`, requires a trusted parent plus a
current-user, single-link inode before mutation, and is deliberately never unlinked; do
not delete it. Keep the old schema-1 directory unchanged; do not edit its manifest in
place.

See also [Python API](python-api.md), [Durable analysis results](analysis-result.md), and
[Report bundles](report-bundle.md).
