# Reports

`prefscope.reporting.Report` is a small container for numerical metrics and caller-defined
pandas tables. It does not run analyses or interpret table columns.

```python
from prefscope import Report

report = Report(
    title="Lens run",
    metrics={"n_rows": 100, "mean_l0": 3.4},
    metadata={"model": "example"},
)
report.add_table("feature_activity", feature_activity)
report.save("results/report")
```

The output is intentionally ordinary:

```text
results/report/
├── report.json
└── tables/
    └── table_0.csv
```

`report.json` stores the title, metrics, caller metadata, and a mapping from table names
to generated relative paths such as `tables/table_0.csv`.

`save` requires a new destination directory. An existing directory, even an empty one,
raises `FileExistsError` without changing its contents. Use a new path for each report.

`Report` does not provide privacy filtering, evidence taxonomies, statistical inference,
narrative generation, HTML rendering, or a report compiler. Callers decide what their
metrics mean and what data is safe to share.
