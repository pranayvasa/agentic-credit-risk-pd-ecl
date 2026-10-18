"""Node 2: train / test / pseudo-OOT split on SK_ID_CURR."""

from __future__ import annotations

from acr.audit import node
from acr.core.io import read
from acr.core.schema import ID, TARGET
from acr.core.split import PARTITIONS, split_population, summary
from acr.nodes._util import artefact_dir, cfg


@node(
    "split",
    role="Optional stratified customer sample, then stratified 75:25 (dev:OOT) and 75:25 "
    "(train:test) split on SK_ID_CURR; persist ID lists.",
    reads=["config", "sample_frac", "dq"],
    writes=["split"],
)
def split(view):
    c = cfg(view)
    app = read(view["dq"]["tables"]["application"]["path"], [ID, TARGET])
    assign = split_population(app, c.base.split.oot_frac, c.base.split.test_frac_of_dev,
                              c.base.seed, view.get("sample_frac"))  # fmt: skip
    out = artefact_dir(view, "split")
    assign.to_parquet(out / "assignment.parquet", index=False)
    paths = {}
    for p in PARTITIONS:
        f = out / f"{p}_ids.parquet"
        assign.loc[assign["partition"] == p, [ID]].to_parquet(f, index=False)
        paths[p] = str(f)
    s = summary(assign)
    s.update({"assignment_path": str(out / "assignment.parquet"), "id_paths": paths,
              "sample_frac": view.get("sample_frac"), "pseudo_oot": True})  # fmt: skip
    return {"split": s}
