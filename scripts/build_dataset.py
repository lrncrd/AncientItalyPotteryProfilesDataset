"""
Build the machine-learning release of AIPPD from Appendix A of the book (Cardarelli 2024, ADRIAS 18,
doi:10.4475/0840; https://edipuglia.it/pasap01/Appendix_A.zip).

    python scripts/build_dataset.py path/to/Appendix_A        # the unzipped folder (or the .zip)

Writes data/:
    aippd_profiles_256.npz   masks, bit-packed: masks (N, 256, 32) uint8 -> np.unpackbits(..., axis=-1) gives (N, 256, 256)
                             bool; ids (N,)
    aippd.parquet            one row per vessel: identifiers, archaeological fields, measures, mask as a 1-bit PNG
    aippd_metadata.csv       the same table without the masks
    aippd_bibliography.csv   the 503 publications the drawings come from (bibliography_id -> reference)
    aippd_sites.geojson      the contexts (WGS84 lon/lat; converted from WGS 84 / UTM 33N)

Requires numpy, pandas, scipy, pillow, pyarrow, openpyxl, pyshp, pyproj.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"

# Exporting_query.xlsx -> release names (snake case, English)
COLUMNS = {
    "ids": "id", "context": "context", "ID_context": "context_id", "context_typology": "context_type",
    "Region": "region", "Facies": "facies", "wide_chronology": "period", "close_chronology": "subperiod",
    "ID_cronologia": "chronology_id", "Morphology": "morphology", "Functional_interpretation": "form",
    "Functional_class": "functional_class", "manufacturing_technique": "manufacturing_technique", "material": "material",
    "funeral_function": "funerary_function", "Grave": "grave", "ID_bibliography": "bibliography_id", "Pag.": "page",
    "Fig./Tav.": "figure_or_plate", "n.": "item_number", "Note": "note",
}
MEASURES = {"height (cm)": "height_cm", "width (cm)": "max_width_cm", "diameter (cm)": "rim_diameter_cm",
            "volume (l)": "capacity_l", "surface (mq)": "outer_surface_m2"}


def source(path: Path) -> Path:
    if path.suffix == ".zip":
        dest = path.with_suffix("")
        if not dest.exists():
            zipfile.ZipFile(path).extractall(dest.parent)
        path = dest
    return path / "Appendix_A" if (path / "Appendix_A").exists() else path


def png_bytes(mask: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(mask).convert("1").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def main(src: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shapefile
    from pyproj import Transformer

    src = source(src)
    OUT.mkdir(exist_ok=True)
    z = np.load(src / "pots_data" / "images_csr_rescaled_256.npz")
    m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    masks = m.toarray().reshape(-1, 256, 256).astype(bool)
    info = pd.read_csv(src / "pots_data" / "info.csv")
    assert len(info) == len(masks)

    tab = pd.read_excel(src / "Exporting_query.xlsx")
    tab = tab.set_index("ids").reindex(info["ids"]).reset_index()
    meta = tab.rename(columns=COLUMNS)[list(COLUMNS.values())]
    for k, v in MEASURES.items():
        meta[v] = info[k].values
    for c in ("context_id", "chronology_id", "bibliography_id"):
        meta[c] = meta[c].astype("Int64")
    for c in ("grave", "page", "figure_or_plate", "item_number"):
        meta[c] = meta[c].map(lambda v: None if pd.isna(v) else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)))
    meta["note"] = meta["note"].fillna("")
    meta.to_csv(OUT / "aippd_metadata.csv", index=False)

    np.savez_compressed(OUT / "aippd_profiles_256.npz", masks=np.packbits(masks, axis=-1), ids=info["ids"].values.astype(str))

    table = pa.Table.from_pandas(meta, preserve_index=False)
    table = table.append_column("mask_png", pa.array([png_bytes(mk) for mk in masks], type=pa.binary()))
    pq.write_table(table, OUT / "aippd.parquet", compression="zstd")

    bib = pd.read_excel(src / "full_bibliography.xlsx").rename(columns={
        "Bibliographic references index": "bibliography_id", "short_biblio": "short", "long_biblio": "reference",
        "Records for References": "n_vessels"})
    bib.to_csv(OUT / "aippd_bibliography.csv", index=False)

    to_wgs = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    shp = shapefile.Reader(str(src / "shp" / "sites"))
    counts = meta.groupby("context_id").size()
    feats = []
    for rec, shape in zip(shp.records(), shp.shapes()):
        x, y = shape.points[0]
        lon, lat = to_wgs.transform(x, y)
        cid = int(rec["ID"])
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                      "properties": {"context_id": cid, "context": rec["site"], "n_vessels": int(counts.get(cid, 0))}})
    json.dump({"type": "FeatureCollection", "features": feats}, open(OUT / "aippd_sites.geojson", "w"), ensure_ascii=False)

    for f in sorted(OUT.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 1e6:7.1f} MB")
    print(f"{len(meta)} vessels, {meta['context'].nunique()} contexts, {len(bib)} publications, {len(feats)} sites")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
