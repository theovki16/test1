"""
F7_pressure_maps.py
Locate pressure-work and pressure-transport contributions in the CDA passage.

Reads F0/F1 fields, never changes them, and uses the exact cross-case band masks.
All maps use each case's signed, all-region, nonzero-frequency production total.
Raw midspan Fourier modes also permit an illustrative instantaneous xy analysis.
That analysis omits spanwise derivatives and is NOT the full 3-D pressure budget.
"""
from plot_axis_format import format_figure
from pathlib import Path
from io import BytesIO
from tempfile import NamedTemporaryFile
import csv
import hashlib
import json
import sys

import numpy as np
from scipy.io import loadmat
from fourier_loss_native import f0_input_scale
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FuncFormatter

# User settings.
FOURIER_BASE = Path(__file__).resolve().parent.parent
SUPPORT_BASE = Path(r"C:\Users\Dell User\Documents\GenoVa\For_Theo\Mean_data_3DNS")
OUT_DIR = FOURIER_BASE / "F7_pressure_maps"
CASES = {"CDA_44deg": -8, "CDA_48deg": -4, "CDA_52deg": 0,
         "CDA_56deg": 4, "CDA_60deg": 8}
TERMS = ["P", "Pi", "Dp"]
REGIONS = ["passage", "SS-BL", "PS-BL", "wake", "all"]
BANDS = ["bp", "low", "bb", "all"]
NBCHUNK = {1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4, 7: 1, 8: 4, 9: 4}
SNAPSHOT_INDEX = 0                  # Relative to the selected Fourier record; change to inspect another instant.
RECONSTRUCT_INSTANT = True
USE_CACHE = True                    # Reuse 2-D fields only when their input fingerprints still match.
CACHE_VERSION = 1
ROWS_PER_READ = 4096                # Bound memory while reading the large frequency fields.
PLOT_STRIDE = 4                     # Display thinning only; every integral uses the full grid.
X_LIM = (-0.28, 2.33)
Y_LIM = (-0.10, 1.32)
LINEAR_LIMIT = 0.01                 # Linear colour range in normalized integrated-entry units.
DPI = 300
sys.path.insert(0, str(FOURIER_BASE / "local"))
from F1_compare_cases import make_bands, DET_HALF, LOW_MAX
plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def atomic_bytes(path, content):
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(content)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def signature(paths, settings):
    records = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    payload = json.dumps({"files": records, "settings": settings}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_load(path, fingerprint):
    if USE_CACHE and path.is_file():
        with np.load(path, allow_pickle=False) as z:
            if str(z["fingerprint"]) == fingerprint:
                return {k: z[k].copy() for k in z.files if k != "fingerprint"}
    return None


def cache_save(path, fields, fingerprint):
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temp = Path(stream.name)
        np.savez_compressed(stream, fingerprint=fingerprint, **fields)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def field_key(term, band, name):
    return f"{term}__{band}__{name}"


def load_support(case):
    pp = SUPPORT_BASE / case / "pp"
    with np.load(pp / "grid/topology.npz") as z:
        topo = {k: z[k].copy() for k in z.files}
    with np.load(pp / "mean_time/reference.npz") as z:
        rho1, V1 = float(z["rho1"]), float(z["V1"])
    names = topo["names"].astype(str).tolist()
    grids = {}
    for name in names:
        with np.load(pp / f"grid/grid_{name}.npz") as g, np.load(pp / f"masks/label_{name}.npz") as m:
            if not np.array_equal(g["w"], m["weight"]):
                raise ValueError(f"{case}/{name}: grid/mask weights differ.")
            grids[name] = {k: g[k].copy() for k in ["x", "y", "w"]}
            grids[name]["area_w"] = g["w"].copy()  # Only reconstructed xy densities need area integration.
            grids[name]["w"] = np.ones_like(g["w"])  # Integrated-entry sums, no new area factor.
            grids[name]["label"] = m["label"].copy()
            if name == "ring":
                topo["rmap"] = g["rmap"].copy()
    with np.load(pp / "masks/polygons.npz") as z:
        polygons = [z[k].copy() for k in ["ss", "ps", "wake"]]
    return {"case": case, "incidence": CASES[case], "topo": topo,
            "rho1": rho1, "V1": V1, "names": names, "grids": grids, "polygons": polygons}


def read_band_fields(path, masks):
    a = np.load(path, mmap_mode="r")
    if a.ndim != 3 or a.shape[-1] != masks.shape[1] or not a.flags.c_contiguous:
        raise ValueError(f"Unexpected modal field layout: {path}, {a.shape}")
    shape = a.shape[:2]
    flat = a.reshape(-1, a.shape[-1])
    values = np.empty((len(masks), len(flat)), dtype=np.float64)
    for start in range(0, len(flat), ROWS_PER_READ):
        stop = min(start + ROWS_PER_READ, len(flat))
        part = np.asarray(flat[start:stop], dtype=np.float64)
        if not np.isfinite(part).all():
            raise ValueError(f"Non-finite source field: {path}")
        for ib, mask in enumerate(masks):
            values[ib, start:stop] = part[:, mask].sum(axis=1)
    del part, flat
    a._mmap.close()
    return [v.reshape(shape) for v in values]


def aggregate_case(case):
    data = load_support(case)
    fou = FOURIER_BASE / case / "pp/fourier"
    with np.load(fou / "layout.npz") as z:
        N = int(z["N"])
        f1 = np.abs(z["f_bp"][:N // 2 + 1])
        data["periods"] = float(z["periods"])
    with np.load(fou / "spectra.npz") as z:
        spec = z["spec"].copy()
        compare_f1 = "budget_convention" in z and str(z["budget_convention"]) == "cell_volume_weighted_span_summed_v1"
        terms, regions = z["terms"].astype(str).tolist(), z["regions"].astype(str).tolist()
        np.testing.assert_allclose(f1, z["f_bp"], rtol=1e-12, atol=1e-12)
    one = make_bands(f1)
    two = np.zeros((3, N), dtype=bool)
    two[:, :N // 2 + 1] = one
    two[:, N - 1:N // 2:-1] = one[:, 1:N // 2]
    if np.any(two[:, 0]) or not np.all(two[:, 1:].sum(axis=0) == 1):
        raise ValueError("The bands must partition all nonzero two-sided bins once.")
    present = [t for t in TERMS if t in terms]
    paths = [fou / f"{t}_{name}.npy" for t in present for name in data["names"]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    fingerprint = signature(paths + [fou / "layout.npz"],
                            {"version": CACHE_VERSION, "det_half": DET_HALF, "low_max": LOW_MAX})
    target = OUT_DIR / f"fields_{case}.npz"
    fields = cache_load(target, fingerprint)
    if fields is None:
        fields = {}
        for term in present:
            print(f"{case}: reducing {term} on all six arrays ...", flush=True)
            for name in data["names"]:
                arrays = read_band_fields(fou / f"{term}_{name}.npy", two)
                for band, value in zip(BANDS[:3], arrays):
                    fields[field_key(term, band, name)] = value
                fields[field_key(term, "all", name)] = sum(arrays)
        cache_save(target, fields, fingerprint)
    # Cache values retain F0 storage units; convert once on load, without spatial weights.
    scale = f0_input_scale(case, 'P')
    fields = {key: value * scale for key, value in fields.items()}
    data.update({"fields": fields, "present": present, "N": N})
    data["P_ref"] = integral(data, "P", "all")  # Use the same summed entries as the maps.
    if not np.isfinite(data["P_ref"]) or abs(data["P_ref"]) < 1e-12:
        raise ValueError(f"{case}: production reference is unavailable or effectively zero.")
    # Verify frequency-family and spatial partitions independently of old metadata.
    for term in present:
        for name in data['names']:
            total = get_field(data, term, 'all', name)
            families = sum(get_field(data, term, band, name) for band in BANDS[:3])
            np.testing.assert_allclose(total, families, rtol=1e-10, atol=1e-10)
        for ib, band in enumerate(BANDS):
            actual = integral(data, term, band)
            regional = sum(integral(data, term, band, region) for region in REGIONS[:4])
            np.testing.assert_allclose(actual, regional, rtol=1e-10, atol=1e-8)
            if compare_f1:
                mask = one[ib] if ib < 3 else (np.arange(len(f1)) > 0)
                for region in REGIONS:
                    expected = spec[terms.index(term), regions.index(region), mask].sum()
                    np.testing.assert_allclose(integral(data, term, band, region), expected,
                                               rtol=1e-8, atol=1e-8)
    print(f"{case}: all regional integrals verified; P_ref={data['P_ref']:+.10e}.", flush=True)
    return data


def get_field(data, term, band, name):
    fields = data["fields"]
    if term == "Pi-Dp":
        a, b = fields.get(field_key("Pi", band, name)), fields.get(field_key("Dp", band, name))
        return None if a is None or b is None else a - b
    return fields.get(field_key(term, band, name))


def integral(data, term, band, region="all"):
    value = 0.0
    for name, g in data["grids"].items():
        q = get_field(data, term, band, name)
        if q is None:
            return np.nan
        mask = g["w"] > 0
        if region != "all":
            mask &= g["label"] == REGIONS.index(region)
        value += np.sum(q[mask] * g["w"][mask])
    return float(value)


def statistics(data, term, band, region="all"):
    positive = negative = 0.0
    peak, peak_x, peak_y, peak_array = 0.0, np.nan, np.nan, ""
    for name, g in data["grids"].items():
        q = get_field(data, term, band, name)
        if q is None:
            return None
        take = g["w"] > 0
        if region != "all":
            take &= g["label"] == REGIONS.index(region)
        v = q[take] * g["w"][take]
        positive += v[v > 0].sum()
        negative += v[v < 0].sum()
        score = np.where(take, np.abs(q), -1.0)
        pos = np.unravel_index(np.argmax(score), score.shape)
        if score[pos] > abs(peak):
            peak, peak_x, peak_y, peak_array = float(q[pos]), float(g["x"][pos]), float(g["y"][pos]), name
    signed, absolute = positive + negative, positive - negative
    return {"case": data["case"], "incidence_deg": data["incidence"], "term": term,
            "band": band, "region": region, "signed_integral": signed,
            "positive_integral": positive, "negative_integral": negative,
            "absolute_spatial_integral": absolute, "P_reference": data["P_ref"],
            "signed_over_P_total": signed / data["P_ref"],
            "absolute_over_P_total": absolute / data["P_ref"],
            "spatial_cancellation": 1 - abs(signed) / absolute if absolute else 0.0,
            "peak_signed_density": peak / data["P_ref"],
            "peak_x_cax": peak_x, "peak_y_cax": peak_y, "peak_array": peak_array}


def configured_record(case, job):
    p = FOURIER_BASE / case / f"submission_fourier/{job}_block.sh"
    result = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        for key in ["NSNAPS", "FIRST"]:
            if line.startswith(key + "=") and ":-" in line:
                result[key] = int(line.split(":-", 1)[1].split("}", 1)[0])
    return result


def reconstruct_frame(path, N, frame, scale):
    a = np.load(path, mmap_mode="r")
    if a.ndim != 3 or a.shape[-1] != N or not np.iscomplexobj(a):
        raise ValueError(f"Invalid midspan modes: {path}")
    flat = a.reshape(-1, N)
    phase = np.exp(2j * np.pi * np.arange(N) * frame / N)
    phase[0] = 0.0                  # Exact nonzero-frequency reconstruction; removes each point's record mean.
    out = np.empty(len(flat), dtype=np.float64)
    sample = np.asarray(flat[:min(16, len(flat))], dtype=np.complex128).copy()
    direct = sample @ phase
    inverse = np.fft.ifft(sample * N, axis=1)[:, frame] - sample[:, 0]
    np.testing.assert_allclose(direct, inverse, rtol=1e-10, atol=1e-8)
    if np.max(np.abs(direct.imag)) > 1e-5 * max(float(np.max(np.abs(direct.real))), 1e-8):
        raise ValueError(f"Modes do not reconstruct a real field: {path}")
    for start in range(0, len(flat), ROWS_PER_READ):
        stop = min(start + ROWS_PER_READ, len(flat))
        part = np.asarray(flat[start:stop])
        if frame == 0:
            out[start:stop] = part[:, 1:].real.sum(axis=1, dtype=np.float64) / scale
        else:
            out[start:stop] = (part @ phase).real / scale
    shape = a.shape[:2]
    del part, flat
    a._mmap.close()
    return out.reshape(shape)


def derivatives_xy(x, y):
    xi, xj = np.gradient(x, axis=(0, 1), edge_order=2)
    yi, yj = np.gradient(y, axis=(0, 1), edge_order=2)
    J = xi * yj - xj * yi
    if np.any(np.abs(J) < 1e-18):
        raise ValueError("Singular in-plane grid metric.")
    # Same metric order and data-gradient edge order as the HPC helpers.
    dix, djx, diy, djy = [(v / J).astype(np.float32).astype(float)
                         for v in (yj, -yi, -xj, xi)]
    def grad(f):
        fi, fj = np.gradient(f, axis=(0, 1), edge_order=1)
        return fi * dix + fj * djx, fi * diy + fj * djy
    return grad


def to_local_2d(blocks, data):
    rmap = data["topo"]["rmap"]
    ring = np.stack([blocks[int(b)][int(i), ::-1] for b, i in rmap])
    return {"ring": ring, **{name: blocks[int(name)] for name in data["names"] if name != "ring"}}


def instantaneous_case(data, reference=None):
    case, N = data["case"], data["N"]
    raw = FOURIER_BASE / case / "STATS_FOURIER"
    paths = [raw / f"modes_{q}_{c}_{b}.npy" for q in ["p", "u", "v"]
             for b, count in NBCHUNK.items() for c in range(count)]
    if not any(raw.glob("modes_p_*.npy")):
        return None
    if reference is None:
        print(case + ': optional instantaneous comparison needs a separately matched reference.', flush=True)
        return None
    data["instant_P_ref"] = data["P_ref"] / float(reference)
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(f"{case}: an instantaneous reconstruction needs every p/u/v chunk.")
    if not 0 <= SNAPSHOT_INDEX < N:
        raise ValueError("SNAPSHOT_INDEX must be within the selected record.")
    spec_record, pressure_record = configured_record(case, "spec"), configured_record(case, "dp")
    if spec_record != pressure_record or spec_record.get("NSNAPS") != N:
        raise ValueError(f"{case}: configured pressure/velocity records differ.")
    grid_path = FOURIER_BASE / case / "grid/grid_split.mat"
    ref_path = SUPPORT_BASE / case / "pp/mean_time/reference.npz"
    fingerprint = signature(paths + [grid_path, ref_path],
                            {"version": CACHE_VERSION, "frame": SNAPSHOT_INDEX, "record": spec_record})
    target = OUT_DIR / f"instant_xy_{case}_q{SNAPSHOT_INDEX:03d}.npz"
    cached = cache_load(target, fingerprint)
    if cached is not None:
        data["instant"] = cached
        return data
    print(f"{case}: reconstructing midspan nonzero-frequency snapshot q={SNAPSHOT_INDEX} ...", flush=True)
    gm = loadmat(grid_path, squeeze_me=True, struct_as_record=False)["grid"]
    maps = {q: {} for q in ["Pi_xy", "Dp_xy", "p_div_xy", "product_defect_xy", "wz_xy"]}
    rho1, V1, cax = data["rho1"], data["V1"], float(data["topo"]["L_ref"])
    for b, count in NBCHUNK.items():
        x, y = np.asarray(gm[b - 1].x), np.asarray(gm[b - 1].y)
        nxb = x.shape[0] // count
        block = {q: np.empty(x.shape, dtype=np.float64) for q in maps}
        for c in range(count):
            u = reconstruct_frame(raw / f"modes_u_{c}_{b}.npy", N, SNAPSHOT_INDEX, V1)
            v = reconstruct_frame(raw / f"modes_v_{c}_{b}.npy", N, SNAPSHOT_INDEX, V1)
            p = reconstruct_frame(raw / f"modes_p_{c}_{b}.npy", N, SNAPSHOT_INDEX, rho1 * V1**2)
            sl = slice(c * nxb, (c + 1) * nxb)
            grad = derivatives_xy(x[sl], y[sl])
            px, py = grad(p)
            ux, uy = grad(u)
            vx, vy = grad(v)
            pi = -cax * (u * px + v * py)
            dp = -cax * (grad(p * u)[0] + grad(p * v)[1])
            pd = cax * p * (ux + vy)
            block["Pi_xy"][sl] = pi
            block["Dp_xy"][sl] = dp
            block["p_div_xy"][sl] = pd
            block["product_defect_xy"][sl] = pi - dp - pd
            block["wz_xy"][sl] = cax * (vx - uy)
        for q in maps:
            maps[q][b] = block[q]
    fields = {}
    for term, blocks in maps.items():
        for name, value in to_local_2d(blocks, data).items():
            fields[field_key(term, "instant", name)] = value
    fields["snapshot_index"] = np.array(SNAPSHOT_INDEX)
    fields["configured_first_column"] = np.array(spec_record["FIRST"])
    fields["scope"] = np.array("Midspan xy only; DC removed; paired-record alignment inferred from matching job defaults, not embedded runtime timestamps.")
    cache_save(target, fields, fingerprint)
    data["instant"] = fields
    return data


def display_indices(n):
    return np.unique(np.r_[np.arange(min(5, n)), np.arange(0, n, PLOT_STRIDE), n - 1])


def plot_maps(cases, columns, stem, title, instant=False):
    cmap = plt.get_cmap("PuOr_r").copy()
    limit = 0.0
    for d in cases:
        for term, band, label in columns:
            for name in d["names"]:
                q = d.get("instant", {}).get(field_key(term, band, name)) if instant else get_field(d, term, band, name)
                if q is not None:
                    limit = max(limit, float(np.max(np.abs(q))) / abs(d["instant_P_ref"] if instant else d["P_ref"]))
    limit = limit if limit > 0 else LINEAR_LIMIT
    linear_core = min(LINEAR_LIMIT, limit * 0.01)  # Adapt display scale to entry contributions.
    norm = SymLogNorm(linthresh=linear_core, vmin=-limit, vmax=limit, base=10)
    row_height = 0.90 if len(columns) == 4 else 1.25
    size = (7.5, row_height * len(cases) + 0.7)
    fig, axes = plt.subplots(len(cases), len(columns), figsize=size, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.025, hspace=0.025, wspace=0.015)
    for row, d in enumerate(cases):
        pitch = float(d["topo"]["pitch"])
        wall = d["grids"]["ring"]
        for col, (term, band, label) in enumerate(columns):
            ax = axes[row, col]
            any_data = False
            for name, g in d["grids"].items():
                q = d.get("instant", {}).get(field_key(term, band, name)) if instant else get_field(d, term, band, name)
                if q is None:
                    continue
                any_data = True
                ii, jj = display_indices(q.shape[0]), display_indices(q.shape[1])
                sub = np.ix_(ii, jj)
                for shift in [-pitch, 0.0, pitch]:
                    yy = g["y"][sub] + shift
                    if yy.max() < Y_LIM[0] or yy.min() > Y_LIM[1]:
                        continue
                    ax.pcolormesh(g["x"][sub], yy, q[sub] / (d["instant_P_ref"] if instant else d["P_ref"]),
                                  cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
            if not any_data:
                ax.set_facecolor("0.85")
                ax.text(0.5, 0.5, "Not available", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8)
            for shift in [-pitch, 0.0, pitch]:
                ax.fill(wall["x"][:, 0], wall["y"][:, 0] + shift, "white", ec="0.15", lw=0.3)
                if not instant:
                    for polygon in d["polygons"]:
                        ax.plot(polygon[:, 0], polygon[:, 1] + shift, "--", color="0.2", lw=0.3, alpha=0.6)
            ax.set(xlim=X_LIM, ylim=Y_LIM)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=7, length=2, pad=1)
            if row == 0:
                ax.set_title(label, fontsize=9)
            if row == len(cases) - 1:
                ax.set_xlabel(r"$x/c_{\mathrm{ax}}$", fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(f"{d['case'][4:]} (i={d['incidence']:+d})\n" + r"$y/c_{\mathrm{ax}}$", fontsize=9)
            else:
                ax.tick_params(labelleft=False)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    ticks = [0.0] + [sign * 10.0**power
                      for power in range(int(np.floor(np.log10(linear_core))), int(np.ceil(np.log10(limit))) + 1)
                      for sign in [-1, 1] if linear_core <= 10.0**power <= limit]
    cbar = fig.colorbar(scalar, ax=axes, fraction=0.025, pad=0.015, shrink=0.94, ticks=sorted(ticks))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=7, length=2, pad=2)
    cbar.set_label(r"$\mathcal{Q}^{\star}/P_\Sigma^\star$", fontsize=10, labelpad=5)
    fig.suptitle(title, fontsize=10)
    footer = ("Local integrated entry contribution in chord coordinates; shared symmetric-log scale. "
              "Periodic images shown; full-grid integrals use one passage.")
    if instant:
        footer = ("Midspan xy contributions only; not the full 3-D terms. DC removed; "
                  f"selected-record snapshot q={SNAPSHOT_INDEX}; cases are not phase aligned.")
    fig.supxlabel(footer, fontsize=6.5)
    for ext in ["png", "pdf"]:
        stream = BytesIO()
        format_figure(fig)
        fig.savefig(stream, format=ext, dpi=DPI, bbox_inches="tight", pad_inches=0.06)
        atomic_bytes(OUT_DIR / f"{stem}.{ext}", stream.getvalue())
    plt.close(fig)
    print("Saved", stem, flush=True)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spatial_checks(data, rows):
    # Compare absolute magnitudes fairly and check artificial chunk interfaces.
    gm = loadmat(FOURIER_BASE / data["case"] / "grid/grid_split.mat",
                 squeeze_me=True, struct_as_record=False)["grid"]
    blocks = {}
    for b, count in NBCHUNK.items():
        edge = np.zeros(np.asarray(gm[b - 1].x).shape, dtype=bool)
        width = edge.shape[0] // count
        for cut in range(width, edge.shape[0], width):
            edge[cut - 1:cut + 1] = True  # Both adjacent node rows; excludes true block outer edges.
        blocks[b] = edge
    edges = to_local_2d(blocks, data)
    area = sum(g["w"].sum() for g in data["grids"].values())
    edge_area = sum(g["w"][edges[name]].sum() for name, g in data["grids"].items())
    absolute_P = next(r["absolute_spatial_integral"] for r in rows
                      if r["case"] == data["case"] and r["term"] == "P"
                      and r["band"] == "all" and r["region"] == "all")
    result = []
    for term in ["P", "Pi", "Dp", "Pi-Dp"]:
        parts = np.zeros(4)
        edge_abs = edge_signed = total_abs = 0.0
        for name, g in data["grids"].items():
            q = get_field(data, term, "all", name)
            if q is None:
                break
            v = q * g["w"]
            a = np.abs(v)
            edge_abs += a[edges[name]].sum()
            edge_signed += v[edges[name]].sum()
            total_abs += a.sum()
            x = g["x"]
            for k, mask in enumerate([x < 0, (x >= 0) & (x < 0.1),
                                      (x >= 0.1) & (x < 1), x >= 1]):
                parts[k] += a[mask].sum()
        else:
            result.append({"case": data["case"], "term": term,
                           "absolute_Q_over_absolute_P": total_abs / absolute_P,
                           "internal_chunk_edge_entry_fraction": edge_area / area,
                           "absolute_Q_fraction_on_chunk_edges": edge_abs / total_abs,
                           "signed_chunk_edge_Q_over_P_total": edge_signed / data["P_ref"],
                           **dict(zip(["absolute_fraction_x_below_0", "absolute_fraction_x_0_to_0p1",
                                       "absolute_fraction_x_0p1_to_1", "absolute_fraction_x_above_1"],
                                      parts / total_abs))})
    return result


def main():
    OUT_DIR.mkdir(exist_ok=True)
    datasets = [aggregate_case(case) for case in CASES]
    rows = [statistics(d, term, band, region) for d in datasets
            for term in ["P", "Pi", "Dp", "Pi-Dp"] for band in BANDS for region in REGIONS]
    rows = [r for r in rows if r is not None]
    write_csv(OUT_DIR / "pressure_integrals.csv", rows)
    checks = [r for d in datasets for r in spatial_checks(d, rows)]
    write_csv(OUT_DIR / "spatial_diagnostics.csv", checks)
    plot_maps(datasets, [("P", "all", "P"), ("Pi", "all", r"$\Pi$"),
                         ("Dp", "all", r"$D_p$"), ("Pi-Dp", "all", r"$\Pi-D_p$")],
              "pressure_all_frequencies", "Production and pressure terms: all resolved nonzero frequencies")
    labels = ["BP family", "Low frequency", "Broadband"]
    for term, symbol in [("Pi", r"$\Pi$"), ("Dp", r"$D_p$")]:
        plot_maps(datasets, [(term, band, label) for band, label in zip(BANDS[:3], labels)],
                  f"{term}_bands", symbol + ": span-summed frequency families")
    inst = [instantaneous_case(d) for d in datasets] if RECONSTRUCT_INSTANT else []
    inst = [d for d in inst if d is not None]
    if inst:
        plot_maps(inst, [("Pi_xy", "instant", r"$\pi_{xy}$"),
                         ("Dp_xy", "instant", r"$d_{p,xy}$"),
                         ("p_div_xy", "instant", r"$p'\theta'_{xy}$"),
                         ("product_defect_xy", "instant", r"$\mathcal{E}_{xy}$")],
                  "instantaneous_pressure_xy", "Illustrative instantaneous pressure terms: midspan xy only",
                  instant=True)
        instant_rows = []
        for d in inst:
            temp = dict(d, fields=d["instant"], P_ref=d["instant_P_ref"],
                        grids={name: dict(g, w=g["area_w"]) for name, g in d["grids"].items()})
            for term in ["Pi_xy", "Dp_xy", "p_div_xy", "product_defect_xy"]:
                row = statistics(temp, term, "instant")
                instant_rows.append(row)
        write_csv(OUT_DIR / "instantaneous_xy_integrals.csv", instant_rows)
    lines = [
        "F7 pressure-field analysis",
        "P, Pi, Dp: supplied integrated contributions; maps sum all nonzero FFT bins or the exact F1_compare_cases bands.",
        "Pi is complete pressure-gradient work. Dp is pressure transport, already contained in Pi.",
        "Pi-Dp is a continuum pressure-dilatation diagnostic plus a discrete product-rule discrepancy; it is not separately proven physical dilatation.",
        "All maps divide each case's local rate by its signed all-region/all-nonzero-frequency P total.",
        "Colour is integrated entry contribution in nondimensional chord area, not a pointwise Pi/P ratio.",
        "The sum of displayed entry ratios equals the corresponding Q/P_total ratio over one passage.",
        "Geometry and masks: For_Theo/Mean_data_3DNS/<case>/pp/grid and pp/masks.",
        f"Only displayed grids are thinned (stride {PLOT_STRIDE}); integrations and validations use every weighted node.",
        "Periodic copies in the figures are visual only. Full supplied mesh used for integrals.",
        "52deg: P available; pressure fields/modes unavailable. Its frequency spacing and range differ from the other cases.",
        "",
        "Instantaneous reconstruction",
        "Raw modes_p/u/v in STATS_FOURIER are midspan Fourier modes with a forward 1/N factor.",
        "Reconstruction uses sum_n mode[n] exp(2 pi i n q/N); no extra 1/N factor. The DC coefficient is removed.",
        "Thus reconstructed fluctuations are relative to the selected record's temporal mean, not necessarily the original database mean.",
        "Pi_xy=-u' dp'/dx-v' dp'/dy; Dp_xy=-d(p'u')/dx-d(p'v')/dy.",
        "p_div_xy=p'(du'/dx+dv'/dy) is only the in-plane part, not full 3-D pressure dilatation.",
        "product_defect_xy=Pi_xy-Dp_xy-p_div_xy; it measures the product-rule discrepancy of these xy finite differences.",
        "Original block/chunk boundaries and HPC x/y metric/data-gradient edge orders are retained.",
        "Out-of-plane derivatives and the spanwise average cannot be recovered from a single midspan plane.",
        "Matching N and current spec/dp FIRST defaults are checked; original runtime alignment is not embedded in the NPY files.",
        "Snapshots across cases are not phase aligned. A single instant does not establish statistical causality.",
        "",
        "Other existing 2-D snapshots",
        "For_Theo/Mean_data_3DNS/snapshot_cuts/cut_44deg.npz, cut_52deg.npz, cut_60deg.npz",
        "These contain x,y,u,v,wz per original block and frame/counter identifiers; no pressure, density or temperature.",
        "Their generator defaults to total velocity/vorticity, but the mode is not stored in the NPZ; their counters are not matched to this reconstruction.",
        "",
        "All-region / all-frequency summary; ratios relative to case-total production:",
    ]
    for d in datasets:
        for term in ["Pi", "Dp"]:
            row = next((r for r in rows if r["case"] == d["case"] and r["term"] == term
                        and r["band"] == "all" and r["region"] == "all"), None)
            if row:
                lines.append(f"{d['case']} {term}: signed/P={row['signed_over_P_total']:+.8g}; "
                             f"integral(abs(sum_frequency Q))/P={row['absolute_over_P_total']:.8g}; "
                             f"spatial cancellation={row['spatial_cancellation']:.5f}")
                local = [r for r in rows if r["case"] == d["case"] and r["term"] == term
                         and r["band"] == "all" and r["region"] != "all"]
                dominant = max(local, key=lambda r: r["absolute_spatial_integral"])
                lines.append(f"  Largest absolute regional contribution: {dominant['region']}; "
                             f"strongest |local field| at x={row['peak_x_cax']:.5g}, "
                             f"y={row['peak_y_cax']:.5g}, array={row['peak_array']}.")
    lines += ["", "Additional checks (spatial_diagnostics.csv):",
              "Absolute-Q / absolute-P compares spatial L1 magnitudes of the frequency-summed fields; it is not an energy fraction.",
              "The passage mask is a geometric remainder and includes leading-edge points outside the BL masks.",
              "Chunk-edge strips contain both node rows adjacent to internal i-chunk cuts; they do not test every block boundary or wall."]
    for r in checks:
        if r["term"] in ["Pi", "Dp", "Pi-Dp"]:
            lines.append(f"{r['case']} {r['term']}: absolute-Q/absolute-P={r['absolute_Q_over_absolute_P']:.6g}; "
                         f"internal chunk-edge entry fraction={r['internal_chunk_edge_entry_fraction']:.6g}; "
                         f"fraction of absolute-Q on those edges={r['absolute_Q_fraction_on_chunk_edges']:.6g}.")
    if inst:
        lines += ["", "Instantaneous finite-difference check: not a test of the full 3-D budget."]
        for d in inst:
            rr = {r["term"]: r for r in instant_rows if r["case"] == d["case"]}
            ratio = rr["product_defect_xy"]["absolute_spatial_integral"] / rr["Pi_xy"]["absolute_spatial_integral"]
            lines.append(f"{d['case']}: integral(abs(product_defect_xy))/integral(abs(Pi_xy))={ratio:.6g}.")
    (OUT_DIR / "pressure_checks_and_interpretation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-18:]), flush=True)
    print("Outputs:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
