#!/usr/bin/env python3
"""Animal-level paired cluster permutation tests of cleaned LFP spectrograms."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats
from lfp_session_labels import animal_for_session

ROUTES = {"retro": ("BA", "IA", "S"), "ortho": ("BK", "oIA", "oBA")}
N_CONTRASTS = 18  # six regional + four groups of three pairwise contrasts


def figure_specs():
    specs = []
    for tastes in ROUTES.values():
        for taste in tastes:
            specs.append((f"GC_vs_PC_{taste}", [("GC", taste), ("PC", taste)]))
    for region in ("GC", "PC"):
        for route, tastes in ROUTES.items():
            specs.append((f"{region}_{route}", [(region, t) for t in tastes]))
    return specs


def paired_animals(region_data, conditions, mapping):
    """Complete sessions for ALL conditions in a figure; equal-weight animals."""
    lookup = {}
    for (region, taste), values in region_data.items():
        key = (region.upper(), taste)
        if key in lookup:
            raise ValueError(f"Ambiguous region/stimulus key: {key}")
        sessions = values["sessions"]
        if len(sessions) != len(set(sessions)):
            raise ValueError(f"Duplicate sessions for {key}")
        lookup[key] = dict(zip(sessions, values["spectrograms"]))
    session_sets = [set(lookup.get(c, {})) for c in conditions]
    common = sorted(set.intersection(*session_sets))
    excluded = sorted(set.union(*session_sets) - set(common))
    grouped = {}
    for session in common:
        grouped.setdefault(animal_for_session(session, mapping), []).append(session)
    animals = sorted(grouped)
    arrays = [np.asarray([np.mean([lookup[c][s] for s in grouped[a]], axis=0)
                          for a in animals]) for c in conditions]
    return arrays, animals, grouped, excluded


def t_map(difference):
    mean = difference.mean(axis=0)
    sem = difference.std(axis=0, ddof=1) / np.sqrt(len(difference))
    # A deterministic regularizer keeps constant nonzero effects finite, while
    # exactly zero differences remain zero, including during sign flips.
    floor = np.finfo(float).eps * max(1., float(np.max(np.abs(difference))))
    return mean / np.maximum(sem, floor)


def clusters(t_values, threshold):
    """Separate positive/negative clusters, four-neighbor time/frequency grid."""
    found = []
    for sign in (1, -1):
        labels, count = ndimage.label(sign * t_values > threshold)
        for label in range(1, count + 1):
            mask = labels == label
            found.append((mask, float(np.abs(t_values[mask]).sum()), sign))
    return found


def cluster_test(difference, permutations=10000, seed=0, alpha=.05):
    difference = np.asarray(difference, dtype=float)
    if difference.ndim != 3 or len(difference) < 2 or not np.isfinite(difference).all():
        raise ValueError("Need finite animal x frequency x time data from >=2 animals")
    if permutations < 1 or not 0 < alpha < 1:
        raise ValueError("permutations must be positive and alpha between 0 and 1")
    n = len(difference)
    threshold = float(stats.t.ppf(.975, n - 1))
    observed_t = t_map(difference)
    observed = clusters(observed_t, threshold)
    # For a two-sided maximum, global sign reversal is equivalent. Fix the
    # first sign positive and enumerate the 2**(n-1) unique permutations.
    exact = 2 ** (n - 1) <= permutations
    rng = np.random.default_rng(seed)
    signs = (itertools.product((-1, 1), repeat=n - 1) if exact else
             (rng.choice((-1, 1), n - 1) for _ in range(permutations)))
    null = []
    for tail in signs:
        flipped = difference * np.array((1, *tail))[:, None, None]
        null.append(max((mass for _, mass, _ in clusters(t_map(flipped), threshold)),
                        default=0.))
    null = np.asarray(null)
    p_map = np.ones(difference.shape[1:])
    records = []
    for mask, mass, sign in observed:
        exceed = np.count_nonzero(null >= mass * (1 - 1e-12))
        p = exceed / len(null) if exact else (exceed + 1) / (len(null) + 1)
        adjusted = min(1., p * N_CONTRASTS)
        p_map[mask] = adjusted
        records.append({"mass": mass, "sign": sign, "bins": int(mask.sum()),
                        "p_cluster": p, "p_family": adjusted})
    return {"t": observed_t, "p_family": p_map, "significant": p_map <= alpha,
            "null_max_mass": null, "clusters": records, "exact": exact,
            "permutations": len(null), "threshold": threshold,
            "minimum_family_p": min(1., N_CONTRASTS / (len(null) if exact else len(null)+1))}


def make_figure(name, conditions, arrays, animals, freq, time, test_mask,
                output, args):
    pairs = list(itertools.combinations(range(len(conditions)), 2))
    fig, axes = plt.subplots(2 if len(conditions) == 3 else 1, 3,
                             figsize=(16, 8 if len(conditions) == 3 else 4.7),
                             constrained_layout=True, squeeze=False)
    means = [a.mean(axis=0) for a in arrays]
    scale = max(float(np.max(np.abs(means))), 1e-12)
    for ax, mean, condition in zip(axes.flat, means, conditions):
        im = ax.pcolormesh(time, freq, mean, cmap="RdBu_r", vmin=-scale, vmax=scale,
                           shading="auto", rasterized=True)
        ax.set_title(" ".join(condition))
        fig.colorbar(im, ax=ax, label="Baseline z-scored power")
    diff_axes = list(axes.flat)[len(conditions):]
    diff_scale = max(max(float(np.max(np.abs(means[i]-means[j]))) for i,j in pairs), 1e-12)
    results = []
    for index, ((i, j), ax) in enumerate(zip(pairs, diff_axes)):
        label = f"{' '.join(conditions[i])} − {' '.join(conditions[j])}"
        difference = arrays[i] - arrays[j]
        result = cluster_test(difference[..., test_mask], args.permutations,
                              args.seed + index, args.alpha)
        significant = np.zeros(difference.shape[1:], dtype=bool)
        significant[:, test_mask] = result["significant"]
        im = ax.pcolormesh(time, freq, difference.mean(axis=0), cmap="RdBu_r",
                           vmin=-diff_scale, vmax=diff_scale, shading="auto", rasterized=True)
        if significant.any():
            ax.contourf(time, freq, significant.astype(float), levels=[.5, 1.5],
                        colors="none", hatches=["////"])
        count = sum(c["p_family"] <= args.alpha for c in result["clusters"])
        ax.set_title(f"{label}\n{count} significant cluster(s)")
        fig.colorbar(im, ax=ax, label="Difference in baseline z-score")
        np.savez_compressed(output / f"{name}_contrast_{index+1}.npz",
                            animals=np.array(animals), animal_difference=difference,
                            frequency_hz=freq, time_s=time, test_mask=test_mask,
                            t=result["t"], p_family=result["p_family"],
                            significant=significant, null_max_mass=result["null_max_mass"])
        results.append({"contrast": label, **{k: v for k,v in result.items()
                        if k not in ("t", "p_family", "significant", "null_max_mass")}})
    for ax in axes.flat:
        ax.axvline(0, color="black", linestyle="--", linewidth=.8)
        ax.set(xlabel="Time from stimulus (s)", ylabel="Frequency (Hz)")
    fig.suptitle(f"{name.replace('_', ' ')} | {len(animals)} animals\n"
                 f"Hatching: cluster p ≤ {args.alpha:g}, corrected across 18 contrasts")
    fig.savefig(output / f"{name}.png", dpi=200)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-list")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--animal-map", help="JSON mapping session basename to animal ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--test-start", type=float, default=0.)
    parser.add_argument("--test-end", type=float, default=1.75)
    args = parser.parse_args()
    if args.permutations < 1 or not 0 < args.alpha < 1 or args.test_start >= args.test_end:
        parser.error("Check permutations, alpha, and test window")
    # Lazy imports keep the statistical engine testable without spike-sorting dependencies.
    from grand_average_LFP import collect_session
    from lfp_common import (get_dataset_dirs, DEFAULT_DATA_LIST,
                            PLOT_MAX_FREQ_HZ, SPECTROGRAM_START_MS, SPECTROGRAM_END_MS)
    paths = [str(Path(p).expanduser()) for p in
             get_dataset_dirs(args.dataset, args.data_list or DEFAULT_DATA_LIST)]
    names = [Path(p).name for p in paths]
    if len(names) != len(set(names)):
        parser.error("Session basenames must be unique; remove duplicate datasets")
    mapping = json.loads(Path(args.animal_map).read_text()) if args.animal_map else {}
    animal_map = {s: animal_for_session(path, mapping) for s, path in zip(names, paths)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Use a new/empty output directory to avoid stale results")
    reference = {"freq": None, "relative_time": None}
    region_data = {}
    report = {"settings": vars(args), "animal_map": animal_map, "figures": {},
              "family_size": N_CONTRASTS, "unit": "animal", "excluded_dc": True,
              "dataset_paths": paths, "session_audit": {}}
    for path, name in zip(paths, names):
        session_data = {}
        audit = {"data_dir": path, "animal": animal_map[name]}
        report["session_audit"][name] = audit
        try:
            collect_session(path, session_data, reference, audit=audit)
        except Exception as error:
            audit.update(status="failed", reason=str(error))
            (output / "statistics.json").write_text(json.dumps(report, indent=2))
            raise
        (output / "statistics.json").write_text(json.dumps(report, indent=2))
        if not session_data:
            if audit.get("status") == "excluded":
                print(f"Excluded {name}: {audit['reason']}", flush=True)
                continue
            raise ValueError(f"No usable summaries or documented exclusion for {name}")
        for key, entry in session_data.items():
            target = region_data.setdefault(key, {k: [] for k in entry})
            for field, values in entry.items():
                target[field].extend(values)
    if not region_data:
        raise SystemExit("No sessions passed QC; inspect statistics.json/session_audit")
    freq, time = reference["freq"], reference["relative_time"]
    fm = (freq > 0) & (freq <= PLOT_MAX_FREQ_HZ)
    tm = (time >= SPECTROGRAM_START_MS/1000 - 1e-9) & (time <= SPECTROGRAM_END_MS/1000 + 1e-9)
    freq, time = freq[fm], time[tm]
    test_mask = (time >= args.test_start - 1e-9) & (time <= args.test_end + 1e-9)
    if len(freq) < 2 or test_mask.sum() < 2 or args.test_start < time[0]-1e-9 or args.test_end > time[-1]+1e-9:
        raise ValueError("Requested test window must fit the displayed STFT grid")
    for name, conditions in figure_specs():
        print(f"Comparing {name}", flush=True)
        arrays, animals, sessions, excluded = paired_animals(region_data, conditions, animal_map)
        info = {"animals": animals, "sessions_by_animal": sessions,
                "excluded_unmatched_sessions": excluded}
        if len(animals) < 2:
            info["status"] = "not tested: fewer than two complete animals"
            print(f"  {info['status']}", flush=True)
        else:
            arrays = [a[:, fm][:, :, tm] for a in arrays]
            if not all(np.isfinite(a).all() for a in arrays):
                raise ValueError(f"Nonfinite data in {name}; review normalization/QC")
            info["status"] = "tested"
            info["contrasts"] = make_figure(name, conditions, arrays, animals,
                                             freq, time, test_mask, output, args)
            if info["contrasts"][0]["minimum_family_p"] > args.alpha:
                print("  Sample/permutation count cannot reach family-corrected alpha.")
        report["figures"][name] = info
        (output / "statistics.json").write_text(json.dumps(report, indent=2))
    tested = sum(v["status"] == "tested" for v in report["figures"].values())
    print(f"Saved {tested}/10 figures and statistics.json to {output}")
    if tested < 10:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
