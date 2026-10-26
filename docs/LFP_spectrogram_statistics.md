# GC/PC and within-route spectrogram comparisons

Run `compare_LFP_spectrograms.py` after channel validation and artifact removal.
It reuses `grand_average_LFP.collect_session`, including the same approved
channels, cleaned trials, STFT, trial/channel/frequency baseline z-scoring,
and median over trials and channels. It reads session data, not PNG images.
No new dependencies beyond NumPy, SciPy, Matplotlib and the existing LFP pipeline.

## Ten figures

| Output stem | Conditions | Pairwise tests |
| --- | --- | --- |
| `GC_vs_PC_BA` | GC BA, PC BA | GC − PC |
| `GC_vs_PC_IA` | GC IA, PC IA | GC − PC |
| `GC_vs_PC_S` | GC S, PC S | GC − PC |
| `GC_vs_PC_BK` | GC BK, PC BK | GC − PC |
| `GC_vs_PC_oIA` | GC oIA, PC oIA | GC − PC |
| `GC_vs_PC_oBA` | GC oBA, PC oBA | GC − PC |
| `GC_retro` | BA, IA, S in GC | BA − IA, BA − S, IA − S |
| `GC_ortho` | BK, oIA, oBA in GC | BK − oIA, BK − oBA, oIA − oBA |
| `PC_retro` | BA, IA, S in PC | BA − IA, BA − S, IA − S |
| `PC_ortho` | BK, oIA, oBA in PC | BK − oIA, BK − oBA, oIA − oBA |

These are **18 pairwise tests in 10 figures**. The four within-region figures
compare stimuli *within* a route, as requested; they are not direct matched-odor
ortho-versus-retro tests. S and BK are included in the specified groups without
assuming they are a chemically matched pair. GC/PC region names are case insensitive. For the confirmed legacy convention,
`waterIA`/`waterBA` become retro `IA`/`BA`, and accompanying plain `IA`/`BA` become
ortho `oIA`/`oBA`. Mapping is simultaneous and only activates for an odor when
its `water` label is present. Already canonical sessions are unchanged. Distinct
source names that would collide cause an error instead of pooling routes.
This translation applies to grand averages, individual-session plots, and
statistical comparisons; source metadata and cleaned trials are unchanged.
The statistical audit stores original-to-canonical names for each session.

## Run

Activate the same conda environment used for the existing LFP pipeline. From
this repository, use your existing session-list text file (one absolute session
directory per line). For example:

```bash
cd "$HOME/julia_blech_clust_testing"
DATA_LIST="/absolute/path/to/your/session_list.txt"

# If these steps have already completed for these sessions, skip them.
python validate_LFP_data.py --data-list "$DATA_LIST"
python remove_lfp_artifacts.py --data-list "$DATA_LIST"

python compare_LFP_spectrograms.py \
  --data-list "$DATA_LIST" \
  --output-dir "results/spectrogram_statistics_$(date +%Y%m%d_%H%M%S)" \
  --permutations 10000 \
  --seed 0
```

Alternatively repeat `--dataset /absolute/session/path` to select sessions.
Session basenames must be unique. The full input path is searched for animal IDs.
Thus `/media/thomas/Data/TG25/D1_retroortho2_260311_155032` identifies **TG25**
even though its basename has no animal ID. Names such as
`Feb2_OrthoRetro_TG22_260202_170014` identify animal **TG22**, so all TG22 sessions
are averaged within that animal. Verify the resolved mapping in `statistics.json`.
Conflicting parent/basename animal IDs cause an error. For other naming schemes,
or to override an ID, use `--animal-map animals.json` (keys may be session
basenames or full input paths):

```json
{
  "session_one": "rat_1",
  "session_two": "rat_1",
  "session_three": "rat_2"
}
```

Unresolvable animal IDs cause an error, not an assumption that sessions are
independent animals. Use a new/empty output directory on each run.

## Analysis and correction

1. For each figure, retain only sessions containing **all** its conditions.
   The three-condition figures use the same complete-session set for all three
   pairwise comparisons. Save excluded unmatched sessions in the report.
2. Average each condition over retained sessions within each animal. Compute
   paired condition differences per animal. Animals receive equal weight;
   channels, trials, and repeated sessions are not independent replicates.
   Thus these means can differ from the original session-weighted grand averages.
3. Compute a paired t statistic at each frequency/time bin. The default test
   window is **0–1.75 s**, at available positive frequencies up to **35 Hz**.
   Display −1–1.75 s for context. DC is excluded. `--test-start` and `--test-end`
   change the test window; choose it before viewing results. The repository's
   500 ms STFT window limits temporal resolution and smears stimulus onset.
4. Form separate positive and negative clusters using two-sided uncorrected
   p < .05 (t threshold, df = animals − 1) and four-neighbor frequency/time
   adjacency. Cluster mass is the sum of absolute t values. This follows the
   cluster permutation framework of Maris & Oostenveld (2007) and FieldTrip's
   time-frequency tutorial [1, 2]; this script is an independent implementation.
5. Sign-flip each animal's entire difference map. This preserves dependence
   among bins and requires independent animals and sign-exchangeable differences
   under the null (a symmetric difference distribution). For each permutation,
   take the largest mass across both signs and the whole tested grid. Enumerate
   all 2^(N−1) unique two-sided patterns when this is at most `--permutations`;
   otherwise use seeded Monte Carlo signs with (exceedances + 1)/(draws + 1).
   Exact p values include the observed pattern, use exceedances / patterns, and
   count ties. Zero differences produce no clusters. Near-zero standard errors
   are floored at machine epsilon times max(1, max absolute difference).
6. Multiply cluster-corrected p values by **18** (Bonferroni, capped at 1) to
   control the family of planned contrasts, including when some are unavailable.
   Hatching marks clusters with final p ≤ .05 (`--alpha` changes this threshold).
   The cluster-forming threshold stays .05. This conservative correction applies
   to both tails, the tested time-frequency grid, and the 18 comparisons.

**Small-sample limitation:** six animals permit only 32 unique two-sided sign
patterns. The smallest cluster p is 1/32 = .03125; after the 18-test correction,
its minimum is .5625. Therefore six animals cannot yield a significant result
under this full-family procedure, regardless of effect size or number of trials.
At least ten animals are needed even to make p ≤ .05 attainable (not guaranteed).
More Monte Carlo draws cannot fix the finite number of unique permutations.
The script reports attainable resolution, and stores both cluster and final
family-corrected p values. Do not relabel uncorrected results as significant
because the corrected results are empty; consider a separately prespecified,
smaller scientific hypothesis family if appropriate.

These tests concern **baseline-normalized power responses**, not absolute LFP
power, coherence, or firing rates. Baseline z-scoring can amplify low-variance
baselines; inspect channel QC and per-session plots before interpreting results.
A significant cluster supports a condition difference over the tested domain;
it does not establish exact onset, offset, or significance of each individual
bin in its outline [3]. No significant cluster is not evidence of equivalence.

## Outputs and audit

- Ten PNGs and ten SVGs when every figure has at least two complete animals.
  Regional figures show two condition maps and their difference. Within-route
  figures show three condition maps above three pairwise difference maps.
  Condition maps share a symmetric color scale within a figure; differences
  share their own symmetric scale. Red means positive, blue negative.
- `statistics.json`: settings, resolved animal IDs, retained sessions per animal,
  exclusions, per-session approved channel counts and label mappings,
  cluster masses/directions/p values, permutation counts, exact versus
  Monte Carlo status, and smallest attainable family p for each comparison.
- Eighteen NPZs: paired animal differences, animal IDs, axes, test mask, t maps,
  family-corrected cluster p maps, full-display significance masks, and null maxima.
  `t` and `p_family` cover only tested time bins; `animal_difference` and
  `significant` cover the displayed window. Bins outside clusters have p = 1.

Sessions with no approved channels, or with all tastes excluded by artifact
removal, are skipped with an explicit reason in `statistics.json/session_audit`.
Their channels are not reinstated and thresholds are not relaxed. Other load/QC
errors and incompatible STFT axes stop the run and record the failure. Inspect
`LFP_channel_check/lfp_validation_report.json` in an excluded session to review
its channel rejection reasons. If every session is excluded, the run stops
with the audit saved. Grand averages also skip sessions with no approved channels
before computing their STFT. Fewer than two complete animals produces an
explicit untested entry, no inferential figure for that group, and a nonzero exit
status after processing other groups. Inspect the report before using outputs.
The code's synthetic tests do not validate biological results; actual session
files must be run locally to establish which comparisons are significant.

## References

1. Maris E, Oostenveld R (2007). Nonparametric statistical testing of EEG- and
   MEG-data. *Journal of Neuroscience Methods*, 164(1), 177–190.
   https://doi.org/10.1016/j.jneumeth.2007.03.024
2. FieldTrip. Cluster-based permutation tests on time-frequency data.
   https://www.fieldtriptoolbox.org/tutorial/stats/cluster_permutation_freq/
3. Sassenhagen J, Draschkow D (2019). Cluster-based permutation tests of MEG/EEG
   data do not establish significance of effect latency or location.
   *Psychophysiology*, 56, e13335. https://doi.org/10.1111/psyp.13335
