# Isotope experiment profiles

Random-source experiments may select one named isotope profile with the
`isotope_experiment_profile` runtime-config field:

| Profile | Isotopes | Placement of the third isotope |
| --- | --- | --- |
| `ral_eu154` | Cs-137, Co-60, Eu-154 | legacy area-uniform RA-L support |
| `fukushima_eu154` | Cs-137, Co-60, Eu-154 | Co-60/Eu-154 on compatible activated materials |
| `fukushima_eu152` | Cs-137, Co-60, Eu-152 | concrete surfaces only |
| `fukushima_nb94` | Cs-137, Co-60, Nb-94 | steel/iron surfaces only |
| `fukushima_cs134` | Cs-137, Co-60, Cs-134 | surface contamination |
| `fukushima_sb125` | Cs-137, Co-60, Sb-125 | surface contamination |
| `fukushima_am241` | Cs-137, Co-60, Am-241 | fuel-debris-compatible material surfaces |

For example:

```json
{
  "isotope_experiment_profile": "fukushima_eu152"
}
```

The profile and `random_source_isotopes` are mutually exclusive. A material-
conditioned profile samples each source independently and uniformly with
respect to physical area over only that nuclide's eligible materials. The
runtime stops if the environment has no eligible surface.

The full-spectrum model must contain exactly the selected isotope line
identity. The runtime intentionally rejects an Eu-152 or Nb-94 profile paired
with the legacy Eu-154 model; selecting a truth isotope without a matching PF
state would make correct inference impossible.

`configs/geant4/models/isotope_profile_model_registry_v1.json` binds every
profile to one model file, file digest, and model-contract digest. The RA-L
Eu-154 entry includes the designated-training discrepancy model. Other
profiles currently use their matching physical line mean and are explicitly
labelled runtime-unvalidated; selecting them is supported, but does not imply
that an independent Geant4 accuracy holdout has been passed.

## Decay and detector semantics

The evaluated catalog distinguishes two line representations:

- `lines` is the detector-cps-at-1-m transport basis used by the standard PF.
- `decay_lines` records marginal photons per parent decay and may sum to more
  than one for a cascade.

Marginal line probabilities are never sampled as mutually exclusive decay
branches. Exact cascade studies use
`primary_emission_model=geant4_radioactive_decay`, which creates the parent ion
and delegates branching, conversion electrons, daughter levels, and prompt
gamma correlations to the installed Geant4 RadioactiveDecay/ENSDF data. All
energy deposits arriving within the detector model's 1 microsecond
`coincidence_window_s` are combined into one pulse, so prompt true-coincidence
sum peaks can occur. Deposits from delayed daughter decays outside that window
become separate pulses; in particular, the delayed Ba-137m transition is not
summed with the preceding Cs-137 beta decay merely because Geant4 transported
both in one parent event.

Radioactive-decay mode is fail-closed to analog, unit-weight, isotropic, full
secondary and full detector transport. Its source contract is true parent
activity: `activity_bq` parent decays per second and an expected parent count of
`activity_bq * live_time_s`. Parent-decay times are uniform in the acquisition
window. Geant4 supplies daughter lifetimes, and detector deposits outside the
window are rejected rather than folded into the recorded spectrum.

This contract schedules parent decays and does not create a pre-existing
daughter inventory. It is therefore exact for prompt cascades, but it is not a
secular-equilibrium source-age model. For example, a one-second Cs-137 run does
not receive Ba-137m gamma rays produced by Cs decays before the acquisition.
The native metadata records this limitation as
`scheduled_parent_decays_no_preexisting_daughter_inventory`; applications must
not silently interpret it as an aged equilibrium source.

RDM remains a truth/calibration mode. It must not be connected to the
detector-cps PF runtime until a cascade-aware full-spectrum model with the same
source-strength semantics has been supplied. The standard RA-L mode therefore
remains on its authenticated independent-line detector-cps contract.

## Pre-run decay-cascade comparison

Before a new isotope profile is promoted to a standard PF experiment, run the
predeclared distance diagnostic:

```bash
uv run python scripts/build_geant4_sidecar.py --profile native
uv run python scripts/run_decay_cascade_comparison.py
```

The immutable design is
`configs/validation/decay_cascade_comparison_v1.json`. It compares Co-60,
Eu-152, and Eu-154 at 1.5, 2.5, and 4.0 m with the exact detector assembly from
the standard runtime configuration. Independent isotope/distance cases run in
bounded process-parallel workers, and each Geant4 process remains
multithreaded. The declared total thread budget is never exceeded.

The diagnostic deliberately uses an empty-air scene. It isolates evaluated
decay branching, prompt cascades, detector energy deposition, the 1 microsecond
coincidence window, and the authenticated independent-line basis. Environment
and shield transport discrepancies remain the responsibility of the separate
full-spectrum acceptance corpus.

Both arms use unit-weight Geant4 transport. The RDM arm uses parent activity in
Bq and samples isotropic parent decays; the line-basis arm uses the standard
detector-cps source contract. Their absolute source-rate normalizations are
therefore not compared. The reported quantity is the conditional
detector-pulse mark distribution. RDM history
quotas are fixed before acquisition from detector solid angle and evaluated
mean gamma multiplicity, with a hard maximum. A case with too few detected RDM
pulses is reported as `inconclusive`, never as a pass.

The diagnostic spectrum extends to 3400 keV so the 2505.7 keV Co-60 sum peak
and high-energy Eu pair-sum candidates are observable. This wide axis exists
only behind the native `--decay-comparison-diagnostic` contract. Standard PF
observations remain exactly 0--1700 keV with 851 bins.

Each run writes a design hash, sidecar hash, nuclide-catalog hash, raw scene and
response files, spectra, a case CSV, and a review figure. The figure is also
copied under `~/Pictures` with the unique run ID. Each isotope/distance result
is one of:

- `independent_basis_adequate`: all predeclared upper confidence bounds pass;
- `cascade_aware_model_required`: a lower confidence bound exceeds a gate;
- `inconclusive`: the pulse count or statistical separation is insufficient.

A failure does not automatically fit or deploy a correction. It marks that a
cascade-aware model must be trained on declared environments and independently
validated before the standard config may reference it. This prevents a single
diagnostic run from becoming a run-specific response correction.
