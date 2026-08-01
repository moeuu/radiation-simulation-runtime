# Shared-runtime exact free-flight decision for standard observations

## Decision

Standard full-simulation observations continue to use Geant4 analog,
unit-weight transport. The calibration-only `G4BOptrForceCollision` route is
not selectable by the standard runtime. No "exact free-flight" switch is
provided for actual observations.

This is a fail-closed decision, not a claim that conditional transport is
mathematically impossible. The current sidecar cannot obtain the required
conditional law before transport without either:

1. transporting both weighted force-collision branches, which does not reduce
   work; or
2. reimplementing Geant4 geometry navigation, interaction-length bookkeeping,
   and process arbitration outside the trusted analog kernel.

Neither option is an accuracy-neutral standard-runtime acceleration.

## Exact unit-weight mixture requirement

For a stable photon moving toward a fixed boundary, define

\[
\tau = \int_0^L \Sigma_{\mathrm{tot}}(E,\mathbf{x}(s))\,ds,\qquad
q = 1-\exp(-\tau).
\]

If \(B\) is the event that the first interaction occurs before the boundary,
the analog path law decomposes as

\[
\mathcal{P}
= (1-q)\,\mathcal{P}(\cdot\mid B^c)
+ q\,\mathcal{P}(\cdot\mid B).
\]

A unit-weight Bernoulli implementation is exact only if all of the following
hold:

- the Bernoulli probability is the exact \(q\) for the complete material path;
- the collision distance is drawn from the exact truncated optical-depth law;
- the winning process is drawn with probability
  \(\Sigma_j/\Sigma_{\mathrm{tot}}\) at that distance;
- the no-collision state preserves position, direction, energy, polarization,
  global time, touchable/navigation state, and boundary semantics;
- both branches resume ordinary Geant4 transport with all secondaries;
- no weighted tally, branch weight, or expected count reaches an observation.

The identity above does not make an approximate optical depth, a single
effective material, or a detector-directed geometric shortcut exact.

## Why stock force collision is not a unit-weight speed path

Geant4 documents `G4BOptrForceCollision` as a split scheme. One copy is forced
to collide and another performs a "silent" free flight at zero weight. The
free-flight weight is restored at the volume boundary after accumulating the
non-interaction probability from the wrapped physics processes.

That implementation is appropriate for the fixed-quota mean estimator because
the sidecar clusters all weighted descendants of an original history. It does
not provide the branch probability before the silent flight has traversed the
volume.

Selecting one branch after both branches finish can reproduce a unit-weight
mixture only after buffering complete branch histories, and therefore performs
at least the work of both branches. Selecting at entry would require a separate
exact optical-depth traversal first. Across heterogeneous and sequential
obstacle/shield leaves, that traversal duplicates the work and correctness
contracts already implemented by Geant4's analog stepping manager.

Geant4's analog photon transport already samples physical interaction lengths,
limits steps at geometry boundaries, selects the competing process, and then
performs the full secondary transport. Re-expressing that same sampling as a
Bernoulli mixture is not by itself a speedup or variance reduction.

Primary references:

- [Geant4 event-biasing documentation](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Fundamentals/biasing.html)
- Installed Geant4 11.3.2 headers:
  `G4BOptrForceCollision.hh`, `G4BOptnForceFreeFlight.hh`, and
  `G4BOptnForceCommonTruncatedExp.hh`

## Reconsideration gate

A future implementation may enter the standard runtime only after it:

1. defines the exact pre-boundary domain for arbitrary authored scenes;
2. derives all probabilities from the active Geant4 physics processes and
   current material segments;
3. preserves every transported state variable and all secondary transport;
4. emits integer, unit-weight observations only;
5. matches analog entry-class and energy-bin distributions at 1, 16, and
   32 threads on independent seeds;
6. passes rare-scatter and boundary-adjacent cases without fitted correction;
7. demonstrates a repeatable wall-time reduction after including its optical
   depth and navigation overhead.

Until all seven conditions hold, an attempted standard-runtime flag must be
rejected as unsupported.

## Accuracy-neutral acceleration retained

Persistent Geant4 sessions now update detector and shield physical placements
between runs while retaining the same scene and physics tables. On the small
real-Geant4 regression scene, two moved-pose observations measured:

- 1 thread: 2.14 s with two fresh processes versus 1.09 s persistent;
- 32 threads: 2.34 s with two fresh processes versus 1.19 s persistent.

This optimization changes initialization schedule only. A cached second pose
matches a fresh process exactly in serial and statistically at 16/32 threads.
Later poses are validated against the fixed world bounds before any transport.
