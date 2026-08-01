# Calibration-only first-collision core

This directory contains the dependency-free mathematical core for exact
first-collision variance reduction along a path split into homogeneous
material segments. It is deliberately not part of the standard Geant4
sidecar build.

For segment \(r\), with length \(L_r\) and process macroscopic
cross-sections \(\Sigma_{rj}\), the core uses

\[
\tau_r=L_r\sum_j\Sigma_{rj},\qquad
p_{\mathrm{survive},r}=e^{-\tau_r},\qquad
p_{\mathrm{collide},r}=-\operatorname{expm1}(-\tau_r).
\]

The collision branch is sampled from the exact truncated exponential,

\[
s=-\frac{\log(1-u\,p_{\mathrm{collide},r})}
        {\sum_j\Sigma_{rj}},
\]

and the physical process is selected with probability
\(\Sigma_{rj}/\sum_j\Sigma_{rj}\). The survivor continues to the next
material segment. Collision leaves from every segment plus the final
uncollided leaf conserve the original primary-history weight.

`first_collision_probe.cpp` is a dependency-free contract probe used to
compare the C++ implementation against the batched NumPy oracle. It is not a
transport executable.

## Deliberately missing Geant4 integration

The core remains fail-closed outside an explicitly declared calibration
context. Before it can be used by a calibration executable, a separate
reviewed Geant4 wrapper still has to:

1. use a worker-local `G4Navigator` to enumerate every crossed
   daughter/material boundary;
2. obtain each process cross-section from the wrapped
   `G4BiasingProcessInterface` interaction length at the actual track state;
3. invoke the selected real process through `PostStepDoIt`, then continue
   full secondary transport;
4. attach an auxiliary original-primary and branch-lineage identifier to
   every cloned track, because Geant4 parent IDs alone do not preserve the
   calibration history;
5. aggregate branch contributions and their covariance by original primary
   history in a worker-local event store.

Until those pieces are implemented and independently validated, the standard
unit-weight observation runtime cannot select this core.
