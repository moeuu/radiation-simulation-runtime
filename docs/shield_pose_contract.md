# Spherical-octant shield pose contract

The immutable runtime contract is
`spherical_octant_positive_xyz_incoming_index_v1`.

- A shield orientation index denotes the incoming photon direction from source
  to detector in world coordinates.
- The local Geant4 `G4Sphere` material occupies the positive-X, positive-Y,
  positive-Z octant. Its local material-centre direction is therefore
  `(1, 1, 1) / sqrt(3)`.
- The physical shield material is placed opposite the indexed incoming
  direction. For an active local-to-world rotation `R`, the required identity
  is `R @ (1, 1, 1) / sqrt(3) == -OCTANT_NORMALS[index]`.
- A pair ID is `8 * fe_orientation_index + pb_orientation_index`.
- Robot yaw is a parent transform and must be removed from each shield's local
  rotation so that the commanded octant remains fixed in world coordinates.

Python requests send the contract ID, contract SHA-256, both orientation
indices, and both world quaternions. The native Geant4 sidecar recomputes the
physical material-centre normals and rejects any disagreement before transport.
The same contract identity is stored in MeasurementLog forward-model manifests
and full-spectrum model assets. Legacy artifacts without it are incompatible.

Any semantic change requires a new contract ID, a new payload hash, regenerated
model manifests, and all-64-pair tests under nonzero robot yaw.
