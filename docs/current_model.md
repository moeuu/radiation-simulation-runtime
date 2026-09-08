# Current model and format identifiers

Production has one implementation: the isotope-independent detector Green
operator, XCOM dry-air attenuation, detector-cone Compton transport, and the
predeclared physical uncertainty model. Model profiles differ by isotope catalog
and approval scope, not by implementation generation. PF consumes the runtime's
shared model format constant instead of maintaining a second version choice.

Normal runtime and PF configuration files omit implementation-format numbers.
Loaders resolve them to the current format before publishing or hashing resolved
settings. Explicit unsupported numbers still fail; omitting a number does not
make an obsolete model readable. Physics, acquisition, and inference parameters
remain explicit.

The model manifest's `schema_version: 7`, response schema, protocol format tags,
hash domain strings, and recorded result identifiers stay unchanged. They are
serialization identities, not old code or optional implementations. Relabeling
them would break retained approval, observation, and posterior bindings without
changing the scientific method. Direct all-64 validation and transferred
catalog-independent approval are distinct current evidence types and retain
their distinct serialized schemas. Package and dependency versions also retain
their packaging meaning.

## Removed implementations

- Scene-fitted additive/direct transport response and its historical readers.
- Fixed-quota scene mean-calibration runner, launcher, and dedicated fit tests.
- Low-rank learned spectrum correction, application paths, and training readers.
- Historical discrepancy-training validators for the retired training formats.
- Interaction-opportunity, line-of-sight-only, and pre-XCOM detector-cone bases.

Do not keep copied code in `legacy/`, `archive/`, or version-suffixed modules.
Recover old implementations from Git history in a separate research checkout
when explicitly needed. Keep small rejection cases for unsupported inputs,
current CPU/Torch physics checks, and current model authentication tests.

Explicit nonproduction constructors remain available for isolated likelihood
experiments and mathematical tests. Their unapproved outputs cannot be loaded
through the production manifest loader. Shared discrepancy-calibration artifacts
are also an explicit diagnostic API, not a production-model selector. Neither
path provides the removed historical implementations.

For future replacements, update the canonical implementation and its callers
together. Remove superseded code and dedicated tests once no current consumer
needs them. Keep persisted identifiers stable for behavior-preserving cleanup;
change the scientific contract only when its semantics actually change and
obtain the corresponding new independent evidence. Use unique run directories
for new outputs rather than model-version folder names.
