# Recipe golden certificates

This directory holds one committed, Ed25519-signed **RenderCertificate** per
`RECIPE_GOLDENS` scene in `tests/test_recipe_goldens.py`, plus the public key
(`signing.pub`) used to verify them.

## What these files are

Each `<scene_id>.json` is the full execution report for that scene's golden
render — engine version + per-module WGSL source hashes, adapter identity,
negotiated GPU capabilities, the per-pass timing ledger, the peak allocation
ledger, the (empty, for a clean golden) degradation list, and an Ed25519
`signature` block. The certificate is assembled by
`forge3d.diagnostics.render_certificate()` and written by
`forge3d.certificate.write_certificate()`.

`signing.pub` is the 64-char hex Ed25519 **public** key that verifies every
certificate in this directory.

## How they are used by the test suite

Ordinary recipe pixel tests and the dedicated Metal and NVIDIA/Vulkan physical
lanes prove the selected backend pixels, physical adapter identity, absence of
software fallback, expected adapter features, and fixture provenance. These
tests do not call the certificate emitter or verifier, require a signing secret,
or compare changed pull-request WGSL with protected-base signed certificates.

Pre-merge certificate proof is owned by the protected base and runs only in the
required `workflow_dispatch` with `scope=full` (and scheduled acceptance), not
on ordinary pull-request or push events. It requires the candidate catalog,
`signing.pub`, and certificate bytes to be identical to the protected-base
versions, then uses the public verifier to check signatures and reject tampering
and replay. This proof requires no signing secret.

## How to regenerate

For a genuine certificate rotation, manually dispatch `certificate-refresh.yml`
from protected `main`. The protected production secret signs clean fresh
certificates for the complete recipe catalog; each certificate is checked
against the pinned public key before it is written. Certificate rotation never
creates or updates pixel goldens.

## Production signing provenance

Committed certificates are signed by the repository production key. Only its
public key is tracked in `signing.pub`; the random 32-byte seed is stored as the
GitHub Actions secret `FORGE3D_CERT_SIGNING_KEY`. A protected, explicitly
dispatched acceptance/release signing lane fails when the secret is absent,
when a certificate uses the local development key, when its public key differs
from `signing.pub`, or when verification fails. Routine internal and fork pull
requests receive no production secret and do not run the dedicated base-owned
verifier; their Fast/static contracts exercise candidate-owned contract code.
Pre-merge acceptance requires the separate full manual dispatch, where the
base-owned verifier treats the candidate tree as explicitly untrusted work.

Offline verification needs no secret or native extension:

```bash
python -m forge3d.certificate verify tests/golden/certificates/mapscene_terrain_raster.json \
    --pubkey tests/golden/certificates/signing.pub
```

Local development may still use the clearly labelled
`forge3d.certificate.DEV_SIGNING_SEED`; that key is never accepted for committed
release/golden certificates.

## Key rotation

Rotate atomically in one reviewed PR:

1. Generate a cryptographically random 32-byte Ed25519 seed without printing or
   committing it, and replace the Actions secret.
2. Re-sign every committed certificate with that seed; do not alter its signed
   payload or pixel golden.
3. Replace `signing.pub` with the corresponding public key in the same commit.
4. Run the offline verifier sweep and protected golden lane. Merge only when all
   certificates verify against the new pinned public key and the old key is
   rejected.

If any step fails, restore the previous secret and discard the rotation commit;
never ship a mixed public-key/certificate set.
