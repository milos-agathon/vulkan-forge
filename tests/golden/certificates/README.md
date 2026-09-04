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

In normal (hardware-backed) mode, `test_recipe_goldens_render_and_match`:

1. asserts the committed `<scene_id>.json` exists and
   `forge3d.certificate.verify(cert, signing.pub)` is `True`;
2. asserts the committed certificate's `degradations` are empty; and
3. asserts the FRESH in-process render's `engine.wgsl_module_hashes` match the
   committed certificate's — the load-bearing **shader tamper** check that ties
   the committed certificates to the current WGSL sources. If a shader changes,
   this assertion fails with a message directing you to regenerate.

## How to regenerate

Run the recipe golden suite with the update flag (this also refreshes the
committed pixel goldens under `tests/golden/recipes/`):

```bash
FORGE3D_UPDATE_RECIPE_GOLDENS=1 .venv/Scripts/python -m pytest tests/test_recipe_goldens.py -q
```

Regenerate only after you have verified the pixel goldens are correct. The WGSL
hash check exists precisely so that a shader edit forces a conscious
regeneration rather than silently drifting.

## Production signing provenance

Committed certificates are signed by the repository production key. Only its
public key is tracked in `signing.pub`; the random 32-byte seed is stored as the
GitHub Actions secret `FORGE3D_CERT_SIGNING_KEY`. A protected, explicitly
dispatched acceptance/release signing lane fails when the secret is absent,
when a certificate uses the local development key, when its public key differs
from `signing.pub`, or when verification fails. Routine internal and fork pull
requests receive no production secret; they verify structure, canonicalization,
the pinned public key, and tamper rejection as explicitly untrusted work.

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
