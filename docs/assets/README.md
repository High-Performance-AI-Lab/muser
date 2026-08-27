# docs/assets

`muser-social-card.png` is the 1200×630 social preview image, generated
deterministically from receipted values:

```sh
python3 scripts/generate_social_card.py \
  --png-output docs/assets/muser-social-card.png
python3 scripts/generate_social_card.py --check
```

The SVG is the deterministic source; the committed PNG is its render for
platforms that require a raster social preview. The source names the public
benchmark anchor and retained receipt ID. Do not edit either asset by hand
or add an unreceipted figure.

A real dashboard screenshot belongs here too. Capture it from a running
server (`cargo run -p muser-server -- up --no-open`, then screenshot the
served page); do not synthesize one.
