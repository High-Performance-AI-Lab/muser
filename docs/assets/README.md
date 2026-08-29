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

`muser-onboarding-and-remote-prefill.mp4` is the source-controlled copy of the
real 1600×900 H.264 console capture. The root README embeds the same file as
a GitHub user attachment so GitHub renders its native video player on desktop
and mobile instead of opening the repository file browser. Its SHA-256 is
`442073fdad1e5e226c57bd285197ca1b9ef36b5aef0f611c60d32ac0802b9ea1`.
Accelerated sections are labeled in the video; the answer and telemetry are
shown in real time. `muser-onboarding-and-remote-prefill.png` is an extracted
frame from that capture and is the archival poster.

A real dashboard screenshot belongs here too. Capture it from a running
server (`cargo run -p muser-server -- up --no-open`, then screenshot the
served page); do not synthesize one.
