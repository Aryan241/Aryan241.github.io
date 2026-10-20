---
layout: archive
title: "Research"
permalink: /research/
author_profile: true
---

### Auditing private-detail removal in 3D Gaussian Splatting
*Research proposal, AAAI-27 Undergraduate Consortium (submitted, decision pending), 2026*

Tools that redact a private detail from a 3DGS scene are checked by looking at renders. But a 3DGS file stores the positions, sizes, opacities and colours of hundreds of thousands of Gaussians, and many different files produce the same renders. I ask whether a redacted detail can still be recovered from the file, judged against a model retrained on images that never contained it.

- **Synthetic pilot** (random 256-bit pattern, 9 runs per method): recolouring and fine-tuning made renders unreadable, but the pattern stayed readable from the parameters (bit accuracy 0.85–0.96, against 0.44–0.58 for the retrained model). Selecting Gaussians by rendering contribution, as common 2D-to-3D segmentation methods do, missed about 3% of them, and those still carried the pattern. Rebuilding the whole region reached the retrained model's range, at a 4 dB patch-PSNR cost.
- **Real phone capture** (desk with a blank A4 sheet, pattern inserted onto the sheet digitally; 2 seeds × 3 patterns): fine-tuning and the standard densify and opacity-reset clean-up left the pattern readable from the number of Gaussians per cell (0.84–1.00). Only rebuilding the whole region stayed in the retrained model's range on every read-out inside the region; a weak signal just outside the region remained after every repair.
- **Limits:** one synthetic and one real scene, a known region, and simple hand-made attackers. Next steps are printed patterns, text read by an OCR model, learned attackers, and finding how far beyond the detail the rebuilt region must reach.

### Object-landmark correction for dense 3DGS SLAM
*Equal-contribution co-author with Om Kumar; manuscript in preparation, 2026*

My part was lifting 2D object masks onto the Gaussian map, cleaning them with depth, visibility and multi-view checks, and the evaluation. On a dense 119-frame sequence the cleaned landmark passed the correction guard in 20 of 24 trials (a raw lift: 0 of 24) and cut trajectory error from 0.319 m to 0.144 m. With sparser frame sampling the same correction raised error by 2–24%; the cause is still open.

Projects
======

### Training-free distractor removal for 3D Gaussian Splatting
*Independent project, 2026 · [code](https://github.com/Aryan241/Clean-Splate)*

Cleans an already-trained scene without retraining. Grounding DINO and SAM 2 produce text-prompted masks across the capture, and each Gaussian is kept or pruned by a multi-view vote on where its centre projects. On 12 Nerfbusters scenes: 46–94% smaller files and higher PSNR inside the object mask on 11 (measured on 3 sampled training views per scene, not held-out views). The cleanup step takes 1.5–17 s and under 350 MB of memory per scene on a laptop.

### Single-image height estimation from aerial imagery
*Smart India Hackathon 2026, ISRO/SAC problem statement, team project · [code](https://github.com/Aryan241/Heimdall)*

A 1.72M-parameter decoder on top of a frozen Depth Anything V2 predicts per-pixel scale and shift, turning relative depth into metric height above ground; trained on 16k GAMUS and Dutch LiDAR tiles. On 8 held-out Dutch LiDAR sites: 5.32 m height RMSE (r = 0.67), against 8.37 m for predicting flat ground. Tall buildings and tree canopy are under-estimated, because a single top-down image carries no parallax.
