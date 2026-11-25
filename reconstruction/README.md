# 3D reconstruction of ship targets from multi-angle SAR images

Implementation of J. Li, X. Leng, Y. Liu, Z. Zhou and K. Ji, *3D Reconstruction of Ship Target From Multiangle SAR Images Based on Side-View Projection*, IEEE JSTARS, vol. 19, pp. 650–666, 2026.

| Module | Paper section |
| --- | --- |
| `geometry.py` | II-A/B: global/local frames (eq. 1, 2), height projection factors `k_x, k_y` (eq. 5, 6), projection matrix `E_i` (eq. 7–9) |
| `segmentation.py` | III-A/B: fine segmentation, Radon azimuth estimation (eq. 10–13), occluded footprint completion |
| `voxel.py` | III-C/D: voxel back-projection voting, error region deletion (eq. 14), reconstruction rate and RMSE (eq. 15, 16), dimension and voxel scores |
| `simulate.py` | IV-A/B: point scatterer model and a voxelized ship (hull, superstructure, mast) imaged with self-occlusion |
| `demo.py` | The paper's simulated experiments |

Dependencies: NumPy and SciPy (matplotlib only for `--plot`).

## Usage

```python
import numpy as np
from reconstruction import (
    ShipModel, views_on_circle, simulate_ship_images, segment_target,
    complete_footprint, VoxelGrid, reconstruct, remove_error_regions,
)

ship = ShipModel(heading=np.deg2rad(20))
views = views_on_circle(36, theta_deg=30)            # 10 deg azimuth steps
images, grids = simulate_ship_images(ship, views)

masks = [segment_target(im, smooth=1, open_size=1, ellipse_scale=None) for im in images]
masks = [complete_footprint(m, v, g, ship.deck_outline(), 0.0, ship.freeboard)
         for m, v, g in zip(masks, views, grids)]

lo, hi = ship.bbox()
vg = VoxelGrid.from_bounds(lo, hi, 0.5)
occ = reconstruct(vg, views, masks, grids)            # vote threshold = number of views
occ = remove_error_regions(occ, vg, theta=np.deg2rad(30), heading=ship.heading)
points = vg.points(occ)
```

### Using real images

For real images, each view needs three things:

- A `SARView`. The rotation `alpha` and translation `d_origin` come from orbit information or from the registration of Section III-B: `radon_azimuth` gives the ship axis in the image, and a control point on the stern/bow deck gives the translation. `theta`, and optionally `beta` and `phi`, come from the imaging geometry.
- An `ImageGrid` that maps the image plane to pixels.
- A segmented mask.

`complete_footprint` takes the prior deck outline (`deck_outline(center, heading, length, width)`) and the waterline and deck heights.

## Method

1. **Projection model.** The rotation `alpha_i` and translation `dO_i` take global coordinates to the view's local frame. In the far field the scene is projected along the normal of the slant range plane, so a scatterer at height `z` is offset by `(k_x z, k_y z)`, with `k_x = (sin φ tan β cot θ + tan β)/cos φ` and `k_y = -cot θ`. Together these form the `E_i` of eq. (8).
2. **Segmentation.** Threshold at a multiple of the grayscale mean, then morphological opening and closing, then keep the largest component, fill its holes, and apply an elliptical shape constraint.
3. **Footprint completion.** Side-view imaging hides the far-side hull wall and the deck areas in the superstructure's shadow. The hull is treated as a prism with the prior deck outline between the waterline and the deck plane. Its full projection (the union of the outline projected at every height in between) is added to each mask.
4. **Voting.** Every candidate voxel is projected into every view with `E_i`. It gets a vote when it lands inside the segmentation area. Voxels whose votes reach the threshold (by default every view) are kept.
5. **Error region deletion.** The cloud is rotated so the ship axis is along X and sliced parallel to YOZ. In each slice, the top roof whose flanks fall with slope `tan θ` (eq. 14) is followed from its vertex. The columns are clipped to the height where the profile stops satisfying eq. (14).

## Results on simulated data (`python -m reconstruction.demo`)

**Point scatterers** (18 points, 4 layers, incidence 30°, 10° azimuth steps):

| Views | Reconstruction rate | RMSE |
| --- | --- | --- |
| 5 | 1.0 (22 clusters) | 0.19 m |
| 25 | 1.0 | 0.03 m |

**Ship** (40 × 10 m hull, superstructure, 16 m mast; 36 views; voxel 0.5 m). True voxel-model extents are 39.6 / 10.0 / 16.0 m.

| Stage | Voxel precision | Recall | IoU | L / W / H [m] |
| --- | --- | --- | --- | --- |
| direct | 0.68 | 0.27 | 0.24 | 39.4 / 10.0 / 16.0 |
| + footprint completion | 0.70 | 0.99 | 0.70 | 40.0 / 10.9 / 16.0 |
| + error region deletion | 0.76 | 0.96 | 0.74 | 40.0 / 10.9 / 16.0 |

As in the paper, both of these improve the reconstruction:

- more views at a fixed 10° interval: IoU 0.31 / 0.60 / 0.69 / 0.74 for 8 / 12 / 16 / 36 views;
- a larger interval with four views: IoU 0.26 / 0.43 / 0.66 / 0.66 for 20° / 30° / 60° / 90°.

With speckle and sea clutter, the default segmentation gives IoU 0.72 after both steps. Its 3×3 opening removes the 1 m mast streak, so the mast is lost and the height drops to 10 m.

## Deviations and notes

- **Radon criterion.** Eq. (12) picks the angle with the largest maximum of the Radon projection. For a wide hull that is the diagonal, not the axis (a 40 × 10 m rectangle has a 41.2 m diagonal), which biases the estimate by up to `atan(W/L)`. The default `criterion="energy"` instead maximizes the projection concentration `ΣR²/ΣR`. `criterion="max"` gives the literal form.
- **Footprint completion.** The paper describes this as filling the occluded deck footprint using the constant deck width prior. Here it is implemented as adding the projection of the prior hull prism, deck outline × [waterline, deck]. That covers the shadowed deck areas and the hidden far-side hull wall in one step.
- **Vote threshold.** The paper's text says the threshold should be "equal or close to the total number of pixels in the image". This is read as the number of *images*. The default threshold is all views.
- **Error-region deletion on multi-peak slices.** Eq. (14) uses one vertex per slice. Slices with several peaks (deck plus mast) are handled by processing the vertices from the highest down: each removes only its own roof, and an isolated peak with no sloped flank, such as the mast, is kept. Some slices stay unclipped, which leaves a slightly streaky deck in the plots.
- **Simulator.** Single backscatter from the visible surface voxels, no electromagnetic scattering model, as in the paper's simulated image data. The paper's electromagnetic simulation and its Qilu-1 measured data are not reproduced.

## Tests

```bash
pytest tests/test_reconstruction.py
```
