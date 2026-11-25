"""3D reconstruction of ship targets from multi-angle SAR images based on
side-view projection.

Implementation of J. Li, X. Leng, Y. Liu, Z. Zhou and K. Ji, "3D
Reconstruction of Ship Target From Multiangle SAR Images Based on Side-View
Projection", IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 19,
pp. 650-666, 2026.

- :mod:`reconstruction.geometry`: multi-angle SAR projection model.
- :mod:`reconstruction.segmentation`: fine segmentation, Radon azimuth
  estimation and occluded footprint completion.
- :mod:`reconstruction.voxel`: voxel back-projection voting, error region
  deletion and metrics.
- :mod:`reconstruction.simulate`: simulated point target and ship data.
"""
from .geometry import SARView, ImageGrid, rotation_z, views_on_circle
from .segmentation import (
    segment_target, radon_azimuth, slant_to_ground_azimuth, rasterize_polygon,
    deck_outline, complete_footprint,
)
from .voxel import (
    VoxelGrid, vote, reconstruct, remove_error_regions, cluster_centroids,
    match_points, principal_dimensions, voxel_scores,
)
from .simulate import (
    ShipModel, point_scatterer_model, render_points, voxelize,
    visible_surface_points, simulate_ship_images,
)
