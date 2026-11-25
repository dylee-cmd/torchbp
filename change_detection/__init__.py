"""Unsupervised SAR change detection of small objects.

Implementations of two papers by G. Lee and K.-T. Kim (IEEE JSTARS, 2026):

- :func:`superpixel_change_detection`: "Unsupervised SAR Change Detection
  of Small Objects via Superpixel Classification".
- :func:`fusion_change_detection`: "Unsupervised SAR Change Detection of
  Small Objects via Fusion of Difference Images".

Shared building blocks (difference images, Otsu, SLIC, DBSCAN object-level
evaluation, synthetic data) are in :mod:`change_detection.common`.
"""
from .common import (
    lee_filter, log_ratio, ncb_image, otsu_threshold, morphological_opening,
    slic, kmeans, pca, dbscan, evaluate_objects, ObjectScores,
    make_synthetic_pair,
)
from .superpixel_classification import (
    superpixel_change_detection, SuperpixelResult,
)
from .fusion import fusion_change_detection, FusionResult

__all__ = [
    "lee_filter", "log_ratio", "ncb_image", "otsu_threshold",
    "morphological_opening", "slic", "kmeans", "pca", "dbscan",
    "evaluate_objects", "ObjectScores", "make_synthetic_pair",
    "superpixel_change_detection", "SuperpixelResult",
    "fusion_change_detection", "FusionResult",
]
