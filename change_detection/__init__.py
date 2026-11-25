"""Unsupervised SAR change detection of small objects.

Implementations of two papers by G. Lee and K.-T. Kim (IEEE JSTARS, 2026):

- :func:`superpixel_change_detection`: "Unsupervised SAR Change Detection
  of Small Objects via Superpixel Classification".
- :func:`fusion_change_detection`: "Unsupervised SAR Change Detection of
  Small Objects via Fusion of Difference Images".
- :func:`cs_change_detection`: C. Villamil Lopez and U. Stilla,
  "Monitoring of Construction Activity by Change Detection on SAR Time
  Series Using Coherent Scatterers" (IEEE JSTARS, 2022), for coregistered
  SLC time series.
- :func:`superpixel_emeans_change_detection`: improved superpixel
  classification variant with e-means coarse clustering (needs the
  ``emeans`` package).

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
from .coherent_scatterers import (
    cs_change_detection, CSChangeResult, detect_coherent_scatterers, cd_metric,
    pair_change_classes,
)
from .cs_simulation import make_slc_time_series
from .superpixel_emeans import (
    superpixel_emeans_change_detection, emeans_coarse_classification,
)

__all__ = [
    "lee_filter", "log_ratio", "ncb_image", "otsu_threshold",
    "morphological_opening", "slic", "kmeans", "pca", "dbscan",
    "evaluate_objects", "ObjectScores", "make_synthetic_pair",
    "superpixel_change_detection", "SuperpixelResult",
    "fusion_change_detection", "FusionResult",
    "superpixel_emeans_change_detection", "emeans_coarse_classification",
    "cs_change_detection", "CSChangeResult", "detect_coherent_scatterers",
    "cd_metric", "pair_change_classes", "make_slc_time_series",
]
