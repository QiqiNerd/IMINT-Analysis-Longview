# Contributions

This document describes functional contributions to the public review repository. It is based on the contributors' stated division of work, the paper's description of the experimental workflow, and the available Git history. Commit counts are not used as a proxy for intellectual ownership.

## Qiqi Chen

Qiqi developed the paper-facing broad-area-search implementation and the experimental workflows for Saratov, Natanz, and Fordow. This work includes:

- the two-agent architecture separating image analysis from scoring and navigation;
- cached-tile execution and resumable per-tile outputs;
- quadrant-level analysis and wider contextual views for uncertain cases;
- broad-area tiling, overscan, cropping, mosaicking, and standardized 2560 by 2560 pixel inputs;
- the z16 and z17 image-scale experiments;
- parallel execution support; and
- static offline reports and tile overlays for ranked human review.

Primary files:

- `scripts/scan_tiles_overscan_crop.py`
- `scripts/two_agent_cached_tiles.py`
- `scripts/make_two_agent_report.py`

## Selena Sun

Selena developed much of the collaboration's model-development and evaluation infrastructure. This work includes:

- task-specific prompts and evaluations for anti-aircraft artillery, double fencing, and warhead-storage detection;
- labeled-data preparation and train, validation, and test splitting;
- fine-tuning data preparation and metrics review;
- iterative prompt evaluation, clustering, difference analysis, and prompt updating;
- imagery collection, negative sampling, detection clustering, and metric calculation; and
- the exploratory Streamlit visualization workflow.

Primary directories and files:

- `evals/`
- `fine-tuning/`
- `prompt_iteration/`
- `streamlit_app/`
- `scripts/classify_and_fetch_radius_images.py`
- `scripts/cluster_and_analyze_detections.py`
- `scripts/calculate_metrics.py`
- `scripts/google_maps_collection.py`
- `scripts/generate_russian_negatives.py`
- `scripts/random_coordinates.py`
- `scripts/plot_cluster_detections.py`

Some early files were initially committed through an organizational release account and later maintained by Selena. They are described here by function rather than by automated commit totals.

## Shared research context

The repository emerged from a broader collaboration involving the paper's coauthors and their institutions. Research framing, case selection, domain interpretation, manuscript authorship, ground-truth review, and final analytical conclusions should not be inferred solely from code authorship.

Common configuration, dependency files, repository organization, and integration changes are treated as shared project infrastructure unless a more specific contribution is documented above.

This document is an attribution statement, not an intellectual-property license.
