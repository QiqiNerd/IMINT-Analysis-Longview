# Contributions

## Public review implementation

Qiqi Chen developed the implementation included in this repository. The work covers:

- the two-agent architecture separating image analysis from scoring and navigation;
- cached-tile execution and resumable per-tile outputs;
- quadrant-level analysis and wider contextual views for uncertain cases;
- broad-area tiling, overscan, cropping, mosaicking, and standardized 2560 by 2560 pixel inputs;
- the Saratov, Natanz, and Fordow experimental workflows;
- the z16 and z17 image-scale experiments;
- parallel execution support;
- conversion of nested results into analysis-ready tables; and
- static offline reports and tile overlays for ranked human review.

The corresponding files are:

- `scripts/scan_tiles_overscan_crop.py`
- `scripts/two_agent_cached_tiles.py`
- `scripts/flatten_two_agents_results_long.py`
- `scripts/make_two_agent_report.py`

## Broader collaboration

This code grew out of a broader IMINT research collaboration. Selena Sun developed earlier model-development and evaluation infrastructure, including task-specific evaluations, fine-tuning, prompt iteration, data preparation, and visualization workflows. That code is not included in this focused public review repository.

Research framing, case selection, domain interpretation, manuscript authorship, ground-truth review, and analytical conclusions involved the paper's broader author team and should not be inferred solely from software authorship.

This attribution statement describes the functional boundary of the public review copy. It is not an intellectual-property license.
