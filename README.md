# IMINT Analysis

This repository presents research code from a collaboration on vision-language-model analysis of high-resolution optical satellite imagery. It includes both the paper-facing two-agent broad-area-search pipeline and the supporting evaluation, fine-tuning, prompt-iteration, data-preparation, and visualization workflows developed during the project.

The two-agent implementation supports the experiments described in *Catching Proliferation at the Fence: Vision Language Models for Detecting Nuclear Infrastructure*, including the Saratov, Natanz, and Fordow cases and the comparison between Google Maps zoom levels 16 and 17.

This public review copy is intentionally code-focused. It excludes imagery, coordinate datasets, model outputs, credentials, local environments, and one-off conversion or repair scripts. See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for the functional division of work and [DATA_AND_SHARING.md](DATA_AND_SHARING.md) for the public-sharing boundary.

## Paper-facing two-agent workflow

1. `scripts/scan_tiles_overscan_crop.py` partitions a circular search area into standardized image tiles, downloads and caches source imagery, and records spatial metadata in a manifest.
2. `scripts/two_agent_cached_tiles.py` runs an image-analysis agent followed by a scoring and navigation agent.
3. Ambiguous tiles can trigger one follow-up round: quadrant-level analysis for local detail or a wider contextual view.
4. `scripts/make_two_agent_report.py` reads the run outputs directly and generates a static offline report for ranked human review.

## Supporting research workflows

- `evals/` contains prompts and evaluation utilities for anti-aircraft artillery, double-fence, and warhead-storage detection.
- `fine-tuning/` contains training-record preparation and fine-tuning metric visualization.
- `prompt_iteration/` contains the iterative prompt-analysis, clustering, evaluation, and update workflow.
- `streamlit_app/` contains an exploratory imagery visualization interface.
- `scripts/classify_and_fetch_radius_images.py` performs model-assisted regional scanning.
- `scripts/cluster_and_analyze_detections.py` and `scripts/plot_cluster_detections.py` group and visualize detections.
- `scripts/google_maps_collection.py` supports imagery collection.
- `scripts/calculate_metrics.py`, `scripts/generate_russian_negatives.py`, and `scripts/random_coordinates.py` support evaluation-data preparation and analysis.

One-off utilities such as CSV/JSON converters, result flatteners, batch rerun wrappers, repair scripts, and duplicate report implementations are not included.

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Provide only the credentials required for the workflow being run. Never commit `.env`, downloaded imagery, or generated outputs.

## Generate a tile cache

The following example uses the paper's baseline tile geometry without prescribing a study location. Imagery retrieval must comply with the provider's terms.

```bash
python scripts/scan_tiles_overscan_crop.py \
  --lat <CENTER_LATITUDE> \
  --lon <CENTER_LONGITUDE> \
  --radius_km 30 \
  --tile_km 1.9 \
  --size 640 \
  --scale 2 \
  --maptype satellite \
  --auto_mosaic \
  --output_px 2560 \
  --mosaic_overlap_m 0 \
  --skip_raw \
  --output ../IMINT-Images/example_z16
```

Run `python scripts/scan_tiles_overscan_crop.py --help` for zoom controls and the complete parameter set. In the paper, z16 corresponds to approximately 1.9 km square tiles and z17 to approximately 0.95 km square tiles, with latitude-dependent ground sampling distance.

## Run the two agents

```bash
python scripts/two_agent_cached_tiles.py \
  --cache_dir ../IMINT-Images/example_z16 \
  --model gpt-4o \
  --workers 1 \
  --output ../IMINT-Outputs/example_run \
  --copy_final_yes \
  --copy_final_unresolved
```

The pipeline writes resumable run metadata, per-tile JSON records, and CSV/JSONL summaries. The model name is an example; use a vision-capable model available to the reviewing account.

## Generate the review report

```bash
python scripts/make_two_agent_report.py \
  --run_dir ../IMINT-Outputs/example_run \
  --scan_dir ../IMINT-Images/example_z16 \
  --output_dir ../IMINT-Outputs/example_report \
  --neighbors none
```

Open `example_report/index.html` locally to inspect ranked tiles, scores, navigation decisions, and model explanations.

## Offline verification

Syntax-check the repository without calling an external API:

```bash
PYTHONPYCACHEPREFIX=/tmp/imint-pycache python -m compileall -q .
```

Exercise static report generation using the included synthetic fixture:

```bash
python scripts/make_two_agent_report.py \
  --run_dir examples/minimal_run \
  --output_dir /tmp/imint_minimal_report \
  --neighbors none \
  --overwrite
```

The fixture contains no imagery or real-world location data.

## Limitations

This is research code, not an operational verification system. Its outputs are triage recommendations rather than evidence of nuclear activity. Performance depends on imagery date, resolution, visibility, model version, prompt configuration, tiling geometry, and the human review threshold. Optical imagery cannot reveal concealed or non-visual signatures, and false positives can arise from ordinary industrial, transportation, agricultural, or terrain features.

## License

No open-source license is granted. The code is publicly visible for technical review, but no permission is granted to reproduce, redistribute, or incorporate it into another product without authorization from the relevant rights holders.
