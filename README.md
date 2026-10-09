# VLM Broad Area Search for Layered Security Perimeters

This repository is a compact technical review of a vision-language-model pipeline for triaging high-resolution optical satellite imagery. The system searches broad areas for facilities with layered security perimeters and ranks candidate image tiles for human review.

The implementation supports the experiments described in *Catching Proliferation at the Fence: Vision Language Models for Detecting Nuclear Infrastructure*, including the Saratov, Natanz, and Fordow cases and the comparison between Google Maps zoom levels 16 and 17.

This public review copy contains only the paper-facing pipeline. Earlier classification, fine-tuning, prompt-iteration, data-preparation, Streamlit, and one-off conversion scripts from the larger collaboration are intentionally omitted.

## Pipeline

1. `scan_tiles_overscan_crop.py` partitions a circular search area into standardized image tiles, downloads and caches source imagery, and records spatial metadata in a manifest.
2. `two_agent_cached_tiles.py` runs an image-analysis agent followed by a scoring and navigation agent.
3. Ambiguous tiles can trigger one follow-up round: quadrant-level analysis for local detail or a wider contextual view.
4. `flatten_two_agents_results_long.py` converts nested per-tile results into an analysis-ready table.
5. `make_two_agent_report.py` generates a static offline report for ranked human review.

## Repository layout

```text
scripts/
  scan_tiles_overscan_crop.py
  two_agent_cached_tiles.py
  flatten_two_agents_results_long.py
  make_two_agent_report.py
examples/
  minimal_run/                 Synthetic output fixture for offline testing
CONTRIBUTIONS.md
DATA_AND_SHARING.md
```

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add the credentials required for the workflow:

```text
OPENAI_API_KEY=...
GOOGLE_MAPS_STATIC_API_KEY=...
```

Never commit `.env`, downloaded imagery, or generated outputs.

## Generate an image-tile cache

This example uses the paper's baseline tile geometry without prescribing a study location. Imagery retrieval must comply with the provider's terms.

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

## Flatten results and generate a report

```bash
python scripts/flatten_two_agents_results_long.py \
  --run_dir ../IMINT-Outputs/example_run

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

Exercise the result-processing and report-generation path using the included synthetic fixture:

```bash
python scripts/flatten_two_agents_results_long.py \
  --run_dir examples/minimal_run \
  --output_csv /tmp/imint_minimal_results.csv

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
