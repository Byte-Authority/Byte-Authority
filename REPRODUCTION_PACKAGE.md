# Reproduction package

This directory is the curated release bundle for the paper. It contains the core runners, compact per-case records, split metadata, and invariant checks used by the primary experiments. It excludes model checkpoints, full generated responses, runtime logs, credentials, private deployment configuration, and the unfiltered experiment tree.

## Dependencies

Install the packages in `requirements.txt`. The runners expect a local tokenizer/checkpoint path supplied on the command line and a local vLLM-compatible inference endpoint when generation is run. AgentDojo runs additionally require the upstream AgentDojo package.

## Upstream inputs

The benchmark inputs are not bundled. Download InjecAgent and AgentDojo from the links in `data/sources.json`, retain their upstream licenses, and place local copies under `data/third_party/`.

## Layout

- `src/` contains prompt construction, provenance-aware encoding, transport, identity replacement, marker-variant search, and AgentDojo adapters.
- `data/results/` contains compact JSONL case records. Generated text and runtime traces are omitted.
- `data/catalogs/` contains marker-variant mappings and public condition names used by the search experiments.
- `data/manifests/` contains split and input-invariant metadata.
- `data/condition_names.json` maps public condition names to their descriptions.
- `licenses/` contains the licenses for the two upstream benchmarks.
- `assets/` contains paper-derived figures and project artwork.

## Running

Run any module with `--help` to see its actual arguments. Supply a local tokenizer/checkpoint and endpoint; no module downloads model files implicitly. The included records can be inspected without model files.
