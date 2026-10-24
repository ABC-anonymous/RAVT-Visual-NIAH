# RAVT Visual-NIAH

Companion code for the manuscript:

> **When Does OCR-Free Retrieval Help? A Controlled Study of Text and Visual
> Retrieval under Degradation**

Suggested GitHub repository name: `ravt-visual-niah`.

This repository contains the code required to generate the benchmark, train
RAVT, run the retrieval baselines, reproduce the aggregate results, and verify
that the released task manifest matches the generated evaluation set. The
manuscript is intentionally not included before publication.

## Repository layout

```text
code/
  server_src/       data generation, training, evaluation, and real-page pipeline
  review_extras18/  unseen-corruption and multi-seed adaptation experiments
  analysis/         result aggregation, bootstrap, and manifest export
reproducibility/
  benchmark_manifest.csv
  EXPECTED_RESULTS.md
  DATA_PROVENANCE.md
  SHA256SUMS.txt
```

Datasets, model weights, trained checkpoints, OCR caches, generated images, and
per-task predictions are excluded from Git because of size and third-party
licensing.

## Environment

The experiments used Python 3.12 and CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Download datasets

WikiText-103 comes from
[`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext):

```bash
python - <<'PY'
from datasets import load_dataset

data = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
data.save_to_disk("dataset/wikitext-103")
PY
```

The real-page evaluation uses the DocVQA split of ViDoRe:
[`vidore/docvqa_test_subsampled_beir`](https://huggingface.co/datasets/vidore/docvqa_test_subsampled_beir).

```bash
hf download vidore/docvqa_test_subsampled_beir \
  --repo-type dataset \
  --local-dir data/docvqa_beir
```

Export the page-image layout expected by the evaluation code:

```bash
python code/server_src/ravt_real_docs/export_real_docs.py \
  --pages 500 --out data/vidore_docvqa
```

All downloaded and generated data go under `data/` or `dataset/`, both of which
are ignored by Git.

## Download model weights

Download the standard text and visual baselines with:

```bash
python code/server_src/eval_suite/download_models.py \
  --out models_local
```

| Model | Source | Local destination |
|---|---|---|
| BGE-M3 | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | `models_local/bge-m3` |
| ByT5-small | [`google/byt5-small`](https://huggingface.co/google/byt5-small) | `models_local/byt5-small` |
| DSE-Phi3 | [`Tevatron/dse-phi3-docmatix-v1`](https://huggingface.co/Tevatron/dse-phi3-docmatix-v1) | `models_local/dse-phi3` |
| ColQwen2 base | [`vidore/colqwen2-base`](https://huggingface.co/vidore/colqwen2-base) | `models_local/colqwen2-base` |
| ColQwen2 adapter | [`vidore/colqwen2-v1.0`](https://huggingface.co/vidore/colqwen2-v1.0) | `models_local/colqwen2` |

Download the additional ColQwen2.5 and ColPali adaptation weights:

```bash
python code/server_src/review_extras5/30_download_baselines.py --dest .
```

Download the CLIP image encoder:

```bash
hf download openai/clip-vit-large-patch14 \
  --local-dir clip_local_weights_224
```

For the PEFT-based ColPali setup:

```bash
hf download google/paligemma-3b-mix-448 \
  --local-dir colpali_offline_weights/base_model
hf download vidore/colpali-v1.2 \
  --local-dir colpali_offline_weights/adapter_model
```

Pretrained RAVT checkpoints and the LoRA adapters trained in this study are
provided as GitHub Release assets, not Git objects. Place them under:

```text
checkpoints/full_p14_rh42_s0.pth
...
checkpoints/full_p14_rh42_s4.pth
checkpoints/colpali_lora_s0/
checkpoints/colpali_lora_s1/
checkpoints/colpali_lora_s2/
checkpoints/colqwen2_lora_s0/
checkpoints/colqwen2_lora_s1/
checkpoints/colqwen2_lora_s2/
```

## Generate Visual-NIAH

```bash
export RAVT_BLOCK_HEIGHT=42
python code/server_src/train_pipeline/prepare_data.py \
  --base-dir . \
  --records 50000 \
  --train-blocks data/processed/train_blocks_rh42 \
  --test-bench data/test_bench_tmm_rh42 \
  --grid tmm \
  --test-tasks 720 \
  --extra-benches image,real \
  --overwrite
```

The fixed seed and task construction are implemented in
`code/server_src/train_pipeline/stage_data.py`; the released manifest is in
`reproducibility/benchmark_manifest.csv`.

## Train RAVT

```bash
python code/server_src/train_pipeline/train.py \
  --train-blocks data/processed/train_blocks_rh42 \
  --row-height 42 \
  --tag rh42 \
  --seeds 0 1 2 3 4
```

Add `--dry-run` first to validate the local installation without a full run.

## Evaluate

```bash
python code/server_src/eval_suite/eval_all.py \
  --models RAVT,BM25,BGE-M3,ColPali,ColQwen2 \
  --benches main \
  --bench-main data/test_bench_tmm_rh42 \
  --row-height 42 \
  --out-dir results/jsonl/eval_tmm_rh42 \
  --on-missing error
```

Run the two degradation probes:

```bash
python code/server_src/eval_suite/eval_all.py \
  --models RAVT,BM25,BGE-M3,ColPali,ColQwen2 \
  --benches image,real \
  --bench-image data/test_image \
  --bench-real data/test_real \
  --row-height 42 \
  --out-dir results/jsonl/eval_axes \
  --on-missing error
```

Run the unseen-corruption probe and multi-seed VLM adaptation:

```bash
bash code/review_extras18/run_core.sh --base "$PWD" --python "$(which python)"
```

Run the real-page DocVQA/ViDoRe evaluation:

```bash
bash code/server_src/run_vidore.sh
```

## Verify the released data

Check the generated task set:

```bash
python code/server_src/eval_suite/audit_benchmark.py \
  --bench data/test_bench_tmm_rh42
```

Compare the resulting JSONL aggregates with
`reproducibility/EXPECTED_RESULTS.md`. Dataset provenance is documented in
`reproducibility/DATA_PROVENANCE.md`; exact probe counts are recorded in
`reproducibility/PROBE_CONFIG.json`. Checksums for the manifest and core
evaluation code are in `reproducibility/SHA256SUMS.txt`.

Run the packaged integrity check with:

```bash
python reproducibility/verify.py
```

## Scope

RAVT performs exact-copy short-block localization from a clean transcription.
It requires row segmentation, does not preserve row order, and is not a
general document question-answering system.
