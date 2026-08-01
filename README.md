# LTO: Latent Thinking Optimization

Experiment code for generating LLM reasoning trajectories under question transforms
(paraphrase, simplify, char/word noise, generated-knowledge prompting, ...), extracting
per-layer representation-geometry metrics from each trajectory, training a reward
classifier over (metrics, hidden state) -> correct/incorrect, and selecting among
candidate trajectories via rejection sampling (LTO Algorithm 1).

This repo was consolidated from two Google Colab notebooks that started out ~95%
duplicated code. All shared logic now lives in the [`lto`](src/lto) package; the two
notebooks are thin experiment drivers that just set config and call into it.

## Repo layout

```
src/lto/
  config.py          seeding, device/dtype, model registry, default run config
  nltk_setup.py       one-shot NLTK corpus download
  metrics.py          entropy / effective-rank / anisotropy / intrinsic-dimension
  transforms/
    char_word.py       character-noise and WordNet-synonym transforms
    llm_rewrite.py      LLM-driven paraphrase/simplify/ICR/knowledge rewrites + format-jitter control
  data.py             dataset loading, prompt construction, answer parsing
  model_io.py         tokenizer/model loading, chat-template formatting
  extraction.py       greedy generation + per-step hidden-state/metric extraction
  pipeline.py         run_multi(): the end-to-end trajectory-collection runner
  classifier.py       TrajectoryTransformer model + training / LR sweep
  lto_algorithm.py    rejection-sampling selection (LTO Algorithm 1) + stability check
  analysis.py         flip-count transitions, qualitative inspection, reward-gap
                       diagnostics, feature-engineering + grouped-CV classifier sweep

notebooks/
  01_llama32_1b_commonsense_qa.ipynb   Llama-3.2-1B run, classifier trained on hidden+metrics
  02_qwen3_1p7b_commonsense_qa.ipynb   Qwen3-1.7B run, classifier trained on metrics only,
                                        plus the extended analysis.py diagnostics
```

## Why a shared package instead of two notebooks

Both notebooks ran the same pipeline (metrics, transforms, dataset loading, model
loading, the trajectory-collection runner, the classifier, the LTO algorithm) against
different models and classifier configs. Keeping that logic in one place means a fix or
change only has to happen once, and the notebooks stay short enough to read in one pass.

## Setup

Local / non-Colab:

```bash
pip install -e .
python -c "from lto import ensure_nltk_data; ensure_nltk_data()"
```

In Colab, each notebook's first cell installs the package straight from this repo:

```python
!pip install -q "transformers>=4.51.0" accelerate bitsandbytes datasets nltk pandas tqdm
!pip install -q git+https://github.com/Braj406/Input-Transformation-LTO.git
```

(If you're iterating on the package itself, `!git clone` the repo into `/content` and
`!pip install -e /content/Input-Transformation-LTO` instead, so edits are picked up
without a re-clone.)

## Config that differs per experiment

| | notebook 1 | notebook 2 |
|---|---|---|
| Model | `llama3.2-1b` | `qwen3-1.7b` |
| Classifier input | hidden states + metrics (`USE_HIDDEN=True`) | metrics only (`USE_HIDDEN=False`) |
| Extra analysis | -- | flip counts, qualitative flip inspection, reward-gap / confidence-bucket diagnostics, feature-engineering + grouped-CV sweep (`lto.analysis`) |

Both notebooks otherwise run the same `TRANSFORMS` config against `commonsense_qa`:
`llm_knowledge` (k=5) plus three fixed-style `llm_simplify*` rewrites (k=1 each).

## Secrets

Notebooks read a Hugging Face token from a Colab secret (`HF_TOKEN`) via
`google.colab.userdata` -- add it under the key icon in the Colab sidebar. Never commit a
token to this repo; `.gitignore` also excludes `*.token` and `.env` as a backstop.

## Outputs

`run_multi(...)` checkpoints `.pt` (raw records incl. hidden states/metrics), `.csv`
(flat summary), and `.config.json` (run metadata) to whatever `drive_dir` you pass it
(e.g. `/content/drive/MyDrive/LTO` in Colab). These are run artifacts, not source --
`.gitignore` excludes `*.pt` / `*.csv` / `*.config.json` from the repo.
