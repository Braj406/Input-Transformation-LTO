# LTO: Latent Thinking Optimization

Experiment code for studying how question transforms (paraphrase, simplify, char/word
noise, generated-knowledge prompting) affect an LLM's reasoning trajectories. For each
trajectory we extract per-layer representation-geometry metrics, train a reward
classifier over `(metrics, hidden state) -> correct/incorrect`, and select among
candidate trajectories via rejection sampling (LTO Algorithm 1).

All shared logic lives in the [`lto`](src/lto) Python package; the two notebooks in
[`notebooks/`](notebooks) are experiment drivers that set config and call into it.

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
  extraction.py       greedy generation + hidden-state/metric/confidence extraction
  pipeline.py         run_multi() trajectory-collection runner + recover_confidence()
  classifier.py       TrajectoryTransformer model + training / LR sweep
  lto_algorithm.py    rejection-sampling selection (LTO Algorithm 1) + stability check
  analysis.py         flip-count transitions, qualitative inspection, reward-gap
                       diagnostics, feature-engineering + grouped-CV classifier sweep

notebooks/
  Transformations_Llama.ipynb   Llama-3.2-1B run, classifier trained on hidden+metrics
  Transformations_Qwen.ipynb    Qwen3-1.7B run, classifier trained on metrics only,
                                 plus the extended lto.analysis diagnostics and
                                 per-token confidence recovery
```

## How to run

Both notebooks are built for Google Colab and run top to bottom, cell by cell.

1. **Open the notebook.** In Colab: `File → Open notebook → GitHub`, paste this repo's
   URL, and pick `notebooks/Transformations_Llama.ipynb` or
   `notebooks/Transformations_Qwen.ipynb`.
2. **Switch to a GPU runtime.** `Runtime → Change runtime type → T4 GPU` (or better). The
   pipeline will technically run on CPU but will be extremely slow.
3. **(Llama notebook only) Add a Hugging Face token.** Llama-3.2 is a gated model —
   request access on its [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
   first, then add your token as a Colab secret named `HF_TOKEN` (key icon in the left
   sidebar) so the login cell can pick it up automatically. Qwen3 is ungated, so the Qwen
   notebook skips this step entirely.
4. **Run cells in order, top to bottom.** The first few cells install dependencies,
   mount Google Drive (used as persistent storage for run artifacts), and set the
   experiment config. From there each cell builds on the last.
5. **Let the trajectory-collection cell run.** This is the slowest step — it generates a
   baseline + transformed response for every question and extracts hidden
   states/metrics. It checkpoints to Drive every 5 questions (`save_every` in
   `COMPACT_CFG`), so if Colab disconnects, re-running the same cell resumes from the
   last checkpoint instead of starting over.
6. **Classifier training will print a full LR sweep.** It trains 10 seeded runs at each
   of 3 learning rates and reports mean ± std AUC per LR, then loads the best run into
   `model` for everything downstream.
7. **The LTO cells run automatically once the classifier is trained** — no extra config
   needed. The Qwen notebook continues with additional diagnostic cells afterward
   (flip counts, qualitative inspection, reward-gap analysis, a feature-engineering
   sweep, and per-token confidence recovery); each is independent and can be run or
   skipped freely.
8. **Re-running a notebook reuses prior work.** Both the trajectory collection and the
   confidence-recovery pass resume from their last Drive checkpoint (`resume=True`) —
   safe to stop and restart a run at any point.

## Config that differs per experiment

| | Llama notebook | Qwen notebook |
|---|---|---|
| Model | `llama3.2-1b` | `qwen3-1.7b` |
| Hugging Face access | gated -- needs `HF_TOKEN` | ungated -- no login needed |
| Classifier input | hidden states + metrics (`USE_HIDDEN=True`) | metrics only (`USE_HIDDEN=False`) |
| Extra analysis | -- | flip counts, qualitative flip inspection, reward-gap / confidence-bucket diagnostics, feature-engineering + grouped-CV sweep, per-token confidence recovery (`lto.analysis`, `recover_confidence`) |

Both notebooks otherwise run the same `TRANSFORMS` config against `commonsense_qa`:
`llm_knowledge` (k=5) plus three fixed-style `llm_simplify*` rewrites (k=1 each).

## Setup outside Colab

```bash
pip install -e .
python -c "from lto import ensure_nltk_data; ensure_nltk_data()"
```

If you're iterating on the package itself from inside Colab, `!git clone` the repo into
`/content` and `!pip install -e /content/Input-Transformation-LTO` instead of installing
from GitHub, so edits are picked up without a re-clone.

## License

MIT -- see [LICENSE](LICENSE).

## Secrets

Notebooks read a Hugging Face token from a Colab secret (`HF_TOKEN`) via
`google.colab.userdata` -- add it under the key icon in the Colab sidebar. Never commit a
token to this repo; `.gitignore` also excludes `*.token` and `.env` as a backstop.

## Outputs

`run_multi(...)` checkpoints `.pt` (raw records incl. hidden states/metrics), `.csv`
(flat summary), and `.config.json` (run metadata) to whatever `drive_dir` you pass it
(e.g. `/content/drive/MyDrive/LTO` in Colab). `recover_confidence(...)` checkpoints a
separate `*_confidence.pt` file the same way. These are run artifacts, not source --
`.gitignore` excludes `*.pt` / `*.csv` / `*.config.json` from the repo.
