# Implicit Aspect Extraction for English and Arabic

Implementation accompanying the work on **implicit aspect extraction using
instruction-tuned LLMs, paraphrase consensus, and knowledge-graph inference**,
evaluated on English and Arabic ABSA benchmarks.

The system is a four-component hybrid pipeline that casts aspect extraction as a
controlled text-generation task:

| Component | Role |
|---|---|
| **BB** — Backbone fine-tuned LLM | Llama-3-8B adapted with QLoRA to emit aspect terms as free text |
| **IG** — Iterative generation | Residual masking across rounds, so secondary aspects are not lost to single-pass decoding |
| **PARA** — Paraphrase module | Generates *k* rewrites per sentence; keeps only aspects reproduced across rewrites |
| **KG** — Knowledge graph | Clue-gated symbolic inference that maps opinion clues to domain-consistent implicit aspects |

A single post-processing step normalizes and deduplicates the unified candidate
pool at the end. The same architecture is used for both languages; only the data
the graph is built from and the preprocessing applied before clue matching differ.

---

## Repository structure

```
.
├── pynb/                     # Colab notebooks — the canonical, runnable artifacts
│   ├── 1.Llama3_LoraFinetuning.ipynb
│   ├── 2.Llama3_ZeroShotPrompting.ipynb
│   ├── 3.EnglishModel_Inference.ipynb
│   ├── 4.ArabicModel_Inference.ipynb
│   ├── 5.English_KG_Inference_and_Evaluation.ipynb
│   ├── 6.Arabic_Knowledge_Graph_(CrossLingual_VS_Arabic_KG).ipynb
│   └── A.ParaphrasingModule.ipynb
├── py/                       # Plain-text exports of the same notebooks (for reading and diffing)
├── datasets/                 # Prepared train/test splits (CSV)
├── Knowledge Graph files/    # Serialized English and Arabic knowledge graphs (JSON)
└── requirements.txt
```

> **Note on `py/`.** These files are Colab exports and retain IPython magics
> (`!pip install`, `%load_ext`) and `google.colab` imports, so they are **not**
> directly executable with `python file.py`. They are included so the code can be
> read, searched and diffed without opening a notebook. To *run* the pipeline,
> use the matching notebook in `pynb/`.

---

## Environment

All experiments were run on **Google Colab** with an **NVIDIA A100 SXM4 (40 GB)**
and the **High-RAM** runtime enabled. The High-RAM runtime is required at LoRA
rank `r = 128`.

| Layer | Version |
|---|---|
| OS | Ubuntu 22.04 LTS (Colab) |
| Python | 3.12 |
| PyTorch | 2.10 (CUDA 12.8 build) |
| CUDA | 12.8 |
| Transformers | 4.56.2 |
| PEFT | 0.18.1 (English) / 0.19.1 (Arabic) |
| TRL | 0.22.2 |
| BitsAndBytes | 0.49.2 |
| Unsloth | 2026.4.6 (English) / 2026.4.8 (Arabic) |
| Accelerate | 1.13.0 |
| spaCy | 3.8 (`en_core_web_sm`) |
| Stanza | 1.11.0 (`tokenize, pos, lemma, depparse`) |
| CAMeL-Tools | 1.5.7 (MLE disambiguator) |
| Sentence-Transformers | `paraphrase-multilingual-MiniLM-L12-v2` |

Install with:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('en')"
camel_data -i disambig-mle-calima-msa-r13          # Arabic morphological disambiguator
```

The notebooks install their own dependencies in their first cells, so on Colab
this step is handled automatically.

### Models

| Model | Used for |
|---|---|
| `unsloth/llama-3-8b-bnb-4bit` | 4-bit backbone for QLoRA fine-tuning and inference |
| `meta/meta-llama-3-8b-instruct` (Replicate) | Zero-shot prompting baseline |
| `gpt-4.1-mini` (OpenAI) | Paraphrase generation |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual similarity for consensus filtering and KG scoring |
| `all-MiniLM-L6-v2` | English-only embedding variant |

Llama-3 is a gated model on Hugging Face: accept the license on the model page
before running, and supply a token.

### Credentials

The notebooks read secrets from the Colab **Secrets** panel
(`google.colab.userdata`). Define the following keys before running:

| Secret name | Purpose |
|---|---|
| `hf_key`, `hf_new` | Hugging Face tokens (gated Llama-3 access) |
| `openAi_key`, `MyGPT` | OpenAI API key for paraphrase generation |
| *(Replicate token)* | Set in-notebook in `2.Llama3_ZeroShotPrompting` |

No credentials are committed to this repository.

---

## Expected Google Drive layout

The notebooks mount Google Drive and read from `MyDrive/colab_files/`. Copy the
contents of `datasets/` and `Knowledge Graph files/` there before running:

```
/content/drive/MyDrive/colab_files/
├── English_train.csv
├── Arabic_train.csv
├── Arabic_test2.csv
├── test_14.csv
├── test_15.csv
├── test_16.csv
├── test_acos.csv
├── m-absa-english-train.csv
├── m-absa-arabic-train.csv
├── m-absa_English_test.csv
├── m-absa_Arabic_test.csv
├── Knowledge_graph_mabsaEnriched.json
├── arabic_enriched_kg.json
└── encoded_kg/
    └── encoded_kg_minilm.pkl        # embedding cache, regenerated on first run
```

---

## Running order

| # | Notebook | What it does |
|---|---|---|
| A | `A.ParaphrasingModule` | Generates 10 paraphrases per sentence via the OpenAI API and writes them back into the CSV `paraphrases` column. Writes incrementally and resumes from a partial file, so it can be interrupted safely. **Run first** — the test CSVs in this repository already ship with this column populated, so this step only needs rerunning for new data. |
| 1 | `1.Llama3_LoraFinetuning` | Fine-tunes Llama-3-8B with QLoRA on the training split. Produces a per-dataset LoRA adapter. |
| 2 | `2.Llama3_ZeroShotPrompting` | Zero-shot prompting baseline via Replicate, for comparison against the fine-tuned backbone. |
| 3 | `3.EnglishModel_Inference` | English explicit track: iterative generation + paraphrase consensus over the fine-tuned backbone. |
| 4 | `4.ArabicModel_Inference` | Arabic explicit track: as above, with CAMeL-Tools preprocessing (clitic segmentation, orthographic normalization, lemma-base disambiguation). |
| 5 | `5.English_KG_Inference_and_Evaluation` | English implicit track: clue detection, KG inference, and full evaluation. |
| 6 | `6.Arabic_Knowledge_Graph_(CrossLingual_VS_Arabic_KG)` | Arabic implicit track, plus the cross-lingual vs. Arabic-native KG comparison. |

Steps 1–2 are independent of each other. Steps 3–4 require the adapter from
step 1. Steps 5–6 require the knowledge-graph JSON files.

---

## Fine-tuning hyperparameters

| Setting | Value |
|---|---|
| LoRA rank `r` | 128 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| Quantization | 4-bit NF4 (QLoRA) |
| Batch size (per device) | 8 |
| Gradient accumulation | 1 |
| Warmup steps | 2 |
| Learning rate | 1e-4 |
| Optimizer | `paged_adamw_32bit` |
| Max steps | dataset-dependent (≈ 4–6 epochs) |
| Eval / save strategy | every 200 steps |
| Model selection | `load_best_model_at_end=True`, `metric_for_best_model=eval_loss` |

---

## Datasets

All corpora are **publicly available benchmarks**, and the files here are the
**official train and test splits, unchanged** — the same sentences, the same gold
annotations and the same split boundaries as the original releases. Nothing was
resampled, filtered or re-annotated.

The only changes are mechanical:

- the distribution format was converted from **XML to CSV**;
- splits from several corpora were concatenated into single files where the
  table below says so;
- a `type` column was added, derived deterministically from the gold annotation
  (a `NULL` opinion target marks an implicit aspect, an annotated span an
  explicit one);
- a `paraphrases` column was added to the test files, generated by the
  paraphrase module.

Each corpus is governed by its own license — consult the sources below.

| Source corpus | Language | Official release |
|---|---|---|
| SemEval-2014 Task 4 | English | https://alt.qcri.org/semeval2014/task4/ |
| SemEval-2015 Task 12 | English | https://alt.qcri.org/semeval2015/task12/ |
| SemEval-2016 Task 5 | English + Arabic | https://alt.qcri.org/semeval2016/task5/ |
| ACOS | English | https://github.com/NUSTM/ACOS |
| M-ABSA | English + Arabic | https://github.com/swaggy66/M-ABSA |
| HAAD | Arabic | https://github.com/msmadi/HAAD |

### Files

| File | Rows | Contents |
|---|---|---|
| `English_train.csv` | 18,432 | SemEval-2014 (8,823), ACOS (4,961), SemEval-2016 (2,799), SemEval-2015 (1,849) |
| `Arabic_train.csv` | 5,974 | SemEval-2016 Arabic hotels (4,776), HAAD books (1,198) |
| `Arabic_test2.csv` | 1,525 | SemEval-2016 Arabic (1,226), HAAD (299) |
| `test_14.csv` | 1,600 | SemEval-2014 test — restaurants + laptops |
| `test_15.csv` | 950 | SemEval-2015 test — restaurants (684) + hotels (266) |
| `test_16.csv` | 586 | SemEval-2016 English test — restaurants |
| `test_acos.csv` | 1,399 | ACOS test — restaurants + laptops |
| `m-absa-english-train.csv` | 8,796 | M-ABSA English train, 7 domains |
| `m-absa-arabic-train.csv` | 5,061 | M-ABSA Arabic train, 4 domains |
| `m-absa_English_test.csv` | 3,794 | M-ABSA English test |
| `m-absa_Arabic_test.csv` | 2,182 | M-ABSA Arabic test |

### Schema

Test files and `Arabic_train.csv` use:

| Column | Description |
|---|---|
| `index` | Row identifier (test files only) |
| `dataset` / `source_dataset` | Originating corpus |
| `domain` | Review domain (`restaurants`, `laptops`, `hotels`, `books`, …) |
| `sentence` | Review sentence |
| `aspect` | Gold aspect term(s), a stringified Python list |
| `category` | Gold `ENTITY#ATTRIBUTE` category, a stringified Python list |
| `type` | `explicit` \| `implicit` \| `mixed` \| `objective` |
| `language` | `English` \| `Arabic` |
| `paraphrases` | Pre-generated paraphrases (stringified list) |

`English_train.csv` omits `index` and `paraphrases`. The two `m-absa-*-train.csv`
files are closer to the upstream M-ABSA release and use a reduced schema
(`m-absa-english-train.csv` names the text column `text` rather than `sentence`).

---

## Knowledge graphs

| File | Structure |
|---|---|
| `Knowledge_graph_mabsaEnriched.json` | English KG — keys: `domains`, `entities`, `aspects`, `adjectives`, `relationships` |
| `arabic_enriched_kg.json` | Arabic KG — keys: `meta`, `domains`, `entities`, `aspects`, `clues`, `edges` |

Both are directed, typed graphs over four vertex kinds — domain, entity, aspect
and clue — connected by `describes_aspect`, `belongs_to` and `is_relevant_to`
relations. Inference retrieves the aspects a matched clue can describe, filters
them for consistency with the inferred sentence domain, and scores the survivors
by similarity to both the clue and the sentence.

Each graph is built **exclusively from the official training split** of its
datasets. No test-split text or labels are used at any stage of construction, so
no test information leaks into evaluation.

---

## Evaluation

Two metrics are reported:

- **Metric A — explicit aspect extraction.** Micro-averaged precision, recall and
  F₁ over gold aspect terms.
- **Metric B — implicit aspect identification.** Evaluated on the KG track. HAAD
  is excluded here: its test split contains no implicit annotations.

Significance is assessed with **paired bootstrap resampling** (n = 10,000) and
**McNemar's test**. Reporting both is deliberate — a high bootstrap *p*-value
alongside a significant McNemar result indicates a component that redistributes
errors without changing the aggregate score.

---

## Citation

```bibtex
@phdthesis{alawwad_implicit_aspect,
  author = {Alawwad, Lujain},
  title  = {Implicit Aspect Extraction on Arabic Language Based on
            Transformers and Text Paraphrasing},
  school = {King Saud University},
  year   = {2026}
}
```

---

## License and terms of use

The code in this repository is released for academic and research use.

The dataset splits are derived from third-party benchmarks, each governed by its
own license — including GPL-2.0 for HAAD and the SemEval task terms for the
SemEval corpora. Anyone reusing them should consult the original releases linked
above and comply with those terms.
