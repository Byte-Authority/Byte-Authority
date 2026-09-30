<h1 align="center">
  <img src="assets/byte-authority-logo-stacked.png" alt="Byte Authority logo" width="330" />
  <br />
  <span style="color:#18324A;">Same Bytes, Different Authority</span>
</h1>

<p align="center">Reserved-token representations in chat-template prompt injection</p>

<p align="center">
  <a href="https://github.com/Byte-Authority/Byte-Authority"><img src="https://img.shields.io/badge/Code-GitHub-2E73B5.svg?logo=github" alt="Code: GitHub" /></a>
  <a href="https://huggingface.co/datasets/YanZhanPKU/Byte-Authority-Evaluation"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-F0C75E.svg?logo=huggingface&logoColor=111827" alt="Dataset: Hugging Face" /></a>
  <a href="https://arxiv.org/abs/2609.35932"><img src="https://img.shields.io/badge/arXiv-2609.35932-C55252.svg" alt="arXiv: 2609.35932" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=19&duration=2800&pause=900&color=5FA34E&center=true&vCenter=true&width=900&lines=Byte+Authority;Same+bytes%2C+different+token+IDs;Reserved+representations+in+agent+prompt+injection" alt="Byte Authority animated tagline" />
</p>

<p align="center"><a href="https://arxiv.org/abs/2609.35932">Paper</a> · <a href="https://github.com/Byte-Authority/Byte-Authority">Code</a> · <a href="https://hf.co/collections/YanZhanPKU/byte-authority">Hugging Face collection</a></p>

## 📣 Latest news

- **2026-09-29:** Code and compact evaluation records are available on GitHub and Hugging Face.
- **2026-09-28:** [Same Bytes, Different Authority](https://arxiv.org/abs/2609.35932) is available on arXiv.

## 💡 Overview

An injected chat-template marker can be decoded as one reserved control token or as ordinary subword tokens. The two encodings produce the same text, while the model receives different token IDs. Byte Authority packages the runners and compact records used to measure this representation-level contrast under fixed bytes, matched token counts, and multi-turn agent execution.

<p align="center">
  <img src="assets/figures/fig1_design.png" alt="Same bytes, different authority: reserved-token and split-token encodings" width="96%" />
</p>

The central comparison keeps the injected string fixed. `reserved` preserves the model's reserved IDs; `split` encodes the same marker text with ordinary subwords; `matched` keeps the reserved IDs while placing the extra tokens elsewhere in the tool response. This separates the reserved representation from the cost of a longer token sequence.

```text
same decoded bytes
        ├── reserved control IDs  →  learned control-token representation
        └── ordinary subwords     →  same text, different representation
```

### 🔎 Evidence map

| Question | Release artifact |
| --- | --- |
| Does the byte-identical encoding change attack outcomes? | `data/results/main/` and `assets/figures/fig_position.pdf` |
| Does the effect survive full agent execution? | `data/results/agentdojo/` and `assets/figures/fig_deployment.pdf` |
| Does swapping only the marker input vector reproduce the contrast? | `data/results/identity/` and `assets/figures/fig_mechanism.pdf` |
| What happens under alternate forged payloads and searched spellings? | `data/results/payload/`, `data/results/search/` |
| Which tokenizer configurations leave tool-protocol tokens outside the standard mitigation? | `data/catalogs/` and `assets/figures/fig_census.pdf` |

## 📦 What is released

The package is intentionally limited to the main experiment artifacts. It contains the span-scoped encoding runners, invariant checks, compact per-case JSONL records, paper figures, source catalogs, and upstream license notices. Generated model text and runtime traces are omitted.

| Path | Contents | Records |
| --- | --- | ---: |
| `data/results/main/` | Primary InjecAgent configurations | 50,000 |
| `data/results/agentdojo/` | Multi-turn AgentDojo runs | 6,849 |
| `data/results/identity/` | Input-vector replacement experiment | 6,120 |
| `data/results/payload/` | Alternate forged-payload checks | 18,360 |
| `data/results/search/` | Calibration and held-out adaptive-search records | 95,986 |

No trained checkpoint is included. The experiments use upstream open-weight checkpoints and benchmark inputs that must be obtained under their own licenses; their sources are recorded in [`data/sources.json`](data/sources.json) and [`licenses/`](licenses/).

## 🔧 Installation

```bash
git clone https://github.com/Byte-Authority/Byte-Authority.git
cd Byte-Authority
python -m pip install -r requirements.txt
```

The runners use a local tokenizer/checkpoint and a vLLM-compatible completion endpoint. Check the actual interfaces before running:

```bash
python src/injecagent_runner.py --help
python src/identity_swap_runner.py --help
python src/agentdojo_bringup.py --help
```

## 🚀 Quickstart

Place local copies of the InjecAgent and AgentDojo inputs under `data/third_party/` as described in [`REPRODUCTION_PACKAGE.md`](REPRODUCTION_PACKAGE.md). A smoke run has this shape; replace the paths, model name, and endpoint with local values:

```bash
python src/injecagent_runner.py \
  --family qwen \
  --tokenizer /path/to/Qwen3-8B \
  --data data/third_party/injecagent/data/test_cases_dh_base.json \
  --tools data/third_party/injecagent/data/tools.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-8B \
  --arms reserved,split,matched,plaintext \
  --n 50 \
  --out data/results/local_smoke.jsonl
```

A record is written only after byte identity, protected-token placement, and returned prompt-ID checks pass. Do not use real secrets or private tool credentials in a local benchmark copy.

## 🗂️ Repository layout

| Path | Role |
| --- | --- |
| `src/` | Prompt construction, span-scoped encoding, token-count controls, identity replacement, AgentDojo transport, and marker search. |
| `data/results/` | Curated compact records from the primary and supporting experiments. |
| `data/catalogs/` | Candidate mappings and tokenizer census metadata. |
| `data/manifests/` | Split and invariant manifests. |
| `assets/figures/` | Paper-derived figures in PDF plus the overview PNG. |
| `assets/byte-authority-logo-*.png` | Stacked transparent, horizontal transparent, and stacked white-background logo lockups. |

## 🤗 Data and project links

- Dataset: [Byte-Authority-Evaluation](https://huggingface.co/datasets/YanZhanPKU/Byte-Authority-Evaluation)
- Collection: [Byte Authority](https://hf.co/collections/YanZhanPKU/byte-authority)
- Code: [Byte-Authority/Byte-Authority](https://github.com/Byte-Authority/Byte-Authority)
- Paper: [arXiv:2609.35932](https://arxiv.org/abs/2609.35932)

## 📄 Citation

```bibtex
@misc{zhan2026samebytesdifferentauthority,
  title={Same Bytes, Different Authority: Reserved-Token Representations in Chat-Template Prompt Injection},
  author={Yan Zhan and Yunze Song and Mengkai Hou and Wanting Zhang and Shaobo Liu and Zhijun Gao},
  year={2026},
  eprint={2609.35932},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2609.35932}
}
```

## 📜 License

The project code, evaluation records, and artwork are released under the [Apache License 2.0](LICENSE). Upstream benchmark licenses are preserved under [`licenses/`](licenses/); model checkpoints and benchmark inputs retain their original terms.
