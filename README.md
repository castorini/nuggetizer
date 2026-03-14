# nuggetizer <img src="docs/nuggetizer-logo.png" width="300" />

[![PyPI](https://img.shields.io/pypi/v/nuggetizer?color=brightgreen)](https://pypi.org/project/nuggetizer/)
[![Downloads](https://static.pepy.tech/personalized-badge/nuggetizer?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nuggetizer)
[![Downloads](https://static.pepy.tech/personalized-badge/nuggetizer?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/week)](https://pepy.tech/project/nuggetizer)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![paper](https://img.shields.io/badge/paper-arxiv-blue.svg?style=flat)](https://arxiv.org/abs/2411.09607)

A powerful tool for information nugget creation, scoring, and assigning to RAG answers using LLMs.
Enables the evaluation of fact recall of RAG answers.

## 📟 Installation

### Install `uv`

Install `uv` with Astral's official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If `uv` already works in your shell, you can skip this step. Otherwise, restart your shell or add `uv` to the current shell session:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Install from PyPI

Create an isolated virtual environment and install the published package:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install nuggetizer
```

### Development Installation

For development or the latest features, install from source with the development toolchain:

```bash
git clone https://github.com/castorini/nuggetizer.git
cd nuggetizer
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev
```

If you prefer not to activate the virtual environment, run commands through `uv run`, for example:

```bash
uv run nuggetizer --help
uv run pre-commit run --all-files
```

### Environment Setup

Create a `.env` file with your API credentials. Nuggetizer supports multiple API providers:

**Azure OpenAI (optional, enable explicitly with `use_azure_openai=True` or `--use-azure-openai`):**
```bash
AZURE_OPENAI_API_BASE=your_azure_endpoint
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_API_KEY=your_api_key
```

**OpenAI API:**
```bash
OPENAI_API_KEY=your_openai_api_key
```

**OpenRouter API:**
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
```

**vLLM Local Server:**
No environment variables needed. vLLM runs locally and doesn't require authentication.

**Note:** Nuggetizer supports multiple API providers. If both OpenAI and OpenRouter keys are available, OpenAI will be used by default. You can explicitly use OpenRouter by passing the `use_openrouter=True` parameter to the Nuggetizer constructor or using the `--use_openrouter` flag in the examples. For vLLM, use `use_vllm=True` and optionally specify `vllm_port` (default: 8000).

## 🚀 Quick Start

The canonical command-line interface is `nuggetizer ...`. If your environment is
activated, run commands directly. If it is not, the development fallback is
`uv run nuggetizer ...`.

### CLI Quick Start

Create nuggets from a batch JSONL file:

```bash
nuggetizer create \
  --input-file pool.jsonl \
  --output-file nuggets.jsonl \
  --model gpt-4o-mini
```

Create nuggets from a direct single-object JSON payload without `qid` or `docid`:

```bash
nuggetizer create \
  --input-json '{"query":"What is Python used for?","candidates":["Python is widely used for web development and data analysis.","Python is also used for automation and machine learning."]}' \
  --output json
```

Assign nuggets to RAG answers:

```bash
nuggetizer assign \
  --input-kind answers \
  --nuggets nuggets.jsonl \
  --contexts answers.jsonl \
  --output-file assignments.jsonl
```

Assign nuggets to a direct single context:

```bash
nuggetizer assign \
  --input-json '{"query":"What is Python used for?","context":"Python is commonly used for web development and data analysis.","nuggets":[{"text":"Python is used for web development.","importance":"vital"},{"text":"Python is used for data analysis.","importance":"okay"}]}' \
  --output json
```

Calculate metrics:

```bash
nuggetizer metrics \
  --input-file assignments.jsonl \
  --output-file metrics.jsonl
```

Inspect the CLI contract:

```bash
nuggetizer describe assign --output json
nuggetizer schema assign-output-answers --output json
nuggetizer validate create --input-json '{"query":"q","candidates":["p"]}' --output json
nuggetizer doctor --output json
nuggetizer view assignments.jsonl --records 1
```

Legacy `scripts/*.py` entrypoints are still supported as compatibility wrappers
around the packaged CLI, but new automation and documentation should prefer
`nuggetizer ...`.

### CLI For Automation

- Use `--output json` for automation; that is the authoritative machine-readable interface.
- `nuggetizer doctor --output json` reports command and backend readiness with explicit `ready`, `missing_env`, or `blocked` states.
- `nuggetizer describe ...` and `nuggetizer schema ...` expose the supported command metadata and payload contracts without running models.
- `nuggetizer validate ...` is non-mutating and returns a real pass or fail instead of performing any repair work.

## Contributing

Contributor setup, local quality gates, and pull request expectations are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

Here's a simple example of how to use nuggetizer:

```python
from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer

# Create a sample request
query = Query(qid="1", text="What are the main features of Python?")
documents = [
    Document(
        docid="1",
        segment="""Python is a high-level programming language known for its 
        simplicity and readability. It supports multiple programming paradigms 
        including procedural, object-oriented, and functional programming."""
    ),
    Document(
        docid="2",
        segment="""Python was created by Guido van Rossum in 1991."""
    ),
    Document(
        docid="3",
        segment="""Python is widely used in web development, data analysis, 
        artificial intelligence, and scientific computing."""
    ),
]
request = Request(query=query, documents=documents)

# Option 1: Single model for all components
nuggetizer = Nuggetizer(model="gpt-4o")  # Uses same model for all components

# Option 2: Different models for each component
nuggetizer_mixed = Nuggetizer(
    creator_model="gpt-4o",  # Model for nugget creation
    scorer_model="gpt-3.5-turbo",  # Model for nugget scoring
    assigner_model="gpt-4o"  # Model for nugget assignment
)

# Option 3: Using OpenRouter API (supports multiple providers)
nuggetizer_openrouter = Nuggetizer(
    model="x-ai/grok-4-fast",  # Grok model via OpenRouter
    use_openrouter=True  # Explicitly use OpenRouter
)

# Option 4: Other OpenRouter models
nuggetizer_claude = Nuggetizer(
    model="anthropic/claude-3.5-sonnet",  # Claude via OpenRouter
    use_openrouter=True  # Explicitly use OpenRouter
)

# Option 5: Using vLLM local server
nuggetizer_vllm = Nuggetizer(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",  # Model name as served by vLLM
    use_vllm=True,  # Use vLLM local server
    vllm_port=8000  # Optional: specify port (default: 8000)
)

# Create and score nuggets
scored_nuggets = nuggetizer.create(request)

# Print nuggets and their importance
for nugget in scored_nuggets:
    print(f"Nugget: {nugget.text}")
    print(f"Importance: {nugget.importance}\n")

# Assign nuggets to a specific document
assigned_nuggets = nuggetizer.assign(query.text, documents[0].segment, scored_nuggets)

# Print assignments
for nugget in assigned_nuggets:
    print(f"Nugget: {nugget.text}")
    print(f"Importance: {nugget.importance}")
    print(f"Assignment: {nugget.assignment}\n")
```

You can also run the default end-to-end example with:
```bash
python3 examples/pipeline_demo.py
```

The default `examples/pipeline_demo.py` uses the async Nuggetizer pipeline. If you want the synchronous version instead, run:
```bash
python3 examples/sync_pipeline_demo.py
```

For an opt-in live smoke test that exercises the packaged CLI against a real
OpenAI-compatible backend, run:

```bash
NUGGETIZER_LIVE_OPENAI_SMOKE=1 uv run pytest tests/test_live_openai_smoke.py
```

**Running with OpenRouter API:**
You can use OpenRouter API to access multiple model providers:
```bash
# Set OpenRouter API key in environment
export OPENROUTER_API_KEY=your_openrouter_api_key

# Use Grok model (free tier) with OpenRouter
python3 examples/pipeline_demo.py --model "x-ai/grok-4-fast" --use_openrouter

# Use Claude model with OpenRouter
python3 examples/pipeline_demo.py --model "anthropic/claude-3.5-sonnet" --use_openrouter

# Use OpenAI models via OpenRouter
python3 examples/pipeline_demo.py --model "openai/gpt-4o-mini" --use_openrouter
```

Or create a `.env` file with your OpenRouter API key:
```bash
echo "OPENROUTER_API_KEY=your_openrouter_api_key" > .env
python3 examples/pipeline_demo.py --model "x-ai/grok-4-fast" --use_openrouter
```

**Running with vLLM Local Server:**
You can use vLLM to run models locally:
```bash
# Use vLLM with default port (8000)
python3 examples/pipeline_demo.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507" --use_vllm

# Use vLLM with custom port (8001)
python3 examples/pipeline_demo.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507" --use_vllm --vllm_port 8001
```

The default `examples/pipeline_demo.py` uses async methods such as `async_create` and `async_assign`. To run it, use:

```bash
python3 examples/pipeline_demo.py
```

**Running async example with OpenRouter API:**
```bash
# Set OpenRouter API key in environment
export OPENROUTER_API_KEY=your_openrouter_api_key


# Use Claude model with OpenRouter
python3 examples/pipeline_demo.py --model "anthropic/claude-3.5-sonnet" --use_openrouter
```

**Running async example with vLLM:**
```bash
# Use vLLM with default port (8000)
python3 examples/pipeline_demo.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507" --use_vllm

# Use vLLM with custom port (8001)
python3 examples/pipeline_demo.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507" --use_vllm --vllm_port 8001
```

## 🛠️ Components

The Nuggetizer class provides a unified interface for:

1. **Nugget Creation & Scoring**: Extracts and scores atomic information nuggets from text
2. **Nugget Assignment**: Assigns nuggets to specific texts

The following CLI commands help run the TREC 2024 RAG Track workflow:

1. First, generate nuggets:
```bash
nuggetizer create \
  --input-file pool.jsonl \
  --output-file nuggets.jsonl \
  --log-level 1
```

2. For RAG answers, we assume they take on the format laid out by the wonderful [TREC 2024 RAG Track](https://trec-rag.github.io/annoucements/2024-track-guidelines/):

```json
{
    "run_id": "ragnarok",
    "topic_id": "2027497",
    "topic": "how often should you take your toddler to the potty when potty training",
    "references": [
        "msmarco_v2.1_doc_51_766815931#2_1606878413", "msmarco_v2.1_doc_51_766815931#1_1606876582", "msmarco_v2.1_doc_51_766815931#5_1606882767", 
        "msmarco_v2.1_doc_51_766815931#6_1606884302", "msmarco_v2.1_doc_51_766815931#3_1606879951", "msmarco_v2.1_doc_51_766815931#4_1606881348", 
        "msmarco_v2.1_doc_37_463237391#10_984448281", "msmarco_v2.1_doc_51_766815931#0_1606874600", "msmarco_v2.1_doc_37_463237391#9_984446615", 
        "msmarco_v2.1_doc_28_472446307#22_1012988885", "msmarco_v2.1_doc_51_766815931#7_1606885873", "msmarco_v2.1_doc_28_472446307#21_1012986800", 
        "msmarco_v2.1_doc_29_562342450#23_1356565296", "msmarco_v2.1_doc_29_562342450#17_1356555947", "msmarco_v2.1_doc_49_418787959#7_861728734", 
        "msmarco_v2.1_doc_49_418787959#6_861726964", "msmarco_v2.1_doc_26_680625866#7_1289507527", "msmarco_v2.1_doc_10_1346272776#19_2165266355", 
        "msmarco_v2.1_doc_56_1491300640#3_3012150696", "msmarco_v2.1_doc_10_672519892#5_1260010758"], 
    "response_length": 192, 
    "answer": [
            {"text": "The frequency with which you should take your toddler to the potty depends on their readiness for potty training.", "citations": [0, 1, 12, 13, 19]}, 
            {"text": "Some sources suggest that toddlers should be taken to the potty about three times a day: first thing in the morning, after mealtimes, and again before bedtime.", "citations": [0, 4, 6, 8]}, 
            {"text": "It is recommended that you watch for facial expressions or poses that may signal that they need to \"go\".", "citations": [6, 8]}, 
            {"text": "If they are reluctant to use the potty, don't force them.", "citations": [6, 8]}, 
            {"text": "Other sources suggest that toddlers should be taken to the potty every two hours, whether they have to go or not.", "citations": [14, 15]}, 
            {"text": "This includes first thing in the morning, before leaving the house, and before naps and bedtime.", "citations": [14, 15]}, 
            {"text": "Some sources recommend taking toddlers to the potty every 30 minutes to an hour.", "citations": [9, 11, 17]}, 
            {"text": "This is to increase the chances of them peeing in the potty instead of on the floor.", "citations": [9, 11]}, 
            {"text": "It is important to keep in mind that every toddler is different, and their potty training journey will be unique to them.", "citations": [0, 4]}, 
            {"text": "It is recommended that you let your toddler lead the way and be gentle throughout the process, as their self-esteem can be fragile during this time.", "citations": [0, 1]}
        ]
}
```
To *easily* generate answers in this format, consider using [Ragnarök](https://github.com/castorini/ragnarok).
Let's now assign the nuggets to the RAG answers:

```bash
nuggetizer assign \
    --input-kind answers \
    --nuggets nuggets.jsonl \
    --contexts ragnarok.jsonl \
    --output-file final_assignments.jsonl

nuggetizer metrics \
    --input-file final_assignments.jsonl \
    --output-file metrics.jsonl
```

For retrieval-style per-candidate assignment, use:

```bash
nuggetizer assign \
    --input-kind retrieval \
    --nuggets nuggets.jsonl \
    --contexts retrieval.jsonl \
    --output-file retrieval_assignments.jsonl
```

Compatibility note:
The older `scripts/*.py` entrypoints still work as thin wrappers around the packaged CLI.

The final output file (`final_assignments.jsonl`) will contain:
- query: The original query
- qid: Query ID
- answer_text: Full answer text
- response_length: Response length
- run_id: Run ID (derived from the RAG answer filename)
- nuggets: Nuggets with their importance labels and assignments

The final metrics file (`metrics.jsonl`) will contain:
- Per-response metrics:
  - `strict_vital_score`: Score counting only full support for vital nuggets
  - `strict_all_score`: Score counting only full support for all nuggets
  - `vital_score`: Score counting full (1.0) and partial (0.5) support for vital nuggets
  - `all_score`: Score counting full (1.0) and partial (0.5) support for all nuggets
- Global mean metrics across all responses (indicated by `qid` as `all`)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is built with the support of Azure's OpenAI credits.

## ✨ References

If you use Nuggetizer, please cite the following relevant papers:

[[SIGIR 2025] The Great Nugget Recall: Automating Fact Extraction and {RAG} Evaluation with Large Language Models](https://dl.acm.org/doi/10.1145/3726302.3730090)

```
@inproceedings{greatnuggetrecall,
      title = "{The Great Nugget Recall}: Automating Fact Extraction and {RAG} Evaluation with Large Language Models",
      author = {Ronak Pradeep and Nandan Thakur and Shivani Upadhyay and Daniel Campos and Nick Craswell and Jimmy Lin},
      booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      year = {2025},
      address_ = {New York, NY, USA},
      publisher = {Association for Computing Machinery},
      pages = {180--190},
      numpages = {11},
      keywords_ = {atomic facts, automatic evaluation, nugget evaluation},
      location_ = {Padua, Italy},
      series = {SIGIR '25}
}
```

[[2411.09607] Initial Nugget Evaluation Results for the {TREC 2024 RAG Track} with the {AutoNuggetizer Framework}](https://arxiv.org/abs/2411.09607)

```
@ARTICLE{pradeep2024autonuggetizer,
  title   = {Initial Nugget Evaluation Results for the {TREC 2024 RAG Track} with the {AutoNuggetizer Framework}},
  author  = {Ronak Pradeep and Nandan Thakur and Shivani Upadhyay and Daniel Campos and Nick Craswell and Jimmy Lin},
  year    = {2024},
  journal = {arXiv:2411.09607}
}
```

[[2406.16828] Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track](https://arxiv.org/abs/2406.16828)
```
@ARTICLE{pradeep2024ragnarok,
  title   = {{Ragnarök}: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track},
  author  = {Ronak Pradeep and Nandan Thakur and Sahel Sharifymoghaddam and Eric Zhang and Ryan Nguyen and Daniel Campos and Nick Craswell and Jimmy Lin},
  year    = {2024},
  journal = {arXiv:2406.16828},
}
```
