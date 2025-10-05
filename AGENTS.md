# Repository Guidelines

## Project Structure & Module Organization
Core streaming logic lives in `generator.py`, which wraps the CSM 1B model and exposes the `Segment` dataclass for conditioning. `main.py` runs the FastAPI realtime companion and coordinates `llm_interface.py`, `rag_system.py`, and `vad.py`. Front-end assets sit in `static/` (JS controllers) and `templates/` (Jinja views). Store LoRA checkpoints under `finetuned_model/`; place raw finetuning audio in `audio_data/`. Persistent settings land in `config/`, and the runtime SQLite store `companion.db` is created at the repo root. Keep downloaded weights and large audio traces out of version control.

## Build, Test, and Development Commands
- `python3.10 -m venv .venv && source .venv/bin/activate` — create and enter the recommended virtualenv.
- `pip install -r requirements.txt` — install GPU-aware dependencies (match CUDA 12.x with your driver).
- `python setup.py` — prepare supporting folders, redirect templates, and optional model caches.
- `python main.py` — launch the realtime web app at `http://localhost:8000`.
- `python run_csm.py` — run the scripted Hugging Face conversation demo.
- `python lora.py` — start LoRA finetuning after adjusting the constants at the top.

## Coding Style & Naming Conventions
Target Python 3.10+, PEP 8 spacing, and snake_case file/function names. Preserve existing type hints and dataclasses and introduce a module-level `logger = logging.getLogger(__name__)` when adding new files. Keep asynchronous FastAPI handlers explicit about threading or GPU locks, and comment only where behaviour is non-obvious. Front-end scripts follow ES modules; keep DOM IDs kebab-case.

## Testing Guidelines
`python test.py` exercises end-to-end streaming on CUDA hardware and downloads reference clips; run it only when GPUs and Hugging Face access are available. For faster checks, mock `load_csm_1b_local` and place targeted unit tests in a future `tests/` package (use `test_<module>.py` naming). Document CUDA driver, model revisions, and any generated audio paths in PRs, and prune large WAVs or move them into `responses/` before committing.

## Commit & Pull Request Guidelines
Recent commits use short imperative summaries (for example, `Update main.py`). Prefer `<area>: <imperative>` such as `generator: reduce decode latency`, reference related issues, and call out configuration or schema changes. Pull requests should explain behavioural impact, list verification commands, and attach screenshots or sample audio paths when UI or speech output changes.

## Security & Configuration Tips
Avoid committing personal Hugging Face tokens, `config/app_config.json`, `companion.db`, or downloaded checkpoints. Authenticate locally with `huggingface-cli login`, rely on environment variables for API keys, and scrub sensitive audio before sharing reproduction bundles.
