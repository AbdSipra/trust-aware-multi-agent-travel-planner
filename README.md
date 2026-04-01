# Trust-Aware Multi-Agent Travel Planner

This repository contains an agentic AI research project for constrained travel
planning with deterministic tools, multi-agent coordination, and a trust-aware
propagation layer for unreliable or corrupted intermediate information.

## Workspace Layout

- `project_code/` runnable code, datasets, experiments, and tests
- `paper/` manually written IEEE-style paper assets
- `poster/` poster assets

## Baseline Workflow

1. Normalize source-grounded local datasets.
2. Run the single-agent baseline on development tasks.
3. Run the naive multi-agent baseline on clean evaluation tasks.
4. Run the trust-aware multi-agent system on attacked evaluation tasks.
5. Compare metrics and inspect saved traces in `project_code/data/runs/`.

## Quick Start

1. Copy `.env.example` to `.env` and fill values if you want Ollama or Groq.
2. Install dependencies from `requirements.txt`.
3. Build the source-grounded benchmark from downloaded public-source snapshots:

```powershell
python project_code/scripts/normalize_source_grounded_data.py
```

4. If you need a fallback fully synthetic benchmark, you can still run:

```powershell
python project_code/scripts/generate_seed_data.py
```

5. Run a baseline experiment:

```powershell
python project_code/run_experiment.py --task-split dev_tasks --system-variant single_agent_tool_use
```

## Notes

- The repository is initialized locally with Git. Create a GitHub remote from
  your own account when you are ready to publish the repo.
- The paper must be written manually. LLMs are used only inside the system.
- Source-grounded raw files are stored in `project_code/data/source_grounded/`.
- Saved experiment traces are written under `project_code/data/runs/`.
