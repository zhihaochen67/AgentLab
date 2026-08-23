# AgentLab

Evaluation and observability platform for LLM agents.

AgentLab provides a lightweight framework for benchmarking, tracing, and analyzing autonomous agents through reproducible tasks, execution traces, and experiment comparisons.

## Overview

AgentLab helps answer:

- How capable is an AI agent?
- Which agent configuration performs better?
- Where does an agent fail during execution?
- How do different prompts or models affect results?

Instead of evaluating agents only by final answers, AgentLab records the complete execution process and provides structured evaluation.

## Features

- **Agent adapters**
  - Unified interface for different agent implementations.
  - Currently supports Repo Doctor evaluation.

- **Benchmark-driven evaluation**
  - Run agents against reproducible task datasets.
  - Measure success rate and execution outcomes.

- **Execution traces**
  - Track agent runs, steps, and results.
  - Store structured evaluation records.

- **Experiment comparison**
  - Compare different agents, prompts, or configurations.

- **Diagnostics and observability**
  - Analyze agent behavior beyond final success/failure.

- **Dashboard visualization**
  - Explore experiments and evaluation results.

## Architecture

```text
Task Dataset
      |
      v
AgentLab Runner
      |
      v
Agent Adapter
      |
      v
Agent Under Evaluation
      |
      v
Trace + Metrics
      |
      v
Experiment Analysis
```

## Repo Doctor Integration

AgentLab can evaluate Repo Doctor as an autonomous coding repair agent.

Example workflow:

```text
Dataset Task
      |
      v
AgentLab
      |
      v
Repo Doctor
      |
      v
Repair Result
      |
      v
Evaluation Metrics
```

This allows measuring:

- repair success rate
- verification results
- execution traces
- experiment differences

## Development

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest -q
```

Run lint:

```bash
uv run ruff check .
```

## Testing

Current status:

- 98 tests passed

## License

MIT