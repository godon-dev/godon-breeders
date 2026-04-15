# godon-breeders

Autonomous breeder agents for optimization using metaheuristic search.

## Architecture

Breeders are self-driving optimization agents that use **Optuna ask/tell pattern** for parameter search - further metaheuristics frameworks may follow. Effectuation and reconnaissance are executed as Windmill scripts on target systems.

The system follows an **engine + strains** architecture: the engine provides the generic optimization loop (algorithm diversity, guardrails, rollback, cooperation, metrics), while strains encapsulate domain-specific knowledge (parameter suggestion, validation).

### Engine (`engine/`)

- **BreederWorker**: Generic optimization agent with lifecycle management, algorithm diversity across parallel workers, guardrail checking, and rollback support
- **Communication**: Cooperative trial sharing between breeders via Optuna database (probabilistic, best, worst, extremes strategies)
- **BreederMetricsClient**: Prometheus metrics pushing via Push Gateway
- **Strain Loader**: Dynamic loading and contract validation of strain modules

### Strains (`strains/`)

Each strain provides domain-specific logic as a pluggable module:
- `suggest_params(trial, settings)` — parameter suggestion for Optuna trials
- `validate_config(config)` — configuration validation (preflight checks)

### Effectuation (`effectuation/`)

Scripts that apply parameter changes to target systems. Each script follows the `(context, targets, settings)` interface contract:
- `context` — static breeder run configuration (credentials, URLs, playbook paths)
- `targets` — list of target systems to apply changes to
- `settings` — the optimizer's parameter suggestions for this trial

Available effectuators:
- **SSH** — applies configuration via Ansible playbooks over SSH
- **HTTP** — applies configuration via HTTP API calls (draft)

### Reconnaissance (`reconnaissance/`)

Scripts that gather metrics to evaluate trial outcomes. Same `(context, targets, settings)` interface contract. Supports:
- **Prometheus** — multi-sample collection, stabilization waits, and aggregation
- **HTTP** — metric collection via HTTP GET with configurable stabilization, multi-sampling, and aggregation

## Available Strains

### linux_performance (`strains/linux_performance/`)
Optimizes Linux system parameters (sysctl, sysfs, cpufreq, ethtool) for improved performance. Supports network, memory, CPU, and custom optimization objectives via Prometheus metrics.

### bench_greenhouse (`strains/bench_greenhouse/`)
Optimizes greenhouse climate simulation parameters. Supports multi-zone heating, ventilation, shading, CO2 injection, lighting, and irrigation. Designed for the `godon-bench-greenhouse` simulation container with HTTP-based effectuation and reconnaissance.

## License

AGPL-3.0
