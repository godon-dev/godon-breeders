# godon-breeders

Autonomous breeder agents for optimization using metaheuristic search.

## Architecture

Breeders are self-driving optimization agents that use **Optuna ask/tell pattern** for parameter search - further metaheuristics frameworks may follow. All coordinated via Windmill flows and executed on target systems.

The system follows an **engine + strains** architecture: the engine provides the generic optimization loop (algorithm diversity, guardrails, rollback, cooperation, metrics), while strains encapsulate domain-specific knowledge (parameter suggestion, effectuation, validation).

### Engine (`engine/`)

- **BreederWorker**: Generic optimization agent with lifecycle management, algorithm diversity across parallel workers, guardrail checking, and rollback support
- **Communication**: Cooperative trial sharing between breeders via Optuna database (probabilistic, best, worst, extremes strategies)
- **BreederMetricsClient**: Prometheus metrics pushing via Push Gateway
- **Strain Loader**: Dynamic loading of strain modules at runtime

### Strains (`strains/`)

Each strain provides domain-specific logic as a pluggable module:
- `suggest_params(trial, settings)` — parameter suggestion for Optuna trials
- `validate_config(config)` — configuration validation (preflight checks)
- `EFFECTUATION_FLOW` — path to the Windmill effectuation flow

## Available Strains

### linux_performance (`strains/linux_performance/`)
Optimizes Linux system parameters (sysctl, sysfs, cpufreq, ethtool) for improved performance. Supports network, memory, CPU, and custom optimization objectives via Prometheus metrics.

## License

AGPL-3.0
