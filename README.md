# godon-breeders

Autonomous breeder agents for optimization using metaheuristic search.

## Architecture

Breeders are self-driving optimization agents that use **Optuna ask/tell pattern** for parameter search for now - further metaheuristics frameworks may follow. All coordinated via Windmill flows and executed on target systems.

### Components

- **BreederWorker**: Autonomous optimization agent with internal lifecycle management
- **Effectuation**: Parameter application strategies (SSH, HTTP, APIs)
- **Reconnaissance**: Metric gathering (Prometheus, custom sources)
- **Communication**: Cooperative trial sharing between breeders via Optuna database

## Available Breeders

### linux_network_stack
Optimizes Linux network parameters (sysctl) for improved TCP performance.

## License

AGPL-3.0
