#
# Copyright (c) 2019 Matthias Tafelmeier.
#
# This file is part of godon
#
# godon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# godon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this godon. If not, see <http://www.gnu.org/licenses/>.
#

import os
import optuna
import logging
import wmill
import random
import hashlib
import datetime
import dateutil.parser
import time
import psycopg2
from typing import Dict, Any, Optional, List
from optuna.trial import TrialState
from optuna.samplers import TPESampler, NSGAIISampler, NSGAIIISampler, RandomSampler, QMCSampler
from optuna.samplers.nsgaii import (
    UniformCrossover,
    UNDXCrossover,
    SPXCrossover,
    BLXAlphaCrossover,
    SBXCrossover,
    VSBXCrossover
)
from scipy.stats import percentileofscore
from f.breeder.linux_performance.breeder_metrics_client import BreederMetricsClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CommunicationCallback:
    """Shares successful trials with cooperating breeders for metaheuristic learning
    
    Supports multiple sharing strategies:
    - probabilistic: Share trials randomly with probability P
    - best: Share only top-performing trials (top percentile)
    - worst: Share only bottom-performing trials (bottom percentile)  
    - extremes: Share both top and bottom-performing trials
    """
    
    def __init__(self, storage: str, share_strategy: str = "probabilistic", 
                 probability: float = 0.8, top_percentile: float = 0.2,
                 bottom_percentile: float = 0.2, min_trials_for_filtering: int = 10,
                 share_within_breeder: bool = True):
        self.storage = storage
        self.share_strategy = share_strategy
        self.com_probability = probability
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
        self.min_trials_for_filtering = min_trials_for_filtering
        self.share_within_breeder = share_within_breeder
        self.logger = logging.getLogger('communication-callback')
        self.logger.setLevel(logging.DEBUG)
    
    def _share_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Share completed trial with all cooperating studies in database"""
        try:
            study_names = study.get_all_study_names(storage=self.storage)
            
            for study_name in study_names:
                if study_name != study.study_name:
                    # Skip sharing within same breeder if disabled
                    if not self.share_within_breeder:
                        breeder_prefix = study.study_name.split('_')[0]
                        if study_name.startswith(breeder_prefix):
                            continue
                    
                    try:
                        cooperating_study = optuna.load_study(study_name=study_name, storage=self.storage)
                        cooperating_study.add_trial(trial)
                        self.logger.info(f"Shared trial {trial.number} with {study_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to share with {study_name}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Communication failed: {e}")
    
    def _should_share_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        """Determine if trial should be shared based on strategy"""
        if self.share_strategy == "probabilistic":
            return random.random() < self.com_probability
        
        # For quality-based strategies, need enough trials for meaningful filtering
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.values]
        if len(completed_trials) < self.min_trials_for_filtering:
            self.logger.debug(f"Insufficient trials ({len(completed_trials)}) for quality filtering, sharing all")
            return True
        
        # Get trial value (use first objective for multi-objective)
        trial_value = trial.values[0] if trial.values else float('inf')
        all_values = [t.values[0] for t in completed_trials if t.values]
        
        if self.share_strategy == "best":
            percentile = percentileofscore(all_values, trial_value)
            return percentile >= (100 - self.top_percentile * 100)
        
        elif self.share_strategy == "worst":
            percentile = percentileofscore(all_values, trial_value)
            return percentile <= self.bottom_percentile * 100
        
        elif self.share_strategy == "extremes":
            percentile = percentileofscore(all_values, trial_value)
            top_threshold = 100 - self.top_percentile * 100
            bottom_threshold = self.bottom_percentile * 100
            return percentile >= top_threshold or percentile <= bottom_threshold
        
        else:
            self.logger.warning(f"Unknown strategy '{self.share_strategy}', defaulting to share")
            return True
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Trial sharing based on configured strategy"""
        if self._should_share_trial(study, trial):
            self.logger.debug(f"Sharing trial {trial.number} (strategy: {self.share_strategy})")
            self._share_trial(study, trial)
        else:
            self.logger.debug(f"Skipping trial sharing for {trial.number} (strategy: {self.share_strategy})")


class BreederWorker:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        breeder_config = config.get('breeder', {})

        self.breeder_type = breeder_config.get('name', 'unknown_breeder')
        self.breeder_uuid = breeder_config.get('uuid', breeder_config.get('name', 'unknown'))
        self.breeder_id = self.breeder_uuid  # For metadata/communication
        # Database name: breeder_ prefix + UUID with underscores (PostgreSQL compatibility)
        self.breeder_db_name = f"breeder_{self.breeder_uuid.replace('-', '_')}"
        self.worker_id = f"{self.breeder_type}_worker_{self.breeder_uuid}"

        # Parse creation timestamp for completion criteria
        creation_ts_str = config.get('creation_ts')
        if not creation_ts_str:
            raise ValueError("Required field 'creation_ts' missing from config")
        self.start_time = dateutil.parser.parse(creation_ts_str)

        # Assign sampler to this worker for algorithm diversity
        self.sampler_type = self._assign_sampler()

        self.study = self._load_or_create_study()
        self.communication_callback = self._setup_communication()

        # Rollback state tracking
        # Extract worker identifiers from config
        self.run_id = config.get('run_id', 0)
        self.target_id = config.get('target_id', 0)

        # Get rollback configuration for this worker's target
        targets = self.config.get('effectuation', {}).get('targets', [])
        if 0 <= self.target_id < len(targets):
            self.target = targets[self.target_id]
        else:
            self.target = targets[0] if targets else {}
            logger.warning(f"Invalid target_id {self.target_id}, using first target")

        self.rollback_config = self.target.get('rollback', {})
        self.rollback_enabled = self.rollback_config.get('enabled', False)

        if self.rollback_enabled:
            logger.info(f"Rollback enabled for target {self.target_id}")
            logger.info(f"Rollback strategy: {self.rollback_config.get('strategy', 'unknown')}")
            # Initialize rollback state in study if not exists
            self._init_rollback_state()

        self._update_state()

        # Initialize metrics client
        self.metrics = BreederMetricsClient(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type
        )
        
    def _assign_sampler(self) -> str:
        """Assign a sampler type to this worker for algorithm diversity"""
        parallel_workers = self.config.get('run', {}).get('parallel', 1)
        
        if parallel_workers <= 1:
            logger.info("Single worker mode, using default TPE sampler")
            return 'tpe'
        
        # Automatically enable algorithm diversity for parallel workers
        # Only use as many samplers as we have workers
        all_samplers = ['tpe', 'nsga2', 'random', 'nsga3', 'qmc']
        num_samplers = min(parallel_workers, len(all_samplers))
        available_samplers = all_samplers[:num_samplers]
        
        # Assign sampler based on worker_id hash for consistent assignment
        worker_hash = int(hashlib.md5(self.worker_id.encode()).hexdigest(), 16)
        sampler_index = worker_hash % len(available_samplers)
        assigned_sampler = available_samplers[sampler_index]
        
        logger.info(f"Algorithm diversity auto-enabled ({len(available_samplers)} samplers for {parallel_workers} workers): Worker {self.worker_id} assigned '{assigned_sampler}' sampler")
        return assigned_sampler
    
    def _create_sampler(self, sampler_type: str) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler instance with randomized parameter selection from known-good profiles
        
        NOTE: This is for algorithm diversity + future hyperheuristic optimization.
        Parameter ranges are based on Optuna documentation and common practices.
        Some ranges are more researched than others - marked with confidence levels below.
        """
        
        # Define parameter profiles with confidence levels
        # HIGH = Well-researched, documented defaults and variations
        # MEDIUM = Educated guesses from GA/BO literature, less certain
        # LOW = Uncertain, needs empirical validation
        sampler_profiles = {
            'tpe': {
                # Valid (multivariate, group) combinations - Optuna requires multivariate=True when group=True
                'multivariate_group': [(True, True), (True, False), (False, False)],  # HIGH
                'constant_liar': [True, False],  # HIGH - Proven for parallel optimization
                'n_startup_trials': [5, 10, 20]  # MEDIUM - Educated guess, needs validation
            },
            'nsga2': {
                'population_size': [30, 50, 75, 100, 125, 150],  # Safe min=30, max=150 with 6 diverse steps
                'mutation_prob': [0.05, 0.1, 0.15],  # LOW - Educated guess from GA literature
                'crossover_prob': [0.8, 0.9, 0.95],  # MEDIUM - Common GA values, less certain for Optuna
                'crossover': ['uniform', 'UNDX', 'SPX', 'BLXAlpha', 'SBX', 'VSBX']  # All Optuna NSGA-II crossovers
            },
            'nsga3': {
                'population_size': [50, 100]  # HIGH - Safe variations on default
            },
            'random': {
                'seed': [None]  # HIGH - Placeholder for future seed diversity
            },
            'qmc': {
                'seed': [None]  # HIGH - Placeholder for future seed diversity
            }
        }
        
        if sampler_type == 'tpe':
            profile = sampler_profiles['tpe']
            multivariate, group = random.choice(profile['multivariate_group'])

            config = {
                'multivariate': multivariate,
                'group': group,
                'constant_liar': random.choice(profile['constant_liar']),
                'n_startup_trials': random.choice(profile['n_startup_trials'])
            }
            logger.info(f"Created TPE sampler with config: {config}")
            return TPESampler(**config)
        
        elif sampler_type == 'nsga2':
            profile = sampler_profiles['nsga2']
            population_size = random.choice(profile['population_size'])
            crossover_name = random.choice(profile['crossover'])

            # Instantiate crossover objects (not strings)
            # NOTE: UNDX and SPX require population_size >= n_parents (3)
            if crossover_name == 'uniform':
                crossover_obj = UniformCrossover()
            elif crossover_name == 'UNDX':
                population_size = max(population_size, 3)  # UNDX needs 3+ parents
                crossover_obj = UNDXCrossover()
            elif crossover_name == 'SPX':
                population_size = max(population_size, 3)  # SPX needs 3+ parents
                crossover_obj = SPXCrossover()
            elif crossover_name == 'BLXAlpha':
                crossover_obj = BLXAlphaCrossover()
            elif crossover_name == 'SBX':
                crossover_obj = SBXCrossover()
            elif crossover_name == 'VSBX':
                crossover_obj = VSBXCrossover()
            else:
                logger.warning(f"Unknown crossover '{crossover_name}', falling back to UniformCrossover")
                crossover_obj = UniformCrossover()

            config = {
                'population_size': population_size,
                'mutation_prob': random.choice(profile['mutation_prob']),
                'crossover_prob': random.choice(profile['crossover_prob']),
                'crossover': crossover_obj
            }
            logger.info(f"Created NSGAII sampler with crossover={crossover_name}, config: {config}")
            return NSGAIISampler(**config)
        
        elif sampler_type == 'nsga3':
            profile = sampler_profiles['nsga3']
            population_size = random.choice(profile['population_size'])
            config = {'population_size': population_size}
            logger.info(f"Created NSGAIII sampler with config: {config}")
            return NSGAIIISampler(**config)
        
        elif sampler_type == 'random':
            seed = random.choice(sampler_profiles['random']['seed'])
            config = {'seed': seed} if seed is not None else {}
            logger.info(f"Created Random sampler with config: {config}")
            return RandomSampler(**config)
        
        elif sampler_type == 'qmc':
            seed = random.choice(sampler_profiles['qmc']['seed'])
            config = {'seed': seed} if seed is not None else {}
            logger.info(f"Created QMC sampler with config: {config}")
            return QMCSampler(**config)
        
        else:
            logger.warning(f"Unknown sampler '{sampler_type}', falling back to TPE")
            return TPESampler()
    
    def _get_db_url(self) -> str:
        db_config = {
            'user': os.environ.get("GODON_ARCHIVE_DB_USER", "postgres"),
            'password': os.environ.get("GODON_ARCHIVE_DB_PASSWORD", "postgres"),
            'host': os.environ.get("GODON_ARCHIVE_DB_SERVICE_HOST", "localhost"),
            'port': os.environ.get("GODON_ARCHIVE_DB_SERVICE_PORT", "5432"),
            'database': self.breeder_db_name  # Uses breeder_ prefix + UUID with underscores
        }
        return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    def _load_or_create_study(self) -> optuna.Study:
        # Create sampler-specific study name for algorithm diversity
        parallel_workers = self.config.get('run', {}).get('parallel', 1)
        if parallel_workers > 1:
            study_name = f"{self.breeder_id}_{self.sampler_type}_study"
        else:
            study_name = f"{self.breeder_id}_study"
        
        directions = [obj.get('direction') for obj in self.config.get('objectives', [])]
        
        try:
            storage = optuna.storages.RDBStorage(url=self._get_db_url())
            study = optuna.load_study(study_name=study_name, storage=storage)
            logger.info(f"Loaded existing study: {study_name} with {len(study.trials)} trials")
        except (KeyError, ValueError):
            storage = optuna.storages.RDBStorage(url=self._get_db_url())
            
            # Create sampler if algorithm diversity is enabled
            sampler = None
            if parallel_workers > 1:
                sampler = self._create_sampler(self.sampler_type)
                logger.info(f"Created study {study_name} with {self.sampler_type} sampler")
            
            study = optuna.create_study(
                study_name=study_name, 
                directions=directions, 
                storage=storage,
                sampler=sampler
            )
            logger.info(f"Created new study: {study_name}")
        
        return study
    
    def _setup_communication(self) -> Optional[CommunicationCallback]:
        """Setup communication callback for breeder cooperation"""
        cooperation_config = self.config.get('cooperation', {})
        parallel_workers = self.config.get('run', {}).get('parallel', 1)
        
        if cooperation_config.get('active', False):
            # Get strategy configuration (controller validates parallel > 1)
            share_strategy = cooperation_config.get('share_strategy', 'probabilistic')
            probability = cooperation_config.get('probability', 0.8)
            top_percentile = cooperation_config.get('top_percentile', 0.2)
            bottom_percentile = cooperation_config.get('bottom_percentile', 0.2)
            min_trials_for_filtering = cooperation_config.get('min_trials_for_filtering', 10)
            storage = self._get_db_url()
            
            # If algorithm diversity is enabled (parallel > 1), share within breeder (across sampler-specific studies)
            share_within_breeder = parallel_workers > 1
            
            logger.info(f"Communication enabled with strategy: {share_strategy}, share_within_breeder: {share_within_breeder}")
            if share_strategy == "probabilistic":
                logger.info(f"  Probability: {probability}")
            else:
                logger.info(f"  Top percentile: {top_percentile}, Bottom percentile: {bottom_percentile}")
                logger.info(f"  Min trials for filtering: {min_trials_for_filtering}")
            
            return CommunicationCallback(
                storage=storage,
                share_strategy=share_strategy,
                probability=probability,
                top_percentile=top_percentile,
                bottom_percentile=bottom_percentile,
                min_trials_for_filtering=min_trials_for_filtering,
                share_within_breeder=share_within_breeder
            )
        else:
            logger.info("Communication disabled")
            return None
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameter values using Optuna trial.

        v0.3 updates:
        - Support sysctl, sysfs, cpufreq, ethtool categories
        - Support list of constraints (multiple disjoint ranges)
        - Support categorical parameters (suggest_categorical)
        - Support float parameters (suggest_float)
        - Handle ethtool's nested structure (interface → parameters)
        """
        params = {}

        # Support multiple settings categories
        supported_categories = ['sysctl', 'sysfs', 'cpufreq', 'ethtool']
        settings = self.config.get('settings', {})

        for category in supported_categories:
            if category not in settings:
                continue

            category_settings = settings[category]

            # Special handling for ethtool (nested: interface → parameters)
            if category == 'ethtool':
                for interface_name, interface_config in category_settings.items():
                    if not isinstance(interface_config, dict):
                        logger.warning(f"Invalid ethtool config for {interface_name}: not a dict")
                        continue

                    for param_name, param_config in interface_config.items():
                        if 'constraints' not in param_config:
                            logger.warning(f"Missing constraints for ethtool.{interface_name}.{param_name}")
                            continue

                        # Suggest value for this parameter
                        value = self._suggest_single_param(
                            trial, param_name, param_config['constraints'],
                            category=f"ethtool.{interface_name}"
                        )
                        # Store with interface prefix
                        params[f"{interface_name}_{param_name}"] = value
                        logger.debug(f"Suggested ethtool.{interface_name}.{param_name} = {value}")

            # Non-ethtool categories (sysctl, sysfs, cpufreq)
            else:
                for param_name, param_config in category_settings.items():
                    if 'constraints' not in param_config:
                        logger.warning(f"Missing constraints for {category}.{param_name}")
                        continue

                    constraints = param_config['constraints']

                    # Suggest value for this parameter
                    value = self._suggest_single_param(trial, param_name, constraints, category=category)
                    params[param_name] = value
                    logger.debug(f"Suggested {category}.{param_name} = {value}")

        return params

    def _suggest_single_param(self, trial: optuna.Trial, param_name: str,
                             constraints_list: List[Dict[str, Any]], category: str) -> Any:
        """
        Suggest a value for a single parameter using Optuna.

        v0.3: Supports list of constraints with multiple disjoint ranges or categorical values.

        Args:
            trial: Optuna trial object
            param_name: Parameter name
            constraints_list: List of constraint objects (ranges or categorical)
            category: Parameter category (for error messages)

        Returns:
            Suggested parameter value (int, float, or str)

        Raises:
            ValueError: If constraint structure is invalid
        """
        if not isinstance(constraints_list, list) or len(constraints_list) == 0:
            raise ValueError(f"{category}.{param_name}: constraints must be a non-empty list")

        # Check constraint type from first constraint
        first_constraint = constraints_list[0]

        # Categorical parameter (has 'values')
        if 'values' in first_constraint:
            # For categorical, use first constraint's values
            # (multiple categorical constraints don't make sense)
            values = first_constraint['values']
            return trial.suggest_categorical(param_name, values)

        # Integer/float range (has 'step', 'lower', 'upper')
        elif 'step' in first_constraint and 'lower' in first_constraint and 'upper' in first_constraint:
            # Multiple disjoint ranges - use first range for suggestion
            # Note: Optuna doesn't support disjoint ranges directly, so we use the first range
            # The sharding logic in controller distributes different ranges to different workers
            constraint = first_constraint
            lower = constraint['lower']
            upper = constraint['upper']
            step = constraint['step']

            # Determine if integer or float based on step type
            if isinstance(step, int) and isinstance(lower, int) and isinstance(upper, int):
                return trial.suggest_int(param_name, lower, upper, step=step)
            else:
                return trial.suggest_float(param_name, lower, upper, step=step)

        else:
            raise ValueError(
                f"{category}.{param_name}: constraint must have either 'values' (categorical) "
                f"or 'step/lower/upper' (numeric range)"
            )
    
    def _execute_trial(self, params: Dict[str, Any]) -> Dict[str, float]:
        flow_path = "f/breeder/linux_performance/effectuation_flow"

        # Get targets from configuration
        targets = self.config.get('effectuation', {}).get('targets', [])

        flow_inputs = {
            'config': self.config,
            'targets': targets,
            'params': params
        }

        logger.info(f"Executing effectuation flow for {len(targets)} targets with params: {list(params.keys())}")

        try:
            job_id = wmill.run_flow_async(path=flow_path, args=flow_inputs)
            logger.debug(f"Effectuation flow job ID: {job_id}")

            result = wmill.get_result(job_id)
            logger.info(f"Effectuation flow completed: {result.get('status')}")

            # Extract metrics from reconnaissance step
            metrics = result.get('metrics', {})
            if not metrics:
                logger.error("No metrics returned from effectuation flow")
                return {obj.get('name'): float('inf') for obj in self.config.get('objectives', [])}

            return metrics

        except Exception as e:
            logger.error(f"Effectuation flow failed: {e}", exc_info=True)
            # Return penalty values for failed trials
            return {obj.get('name'): float('inf') for obj in self.config.get('objectives', [])}

    def _check_guardrails(self, metrics: Dict[str, float]) -> tuple[bool, list[str]]:
        """
        Check if guardrails are violated after a trial.

        v0.3: Guardrails are safety limits that trigger rollback if exceeded.
        Unlike objectives (which we optimize), guardrails are binary constraints.

        Args:
            metrics: Dictionary of metric names → values collected from reconnaissance

        Returns:
            tuple: (violated, violations_list)
                - violated: True if any guardrail was exceeded
                - violations_list: List of violation messages
        """
        guardrails = self.config.get('guardrails', [])

        if not guardrails:
            # No guardrails configured
            return False, []

        violations = []

        for guardrail in guardrails:
            name = guardrail.get('name', 'unknown')
            hard_limit = guardrail.get('hard_limit')
            reconnaissance = guardrail.get('reconnaissance', {})

            if not hard_limit:
                logger.warning(f"Guardrail '{name}' missing hard_limit, skipping")
                continue

            # Extract guardrail metric from results
            # Guardrail metrics should be collected by effectuation flow
            # and returned in the metrics dict
            metric_value = metrics.get(name)

            if metric_value is None:
                logger.warning(f"Guardrail '{name}' metric not found in metrics, skipping check")
                continue

            # Determine if guardrail is violated
            # Direction inferred from parameter semantics
            # For common metrics (CPU, errors, latency) = must not exceed
            # Future enhancement: Add explicit direction field to guardrail config
            if isinstance(hard_limit, (int, float)):
                # Assume "must not exceed" for common safety metrics
                if metric_value > hard_limit:
                    violation_msg = f"Guardrail '{name}' violated: {metric_value} > {hard_limit}"
                    violations.append(violation_msg)
                    logger.error(violation_msg)
                else:
                    logger.debug(f"Guardrail '{name}' OK: {metric_value} <= {hard_limit}")
            else:
                logger.warning(f"Guardrail '{name}' has non-numeric hard_limit, skipping")

        return len(violations) > 0, violations

    def _get_rollback_state_key(self) -> str:
        """Get the key for storing rollback state in study user_attrs"""
        return f'rollback_state_target_{self.target_id}'

    def _init_rollback_state(self) -> None:
        """
        Initialize rollback state in Optuna study user_attrs.

        v0.3: Uses YugabyteDB (Optuna storage) for coordination across workers.
        """
        state_key = self._get_rollback_state_key()

        # Check if already initialized
        existing_state = self.study.user_attrs.get(state_key)
        if existing_state:
            logger.debug(f"Rollback state already initialized for target {self.target_id}")
            return

        # Initialize default rollback state
        import json
        initial_state = {
            'state': 'normal',  # normal, needs_rollback, in_progress, completed
            'consecutive_failures': 0,
            'last_successful_params': None,
            'rollback_strategy': self.rollback_config.get('strategy', 'standard'),
            'version': 0  # For optimistic locking
        }

        self.study.set_user_attr(state_key, json.dumps(initial_state))
        logger.info(f"Initialized rollback state for target {self.target_id}: {initial_state}")

    def _get_rollback_state(self) -> Dict[str, Any]:
        """Load rollback state from Optuna study user_attrs"""
        import json

        state_key = self._get_rollback_state_key()
        state_json = self.study.user_attrs.get(state_key)

        if not state_json:
            logger.warning(f"No rollback state found for target {self.target_id}, initializing")
            self._init_rollback_state()
            state_json = self.study.user_attrs.get(state_key)

        return json.loads(state_json)

    def _update_rollback_state(self, new_state: Dict[str, Any]) -> bool:
        """
        Update rollback state in Optuna study with optimistic locking.

        Args:
            new_state: New rollback state to write

        Returns:
            bool: True if update succeeded, False if version conflict
        """
        import json

        state_key = self._get_rollback_state_key()

        # Increment version for optimistic locking
        new_state['version'] = new_state.get('version', 0) + 1

        self.study.set_user_attr(state_key, json.dumps(new_state))
        logger.debug(f"Updated rollback state for target {self.target_id}: version={new_state['version']}, state={new_state['state']}")

        return True

    def _check_needs_rollback(self) -> bool:
        """
        Check if rollback is needed based on consecutive failures threshold.

        Returns:
            bool: True if rollback should be triggered
        """
        if not self.rollback_enabled:
            return False

        rollback_state = self._get_rollback_state()
        consecutive_failures = rollback_state.get('consecutive_failures', 0)

        # Get rollback strategy configuration
        strategy_name = self.rollback_config.get('strategy', 'standard')
        strategies = self.config.get('rollback_strategies', {})
        strategy = strategies.get(strategy_name, {})

        threshold = strategy.get('consecutive_failures', 3)

        if consecutive_failures >= threshold:
            logger.warning(f"Consecutive failures ({consecutive_failures}) >= threshold ({threshold}), rollback needed")
            return True

        return False

    def _execute_rollback(self) -> bool:
        """
        Execute rollback by applying previous parameters to target.

        Returns:
            bool: True if rollback succeeded, False otherwise
        """
        import json

        logger.info(f"Executing rollback for target {self.target_id}")

        rollback_state = self._get_rollback_state()
        strategy_name = self.rollback_config.get('strategy', 'standard')
        strategies = self.config.get('rollback_strategies', {})
        strategy = strategies.get(strategy_name, {})

        # Determine which parameters to restore
        target_state = strategy.get('target_state', 'previous')  # previous, best, baseline

        if target_state == 'previous':
            params_to_restore = rollback_state.get('last_successful_params')
        elif target_state == 'best':
            # Get best trial from study
            if self.study.best_trials[0]:
                params_to_restore = self.study.best_trials[0].params
            else:
                logger.error("Cannot rollback to 'best': no best trial found")
                return False
        elif target_state == 'baseline':
            # Baseline = no tuning applied (use config defaults)
            params_to_restore = {}
        else:
            logger.error(f"Unknown target_state: {target_state}")
            return False

        if params_to_restore is None:
            logger.error(f"No parameters to restore for target_state={target_state}")
            return False

        logger.info(f"Rolling back to {target_state} state with params: {list(params_to_restore.keys())}")

        # Execute rollback by calling effectuation flow
        try:
            flow_path = "f/breeder/linux_performance/effectuation_flow"

            flow_inputs = {
                'config': self.config,
                'targets': [self.target],  # Only this target
                'params': params_to_restore
            }

            logger.info(f"Executing rollback effectuation flow for target {self.target_id}")
            job_id = wmill.run_flow_async(path=flow_path, args=flow_inputs)
            result = wmill.get_result(job_id)

            logger.info(f"Rollback effectuation completed: {result.get('status')}")

            # Update rollback state to completed
            rollback_state['state'] = 'completed'
            rollback_state['consecutive_failures'] = 0  # Reset counter
            self._update_rollback_state(rollback_state)

            # Track successful rollback
            self.metrics.inc_rollback('success')
            self.metrics.push()

            return True

        except Exception as e:
            logger.error(f"Rollback execution failed: {e}", exc_info=True)

            # Track failed rollback
            self.metrics.inc_rollback('failed')
            self.metrics.push()

            # Handle rollback failure based on on_failure policy
            on_failure = strategy.get('on_failure', 'stop')

            if on_failure == 'stop':
                logger.error("Rollback failed with on_failure=stop, halting optimization")
                rollback_state['state'] = 'failed'
                self._update_rollback_state(rollback_state)
                raise  # Re-raise to stop worker

            elif on_failure == 'continue':
                logger.warning("Rollback failed with on_failure=continue, continuing optimization")
                rollback_state['state'] = 'failed'
                self._update_rollback_state(rollback_state)
                return False

            elif on_failure == 'skip_target':
                logger.error("Rollback failed with on_failure=skip_target, marking target unhealthy")
                rollback_state['state'] = 'skip_target'
                self._update_rollback_state(rollback_state)
                return False

            return False

    def _handle_guardrail_violation(self, params: Dict[str, Any]) -> None:
        """
        Handle guardrail violation by tracking consecutive failures.

        Args:
            params: Parameters that caused the violation
        """
        if not self.rollback_enabled:
            return

        rollback_state = self._get_rollback_state()

        # Increment consecutive failures counter
        rollback_state['consecutive_failures'] = rollback_state.get('consecutive_failures', 0) + 1

        consecutive_failures = rollback_state['consecutive_failures']
        logger.warning(f"Guardrail violation detected. Consecutive failures: {consecutive_failures}")

        # Update state
        if self._check_needs_rollback():
            rollback_state['state'] = 'needs_rollback'
        else:
            rollback_state['state'] = 'normal'

        self._update_rollback_state(rollback_state)

    def _handle_successful_trial(self, params: Dict[str, Any]) -> None:
        """
        Handle successful trial by resetting failure counter and updating last successful params.

        Args:
            params: Parameters from successful trial
        """
        if not self.rollback_enabled:
            return

        rollback_state = self._get_rollback_state()

        # Reset consecutive failures on success
        rollback_state['consecutive_failures'] = 0
        rollback_state['state'] = 'normal'
        rollback_state['last_successful_params'] = params

        self._update_rollback_state(rollback_state)
        logger.debug(f"Reset consecutive failures after successful trial")
    
    def _should_continue(self) -> bool:
        completion_criteria = self.config.get('run', {}).get('completion_criteria', {})
        
        min_iterations = completion_criteria.get('iterations', {}).get('min', 10)
        max_iterations = completion_criteria.get('iterations', {}).get('max', 1000)
        
        n_trials = len(self.study.trials)
        
        if n_trials < min_iterations:
            logger.debug(f"Continuing: {n_trials} < {min_iterations} min iterations")
            return True
        if n_trials >= max_iterations:
            logger.info(f"Stopping: {n_trials} >= {max_iterations} max iterations")
            return False
        
        # Check time budget
        if self._check_time_budget(completion_criteria):
            logger.info("Stopping: Time budget exceeded")
            return False
        
        # Check quality thresholds
        if completion_criteria.get('quality_achieved', False):
            if self._check_quality_thresholds():
                logger.info("Stopping: All quality thresholds achieved")
                return False
            
        return True
    
    def _check_time_budget(self, completion_criteria: dict) -> bool:
        """Check if time budget has been exceeded"""
        timing_config = completion_criteria.get('timing', {})
        end_time_str = timing_config.get('end')
        
        if not end_time_str:
            return False
        
        if not hasattr(self, 'start_time'):
            return False
        
        # Parse time string (e.g., "7d", "24h", "60m")
        import re
        match = re.match(r'(\d+)([dhm])', end_time_str)
        if not match:
            logger.warning(f"Invalid time format: {end_time_str}")
            return False
        
        value, unit = match.groups()
        value = int(value)
        
        # Convert to seconds
        unit_seconds = {'d': 86400, 'h': 3600, 'm': 60}
        budget_seconds = value * unit_seconds[unit]
        
        elapsed_seconds = (datetime.datetime.now() - self.start_time).total_seconds()
        
        return elapsed_seconds >= budget_seconds
    
    def _check_quality_thresholds(self) -> bool:
        """Check if all objectives have reached their quality thresholds"""
        if not self.study.best_trials[0]:
            return False
        
        objectives = self.config.get('objectives', [])
        if not objectives:
            return False
        
        # Check if all objectives have quality_threshold defined
        for objective in objectives:
            if 'quality_threshold' not in objective:
                return False
        
        # Check if all thresholds achieved
        for obj_value, objective in zip(self.study.best_trials[0].values, objectives):
            threshold = objective.get('quality_threshold')
            direction = objective.get('direction', 'minimize')
            
            if direction == 'minimize':
                if obj_value > threshold:
                    return False
            elif direction == 'maximize':
                if obj_value < threshold:
                    return False
        
        return True
    
    def _update_state(self):
        state = {
            'breeder_id': self.breeder_id,
            'total_trials': len(self.study.trials),
            'study_name': self.study.study_name,
            'status': 'running'
        }
        
        if self.study.best_trials[0]:
            state['best_trial_number'] = self.study.best_trials[0].number
            state['best_params'] = self.study.best_trials[0].params
            state['best_values'] = self.study.best_trials[0].values
        
        wmill.set_state(state)
        logger.debug(f"Updated Windmill state: {state}")
    
    def run(self):
        logger.info(f"Starting BreederWorker: {self.worker_id}")
        logger.info(f"Breeder type: {self.breeder_type}, UUID: {self.breeder_uuid}")

        # Mark worker as running
        self.metrics.mark_running()
        self.metrics.push()

        trial_count = 0

        try:
            while self._should_continue():
                # Check if rollback is needed before starting new trial
                if self.rollback_enabled and self._check_needs_rollback():
                    logger.warning("Rollback needed, executing rollback before next trial")
                    rollback_success = self._execute_rollback()

                    if not rollback_success:
                        # Rollback failed, handle based on on_failure policy
                        # _execute_rollback already raised exception if on_failure=stop
                        logger.warning("Rollback failed, continuing with trials")

                    # Apply after-action policy (pause/continue/stop)
                    rollback_state = self._get_rollback_state()
                    strategy_name = self.rollback_config.get('strategy', 'standard')
                    strategies = self.config.get('rollback_strategies', {})
                    strategy = strategies.get(strategy_name, {})
                    after_policy = strategy.get('after', {})
                    after_action = after_policy.get('action', 'continue')

                    if after_action == 'pause':
                        pause_duration = after_policy.get('duration', 300)
                        logger.info(f"Pausing for {pause_duration} seconds after rollback")
                        import time
                        time.sleep(pause_duration)
                    elif after_action == 'stop':
                        logger.info("Rollback completed with after.action=stop, halting optimization")
                        break
                    # continue = just continue to next trial

                trial = self.study.ask()
                logger.info(f"Trial {trial.number} started")

                trial_start_time = time.time()

                try:
                    params = self._suggest_params(trial)
                    metrics = self._execute_trial(params)

                    # Check guardrails before accepting trial results
                    guardrails_violated, violations = self._check_guardrails(metrics)

                    if guardrails_violated:
                        # Guardrail violation - mark trial as failed
                        logger.error(f"Trial {trial.number} failed guardrails: {violations}")
                        self.study.tell(trial, state=TrialState.FAIL)
                        logger.info(f"Trial {trial.number} marked as FAILED (guardrail violation)")

                        # Track guardrail violations
                        for violation_msg in violations:
                            guardrail_name = violation_msg.split(':')[0] if ':' in violation_msg else 'unknown'
                            self.metrics.inc_guardrail_violation(guardrail_name)

                        # Track failures for rollback logic
                        self._handle_guardrail_violation(params)

                        # Track failed trial
                        self.metrics.inc_trial('failed')
                        self.metrics.inc_effectuation('failure')
                    else:
                        # No guardrail violations - accept trial results
                        values = [metrics.get(obj.get('name')) for obj in self.config.get('objectives', [])]
                        self.study.tell(trial, values)

                        trial_duration = time.time() - trial_start_time

                        logger.info(f"Trial {trial.number} completed with values: {values}")

                        # Update metrics
                        self.metrics.inc_trial('complete', value=values[0] if values else None)
                        self.metrics.observe_trial_duration(trial_duration)
                        self.metrics.inc_effectuation('success')

                        # Update best value if this is the new best
                        if self.study.best_trials[0] and self.study.best_trials[0].number == trial.number:
                            self.metrics.set_best_value(values[0] if values else 0)

                        # Reset failure counter on successful trial
                        self._handle_successful_trial(params)

                        if self.communication_callback:
                            frozen_trial = self.study.trials[-1]
                            self.communication_callback(self.study, frozen_trial)

                            # Track shared trials
                            coop_config = self.config.get('cooperation', {})
                            share_strategy = coop_config.get('share_strategy', 'unknown')
                            self.metrics.inc_trial_shared(share_strategy)

                    trial_count += 1
                    if trial_count % 5 == 0:
                        self._update_state()
                        self.metrics.set_total_trials(len(self.study.trials))
                        self.metrics.push()  # Push metrics every 5 trials

                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
                    self.study.tell(trial, state=TrialState.FAIL)
                    logger.info(f"Trial {trial.number} marked as FAILED")

                    # Track failed trial
                    self.metrics.inc_trial('failed')
                    self.metrics.inc_effectuation('failure')

                    # Track failures for rollback logic
                    # (even non-guardrail failures count toward consecutive failures)
                    self._handle_guardrail_violation(params)

        except Exception as e:
            logger.error(f"Breeder {self.breeder_id} failed: {e}", exc_info=True)
            self._update_state()
            raise
        finally:
            # Always mark worker as stopped
            self.metrics.mark_stopped()
            self.metrics.push()

        self._update_state()
        logger.info(f"BreederWorker {self.worker_id} completed {len(self.study.trials)} trials")

        if self.study.best_trials[0]:
            logger.info(f"Best trial: {self.study.best_trials[0].number}")
            logger.info(f"Best params: {self.study.best_trials[0].params}")
            logger.info(f"Best values: {self.study.best_trials[0].values}")


def main(config: Dict[str, Any], breeder_id: str = None, run_id: int = None, target_id: int = None) -> Dict[str, Any]:
    """
    Main entry point for breeder worker.
    
    Args:
        config: Breeder configuration (full or sharded, depending on mode)
        breeder_id: UUID of the breeder (for state updates)
        run_id: Parallel run identifier (for logging)
        target_id: Target identifier (for logging)
    
    Returns:
        Worker execution results
    """
    if breeder_id:
        logger.info(f"Starting worker for breeder: {breeder_id}, run: {run_id}, target: {target_id}")
    
    worker = BreederWorker(config)
    worker.run()
    
    return {
        'worker_id': worker.worker_id,
        'breeder_type': worker.breeder_type,
        'breeder_id': worker.breeder_id,
        'run_id': run_id,
        'target_id': target_id,
        'total_trials': len(worker.study.trials),
        'best_params': worker.study.best_trials[0].params if worker.study.best_trials else None,
        'best_values': worker.study.best_trials[0].values if worker.study.best_trials else None,
        'status': 'completed'
    }