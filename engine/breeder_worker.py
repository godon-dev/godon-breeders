#extra_requirements:
#opentelemetry-api
#opentelemetry-sdk
#opentelemetry-exporter-otlp

import optuna
import random
import hashlib
import datetime
import dateutil.parser
import time
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
from f.breeder.engine.breeder_metrics_client import BreederMetricsClient
from f.breeder.engine.communication import CommunicationCallback
from f.breeder.engine.strain_loader import load_strain
from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


class BreederWorker:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        breeder_config = config.get('breeder', {})

        self.breeder_type = breeder_config.get('name', 'unknown_breeder')
        self.breeder_uuid = breeder_config.get('uuid', breeder_config.get('name', 'unknown'))
        self.breeder_id = self.breeder_uuid
        self.breeder_db_name = f"breeder_{self.breeder_uuid.replace('-', '_')}"
        self.worker_id = f"{self.breeder_type}_worker_{self.breeder_uuid}"

        strain_type = breeder_config.get('type', 'linux_performance')
        self.strain = load_strain(strain_type)

        creation_ts_str = config.get('creation_ts')
        if not creation_ts_str:
            raise ValueError("Required field 'creation_ts' missing from config")
        self.start_time = dateutil.parser.parse(creation_ts_str)

        self.sampler_type = self._assign_sampler()

        self.study = self._load_or_create_study()
        self.communication_callback = self._setup_communication()

        self.run_id = config.get('run_id', 0)
        self.target_id = config.get('target_id', 0)

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
            self._init_rollback_state()

        self._update_state()

        self.metrics = BreederMetricsClient(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type
        )
        
    def _assign_sampler(self) -> str:
        parallel_workers = self.config.get('run', {}).get('parallel', 1)
        
        if parallel_workers <= 1:
            logger.info("Single worker mode, using default TPE sampler")
            return 'tpe'
        
        all_samplers = ['tpe', 'nsga2', 'random', 'nsga3', 'qmc']
        num_samplers = min(parallel_workers, len(all_samplers))
        available_samplers = all_samplers[:num_samplers]
        
        worker_hash = int(hashlib.md5(self.worker_id.encode()).hexdigest(), 16)
        sampler_index = worker_hash % len(available_samplers)
        assigned_sampler = available_samplers[sampler_index]
        
        logger.info(f"Algorithm diversity auto-enabled ({len(available_samplers)} samplers for {parallel_workers} workers): Worker {self.worker_id} assigned '{assigned_sampler}' sampler")
        return assigned_sampler
    
    def _create_sampler(self, sampler_type: str) -> optuna.samplers.BaseSampler:
        
        sampler_profiles = {
            'tpe': {
                'multivariate_group': [(True, True), (True, False), (False, False)],
                'constant_liar': [True, False],
                'n_startup_trials': [5, 10, 20]
            },
            'nsga2': {
                'population_size': [30, 50, 75, 100, 125, 150],
                'mutation_prob': [0.05, 0.1, 0.15],
                'crossover_prob': [0.8, 0.9, 0.95],
                'crossover': ['uniform', 'UNDX', 'SPX', 'BLXAlpha', 'SBX', 'VSBX']
            },
            'nsga3': {
                'population_size': [50, 100]
            },
            'random': {
                'seed': [None]
            },
            'qmc': {
                'seed': [None]
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

            if crossover_name == 'uniform':
                crossover_obj = UniformCrossover()
            elif crossover_name == 'UNDX':
                population_size = max(population_size, 3)
                crossover_obj = UNDXCrossover()
            elif crossover_name == 'SPX':
                population_size = max(population_size, 3)
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
        import os
        db_config = {
            'user': os.environ.get("GODON_ARCHIVE_DB_USER", "postgres"),
            'password': os.environ.get("GODON_ARCHIVE_DB_PASSWORD", "postgres"),
            'host': os.environ.get("GODON_ARCHIVE_DB_SERVICE_HOST", "localhost"),
            'port': os.environ.get("GODON_ARCHIVE_DB_SERVICE_PORT", "5432"),
            'database': self.breeder_db_name
        }
        return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    def _load_or_create_study(self) -> optuna.Study:
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
        cooperation_config = self.config.get('cooperation', {})
        parallel_workers = self.config.get('run', {}).get('parallel', 1)
        
        if cooperation_config.get('active', False):
            share_strategy = cooperation_config.get('share_strategy', 'probabilistic')
            probability = cooperation_config.get('probability', 0.8)
            top_percentile = cooperation_config.get('top_percentile', 0.2)
            bottom_percentile = cooperation_config.get('bottom_percentile', 0.2)
            min_trials_for_filtering = cooperation_config.get('min_trials_for_filtering', 10)
            storage = self._get_db_url()
            
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
    
    def _execute_trial(self, params: Dict[str, Any]) -> Dict[str, float]:
        flow_path = self.strain.EFFECTUATION_FLOW

        targets = self.config.get('effectuation', {}).get('targets', [])

        flow_inputs = {
            'config': self.config,
            'targets': targets,
            'params': params
        }

        logger.info(f"Executing effectuation flow for {len(targets)} targets with params: {list(params.keys())}")

        try:
            import wmill
            job_id = wmill.run_flow_async(path=flow_path, args=flow_inputs)
            logger.debug(f"Effectuation flow job ID: {job_id}")

            result = wmill.get_result(job_id)
            logger.info(f"Effectuation flow completed: {result.get('status')}")

            metrics = result.get('metrics', {})
            if not metrics:
                logger.error("No metrics returned from effectuation flow")
                return {obj.get('name'): float('inf') for obj in self.config.get('objectives', [])}

            return metrics

        except Exception as e:
            logger.error(f"Effectuation flow failed: {e}", exc_info=True)
            return {obj.get('name'): float('inf') for obj in self.config.get('objectives', [])}

    def _check_guardrails(self, metrics: Dict[str, float]) -> tuple[bool, list[str]]:
        guardrails = self.config.get('guardrails', [])

        if not guardrails:
            return False, []

        violations = []

        for guardrail in guardrails:
            name = guardrail.get('name', 'unknown')
            hard_limit = guardrail.get('hard_limit')

            if not hard_limit:
                logger.warning(f"Guardrail '{name}' missing hard_limit, skipping")
                continue

            metric_value = metrics.get(name)

            if metric_value is None:
                logger.warning(f"Guardrail '{name}' metric not found in metrics, skipping check")
                continue

            if isinstance(hard_limit, (int, float)):
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
        return f'rollback_state_target_{self.target_id}'

    def _init_rollback_state(self) -> None:
        state_key = self._get_rollback_state_key()

        existing_state = self.study.user_attrs.get(state_key)
        if existing_state:
            logger.debug(f"Rollback state already initialized for target {self.target_id}")
            return

        import json
        initial_state = {
            'state': 'normal',
            'consecutive_failures': 0,
            'last_successful_params': None,
            'rollback_strategy': self.rollback_config.get('strategy', 'standard'),
            'version': 0
        }

        self.study.set_user_attr(state_key, json.dumps(initial_state))
        logger.info(f"Initialized rollback state for target {self.target_id}: {initial_state}")

    def _get_rollback_state(self) -> Dict[str, Any]:
        import json

        state_key = self._get_rollback_state_key()
        state_json = self.study.user_attrs.get(state_key)

        if not state_json:
            logger.warning(f"No rollback state found for target {self.target_id}, initializing")
            self._init_rollback_state()
            state_json = self.study.user_attrs.get(state_key)

        return json.loads(state_json)

    def _update_rollback_state(self, new_state: Dict[str, Any]) -> bool:
        import json

        state_key = self._get_rollback_state_key()

        new_state['version'] = new_state.get('version', 0) + 1

        self.study.set_user_attr(state_key, json.dumps(new_state))
        logger.debug(f"Updated rollback state for target {self.target_id}: version={new_state['version']}, state={new_state['state']}")

        return True

    def _check_needs_rollback(self) -> bool:
        if not self.rollback_enabled:
            return False

        rollback_state = self._get_rollback_state()
        consecutive_failures = rollback_state.get('consecutive_failures', 0)

        strategy_name = self.rollback_config.get('strategy', 'standard')
        strategies = self.config.get('rollback_strategies', {})
        strategy = strategies.get(strategy_name, {})

        threshold = strategy.get('consecutive_failures', 3)

        if consecutive_failures >= threshold:
            logger.warning(f"Consecutive failures ({consecutive_failures}) >= threshold ({threshold}), rollback needed")
            return True

        return False

    def _execute_rollback(self) -> bool:
        import json

        logger.info(f"Executing rollback for target {self.target_id}")

        rollback_state = self._get_rollback_state()
        strategy_name = self.rollback_config.get('strategy', 'standard')
        strategies = self.config.get('rollback_strategies', {})
        strategy = strategies.get(strategy_name, {})

        target_state = strategy.get('target_state', 'previous')

        if target_state == 'previous':
            params_to_restore = rollback_state.get('last_successful_params')
        elif target_state == 'best':
            if self.study.best_trials:
                params_to_restore = self.study.best_trials[0].params
            else:
                logger.error("Cannot rollback to 'best': no best trial found")
                return False
        elif target_state == 'baseline':
            params_to_restore = {}
        else:
            logger.error(f"Unknown target_state: {target_state}")
            return False

        if params_to_restore is None:
            logger.error(f"No parameters to restore for target_state={target_state}")
            return False

        logger.info(f"Rolling back to {target_state} state with params: {list(params_to_restore.keys())}")

        try:
            import wmill
            flow_path = self.strain.EFFECTUATION_FLOW

            flow_inputs = {
                'config': self.config,
                'targets': [self.target],
                'params': params_to_restore
            }

            logger.info(f"Executing rollback effectuation flow for target {self.target_id}")
            job_id = wmill.run_flow_async(path=flow_path, args=flow_inputs)
            result = wmill.get_result(job_id)

            logger.info(f"Rollback effectuation completed: {result.get('status')}")

            rollback_state['state'] = 'completed'
            rollback_state['consecutive_failures'] = 0
            self._update_rollback_state(rollback_state)

            self.metrics.inc_rollback('success')
            self.metrics.push()

            return True

        except Exception as e:
            logger.error(f"Rollback execution failed: {e}", exc_info=True)

            self.metrics.inc_rollback('failed')
            self.metrics.push()

            on_failure = strategy.get('on_failure', 'stop')

            if on_failure == 'stop':
                logger.error("Rollback failed with on_failure=stop, halting optimization")
                rollback_state['state'] = 'failed'
                self._update_rollback_state(rollback_state)
                raise

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
        if not self.rollback_enabled:
            return

        rollback_state = self._get_rollback_state()

        rollback_state['consecutive_failures'] = rollback_state.get('consecutive_failures', 0) + 1

        consecutive_failures = rollback_state['consecutive_failures']
        logger.warning(f"Guardrail violation detected. Consecutive failures: {consecutive_failures}")

        if self._check_needs_rollback():
            rollback_state['state'] = 'needs_rollback'
        else:
            rollback_state['state'] = 'normal'

        self._update_rollback_state(rollback_state)

    def _handle_successful_trial(self, params: Dict[str, Any]) -> None:
        if not self.rollback_enabled:
            return

        rollback_state = self._get_rollback_state()

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

        if self._check_time_budget(completion_criteria):
            logger.info("Stopping: Time budget exceeded")
            return False

        if completion_criteria.get('quality_achieved', False):
            if self._check_quality_thresholds():
                logger.info("Stopping: All quality thresholds achieved")
                return False

        if self._check_shutdown_requested():
            logger.info("Stopping: Shutdown requested by controller")
            return False

        return True

    def _check_shutdown_requested(self) -> bool:
        try:
            query = "SELECT shutdown_requested FROM breeder_state LIMIT 1;"
            result = self.study.storage._engine.execute(query)

            if result and result.rowcount > 0:
                row = result.fetchone()
                shutdown_requested = row[0] if row else False
                if shutdown_requested:
                    logger.info(f"Shutdown flag is set for breeder {self.breeder_uuid}")
                    return True

            return False

        except Exception as e:
            logger.warning(f"Failed to check shutdown flag: {e}")
            return False
    
    def _check_time_budget(self, completion_criteria: dict) -> bool:
        import re
        timing_config = completion_criteria.get('timing', {})
        end_time_str = timing_config.get('end')
        
        if not end_time_str:
            return False
        
        if not hasattr(self, 'start_time'):
            return False
        
        match = re.match(r'(\d+)([dhm])', end_time_str)
        if not match:
            logger.warning(f"Invalid time format: {end_time_str}")
            return False
        
        value, unit = match.groups()
        value = int(value)
        
        unit_seconds = {'d': 86400, 'h': 3600, 'm': 60}
        budget_seconds = value * unit_seconds[unit]
        
        elapsed_seconds = (datetime.datetime.now() - self.start_time).total_seconds()
        
        return elapsed_seconds >= budget_seconds
    
    def _check_quality_thresholds(self) -> bool:
        if not self.study.best_trials:
            return False

        objectives = self.config.get('objectives', [])
        if not objectives:
            return False

        for objective in objectives:
            if 'quality_threshold' not in objective:
                return False

        for trial in self.study.best_trials:
            all_thresholds_met = True
            for obj_value, objective in zip(trial.values, objectives):
                threshold = objective.get('quality_threshold')
                direction = objective.get('direction', 'minimize')

                if direction == 'minimize':
                    if obj_value > threshold:
                        all_thresholds_met = False
                        break
                elif direction == 'maximize':
                    if obj_value < threshold:
                        all_thresholds_met = False
                        break

            if all_thresholds_met:
                return True

        return False
    
    def _update_state(self):
        import wmill
        state = {
            'breeder_id': self.breeder_id,
            'total_trials': len(self.study.trials),
            'study_name': self.study.study_name,
            'status': 'running'
        }

        wmill.set_state(state)
        logger.debug(f"Updated Windmill state: {state}")
    
    def run(self):
        logger.info(f"Starting BreederWorker: {self.worker_id}")
        logger.info(f"Breeder type: {self.breeder_type}, UUID: {self.breeder_uuid}")

        self.metrics.mark_running()
        self.metrics.push()

        trial_count = 0

        try:
            while self._should_continue():
                if self.rollback_enabled and self._check_needs_rollback():
                    logger.warning("Rollback needed, executing rollback before next trial")
                    rollback_success = self._execute_rollback()

                    if not rollback_success:
                        logger.warning("Rollback failed, continuing with trials")

                    rollback_state = self._get_rollback_state()
                    strategy_name = self.rollback_config.get('strategy', 'standard')
                    strategies = self.config.get('rollback_strategies', {})
                    strategy = strategies.get(strategy_name, {})
                    after_policy = strategy.get('after', {})
                    after_action = after_policy.get('action', 'continue')

                    if after_action == 'pause':
                        pause_duration = after_policy.get('duration', 300)
                        logger.info(f"Pausing for {pause_duration} seconds after rollback")
                        time.sleep(pause_duration)
                    elif after_action == 'stop':
                        logger.info("Rollback completed with after.action=stop, halting optimization")
                        break

                trial = self.study.ask()
                logger.info(f"Trial {trial.number} started")

                trial_start_time = time.time()

                params = None

                try:
                    params = self.strain.suggest_params(trial, self.config.get('settings', {}))
                    metrics = self._execute_trial(params)

                    guardrails_violated, violations = self._check_guardrails(metrics)

                    if guardrails_violated:
                        logger.error(f"Trial {trial.number} failed guardrails: {violations}")
                        self.study.tell(trial, state=TrialState.FAIL)
                        logger.info(f"Trial {trial.number} marked as FAILED (guardrail violation)")

                        for violation_msg in violations:
                            guardrail_name = violation_msg.split(':')[0] if ':' in violation_msg else 'unknown'
                            self.metrics.inc_guardrail_violation(guardrail_name)

                        self._handle_guardrail_violation(params)

                        self.metrics.inc_trial('failed')
                        self.metrics.inc_effectuation('failure')
                    else:
                        values = [metrics.get(obj.get('name')) for obj in self.config.get('objectives', [])]
                        self.study.tell(trial, values)

                        trial_duration = time.time() - trial_start_time

                        logger.info(f"Trial {trial.number} completed with values: {values}")

                        self.metrics.inc_trial('complete', value=values[0] if values else None)
                        self.metrics.observe_trial_duration(trial_duration)
                        self.metrics.inc_effectuation('success')

                        if self.study.best_trials[0] and self.study.best_trials[0].number == trial.number:
                            self.metrics.set_best_value(values[0] if values else 0)

                        self._handle_successful_trial(params)

                        if self.communication_callback:
                            frozen_trial = self.study.trials[-1]
                            self.communication_callback(self.study, frozen_trial)

                            coop_config = self.config.get('cooperation', {})
                            share_strategy = coop_config.get('share_strategy', 'unknown')
                            self.metrics.inc_trial_shared(share_strategy)

                    trial_count += 1
                    if trial_count % 5 == 0:
                        self._update_state()
                        self.metrics.set_total_trials(len(self.study.trials))
                        self.metrics.push()

                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
                    self.study.tell(trial, state=TrialState.FAIL)
                    logger.info(f"Trial {trial.number} marked as FAILED")

                    self.metrics.inc_trial('failed')
                    self.metrics.inc_effectuation('failure')

                    self._handle_guardrail_violation(params)

        except Exception as e:
            logger.error(f"Breeder {self.breeder_id} failed: {e}", exc_info=True)
            self._update_state()
            raise
        finally:
            self.metrics.mark_stopped()
            self.metrics.push()

        self._update_state()
        logger.info(f"BreederWorker {self.worker_id} completed {len(self.study.trials)} trials")

        if self.study.best_trials:
            logger.info(f"Found {len(self.study.best_trials)} Pareto-optimal trials")
            for i, trial in enumerate(self.study.best_trials[:3]):
                logger.info(f"  Trial {i+1}: #{trial.number}, values={trial.values}, params={trial.params}")
            if len(self.study.best_trials) > 3:
                logger.info(f"  ... and {len(self.study.best_trials) - 3} more")


def main(config: Dict[str, Any], breeder_id: str = None, run_id: int = None, target_id: int = None) -> Dict[str, Any]:
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
        'pareto_optimal_trials': len(worker.study.best_trials),
        'status': 'completed'
    }
