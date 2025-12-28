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
from typing import Dict, Any, Optional, List
from optuna.trial import TrialState
from optuna.samplers import TPESampler, NSGAIISampler, NSGAIIISampler, RandomSampler, QMCSampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CommunicationCallback:
    """Shares successful trials with cooperating breeders for metaheuristic learning"""
    
    def __init__(self, storage: str, probability: float = 0.8, share_within_breeder: bool = True):
        self.storage = storage
        self.com_probability = probability
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
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Probabilistic trial sharing based on consolidation probability"""
        if random.random() < self.com_probability:
            self.logger.debug(f"Sharing trial {trial.number} (probability: {self.com_probability})")
            self._share_trial(study, trial)
        else:
            self.logger.debug(f"Skipping trial sharing for {trial.number}")


class BreederWorker:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        breeder_config = config.get('breeder', {})
        
        self.breeder_type = breeder_config.get('name', 'unknown_breeder')
        self.breeder_uuid = breeder_config.get('uuid', breeder_config.get('name', 'unknown'))
        self.breeder_id = self.breeder_uuid  # For database/communication
        self.worker_id = f"{self.breeder_type}_worker_{self.breeder_uuid}"
        
        # Assign sampler to this worker for algorithm diversity
        self.sampler_type = self._assign_sampler()
        
        self.study = self._load_or_create_study()
        self.communication_callback = self._setup_communication()
        self._update_state()
        
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
                'multivariate': [True, False],  # HIGH - Well understood parameter
                'group': [True, False],  # HIGH - Documented behavior
                'constant_liar': [True, False],  # HIGH - Proven for parallel optimization
                'n_startup_trials': [5, 10, 20]  # MEDIUM - Educated guess, needs validation
            },
            'nsga2': {
                'population_size': [20, 50, 100],  # HIGH - Default 50, range well-established
                'mutation_prob': [0.05, 0.1, 0.15],  # LOW - Educated guess from GA literature
                'crossover_prob': [0.8, 0.9, 0.95],  # MEDIUM - Common GA values, less certain for Optuna
                'crossover': ['uniform', 'UNDX', 'SPX']  # LOW - UNDX/SPX need validation
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
            config = {
                'multivariate': random.choice(profile['multivariate']),
                'group': random.choice(profile['group']),
                'constant_liar': random.choice(profile['constant_liar']),
                'n_startup_trials': random.choice(profile['n_startup_trials'])
            }
            logger.info(f"Created TPE sampler with config: {config}")
            return TPESampler(**config)
        
        elif sampler_type == 'nsga2':
            profile = sampler_profiles['nsga2']
            population_size = random.choice(profile['population_size'])
            crossover = random.choice(profile['crossover'])
            
            # Skip crossover-specific parameters if not using that crossover
            crossover_params = {}
            if crossover == 'uniform':
                crossover_params['crossover'] = 'uniform'
            elif crossover in ['UNDX', 'SPX']:
                crossover_params['crossover'] = crossover
                crossover_params['population_size'] = max(population_size, 3)  # UNDX/SPX need >=3 parents
            
            config = {
                'population_size': population_size,
                'mutation_prob': random.choice(profile['mutation_prob']),
                'crossover_prob': random.choice(profile['crossover_prob']),
                **crossover_params
            }
            logger.info(f"Created NSGAII sampler with config: {config}")
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
            'database': self.config.get('breeder_database')
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
            probability = cooperation_config.get('consolidation', {}).get('probability', 0.8)
            storage = self._get_db_url()
            
            # If algorithm diversity is enabled (parallel > 1), share within breeder (across sampler-specific studies)
            share_within_breeder = parallel_workers > 1
            
            logger.info(f"Communication enabled with probability: {probability}, share_within_breeder: {share_within_breeder}")
            return CommunicationCallback(storage=storage, probability=probability, share_within_breeder=share_within_breeder)
        else:
            logger.info("Communication disabled")
            return None
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        
        settings_config = self.config.get('settings', {}).get('sysctl', {})
        for setting_name, setting_config in settings_config.items():
            constraints = setting_config.get('constraints', {})
            step = setting_config.get('step', 1)
            
            lower = constraints.get('lower', 0)
            upper = constraints.get('upper', 1000000)
            
            value = trial.suggest_int(setting_name, lower, upper, step=step)
            params[setting_name] = value
            logger.debug(f"Suggested {setting_name} = {value}")
        
        return params
    
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
    
    def _should_continue(self) -> bool:
        min_iterations = self.config.get('run', {}).get('iterations', {}).get('min', 10)
        max_iterations = self.config.get('run', {}).get('iterations', {}).get('max', 1000)
        
        n_trials = len(self.study.trials)
        
        if n_trials < min_iterations:
            logger.debug(f"Continuing: {n_trials} < {min_iterations} min iterations")
            return True
        if n_trials >= max_iterations:
            logger.info(f"Stopping: {n_trials} >= {max_iterations} max iterations")
            return False
            
        return True
    
    def _update_state(self):
        state = {
            'breeder_id': self.breeder_id,
            'total_trials': len(self.study.trials),
            'study_name': self.study.study_name,
            'status': 'running'
        }
        
        if self.study.best_trial:
            state['best_trial_number'] = self.study.best_trial.number
            state['best_params'] = self.study.best_trial.params
            state['best_values'] = self.study.best_trial.values
        
        wmill.set_state(state)
        logger.debug(f"Updated Windmill state: {state}")
    
    def run(self):
        logger.info(f"Starting BreederWorker: {self.worker_id}")
        logger.info(f"Breeder type: {self.breeder_type}, UUID: {self.breeder_uuid}")
        
        trial_count = 0
        
        try:
            while self._should_continue():
                trial = self.study.ask()
                logger.info(f"Trial {trial.number} started")
                
                try:
                    params = self._suggest_params(trial)
                    metrics = self._execute_trial(params)
                    
                    values = [metrics.get(obj.get('name')) for obj in self.config.get('objectives', [])]
                    self.study.tell(trial, values)
                    
                    logger.info(f"Trial {trial.number} completed with values: {values}")
                    
                    if self.communication_callback:
                        frozen_trial = self.study.trials[-1]
                        self.communication_callback(self.study, frozen_trial)
                    
                    trial_count += 1
                    if trial_count % 5 == 0:
                        self._update_state()
                    
                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
                    self.study.tell(trial, state=TrialState.FAIL)
                    logger.info(f"Trial {trial.number} marked as FAILED")
                    
        except Exception as e:
            logger.error(f"Breeder {self.breeder_id} failed: {e}", exc_info=True)
            self._update_state()
            raise
        
        self._update_state()
        logger.info(f"BreederWorker {self.worker_id} completed {len(self.study.trials)} trials")
        
        if self.study.best_trial:
            logger.info(f"Best trial: {self.study.best_trial.number}")
            logger.info(f"Best params: {self.study.best_trial.params}")
            logger.info(f"Best values: {self.study.best_trial.values}")


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
        'best_params': worker.study.best_trial.params if worker.study.best_trial else None,
        'best_values': worker.study.best_trial.values if worker.study.best_trial else None,
        'status': 'completed'
    }