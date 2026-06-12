#extra_requirements:
#opentelemetry-api
#opentelemetry-sdk
#opentelemetry-exporter-otlp
#psycopg2-binary
#wmill

import optuna
import json
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
from f.breeder.engine.watermark import create_watermark, Watermark
from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_DB_ERROR_PATTERNS = (
    'SerializationFailure',
    '40001',
    'Transaction aborted',
    'Timed out waiting',
    'InternalError_',
    'Transaction metadata missing',
    'Heartbeat',
    'conflict',
)


class BreederWorker:

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        exc_str = str(exc)
        for pattern in _RETRYABLE_DB_ERROR_PATTERNS:
            if pattern in exc_str:
                return True
        try:
            if isinstance(exc, optuna.exceptions.StorageInternalError):
                return True
        except Exception:
            pass
        return False

    def _retry_op(self, fn, description: str, max_retries: int = 4):
        last_error = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1 and self._is_retryable_error(e):
                    wait_time = 2 ** attempt
                    logger.warning(f"{description} attempt {attempt + 1}/{max_retries} failed: {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise
        raise last_error

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        breeder_config = config.get('breeder', {})

        self.breeder_type = breeder_config.get('type', 'unknown_breeder')
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

        self._trial_durations = []

        self._last_heartbeat_ts = 0
        self._heartbeat_interval = 120
        self._last_metric_noise = {}

        self._register_interference_breeder()

        settings = config.get('settings', {})
        self.watermark = None
        if self._has_active_neighbors():
            self.watermark = create_watermark(config, settings, breeder_uuid=self.breeder_uuid)
            if self.watermark:
                logger.info(f"Watermarking enabled: {self.watermark.metadata()}")
        self._watermark_trial_idx = 0
        self._wm_corrected_params = None
        self._watermark_baseline = self._compute_baseline_params(settings)

        # Calibrated detection params — populated via AIMD on first use
        self._calibrated_impulse_params = None
        self._calibrated_hold_params = None
        self._impulse_scale = 1.0
        self._impulse_base_params = None  # AIMD: multiplicative decrease on FAIL

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

    def _get_shared_db_url(self) -> str:
        import os
        pw = os.environ.get('GODON_ARCHIVE_DB_PASSWORD', 'postgres')
        user = os.environ.get('GODON_ARCHIVE_DB_USER', 'postgres')
        host = os.environ.get('GODON_ARCHIVE_DB_SERVICE_HOST', 'localhost')
        port = os.environ.get('GODON_ARCHIVE_DB_SERVICE_PORT', '5432')
        return f"postgresql://{user}:{pw}@{host}:{port}/archive_db"

    def _with_shared_db(self, fn, description: str, max_retries: int = 4):
        last_error = None
        for attempt in range(max_retries):
            conn = None
            try:
                import psycopg2
                conn = psycopg2.connect(self._get_shared_db_url())
                conn.autocommit = True
                return fn(conn)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1 and self._is_retryable_error(e):
                    wait_time = 2 ** attempt
                    logger.warning(f"{description} attempt {attempt + 1}/{max_retries} failed: {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
        raise last_error

    def _compute_baseline_params(self, settings: Dict[str, Any]) -> Dict[str, float]:
        baseline = {}
        for key, spec in settings.items():
            constraints = spec.get('constraints', [])
            if constraints and len(constraints) > 0:
                lower = constraints[0].get('lower', 0)
                upper = constraints[0].get('upper', 1)
                baseline[key] = (lower + upper) / 2.0
        return baseline

    def _register_interference_breeder(self):
        def op(conn):
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS interference_active_breeders (
                    breeder_id VARCHAR(255) PRIMARY KEY,
                    last_seen TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute(
                "INSERT INTO interference_active_breeders (breeder_id, last_seen) "
                "VALUES (%s, NOW()) ON CONFLICT (breeder_id) DO UPDATE SET last_seen = NOW()",
                (self.breeder_id,)
            )
            cur.close()
        try:
            self._with_shared_db(op, "register_interference_breeder")
        except Exception as e:
            logger.warning(f"Failed to register for interference detection: {e}")

    def _heartbeat_interference(self):
        if time.time() - self._last_heartbeat_ts < self._heartbeat_interval:
            return
        self._last_heartbeat_ts = time.time()
        self._register_interference_breeder()

    def _has_active_neighbors(self) -> bool:
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM interference_active_breeders "
                "WHERE breeder_id != %s AND last_seen > NOW() - INTERVAL '360 seconds'",
                (self.breeder_id,)
            )
            count = cur.fetchone()[0]
            cur.close()
            return count > 0
        try:
            return self._with_shared_db(op, "has_active_neighbors")
        except Exception:
            return False

    def _get_detection_mode(self) -> str:
        """Check detection_rounds table and return mode for this breeder.
        
        Returns: 'optimize' | 'hold' | 'impulse'
        """
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT sender_id FROM detection_rounds WHERE status = 'active' LIMIT 1"
            )
            row = cur.fetchone()
            cur.close()
            if row is None:
                return 'optimize'
            sender_id = row[0]
            if sender_id == self.breeder_id:
                return 'impulse'
            return 'hold'
        try:
            return self._with_shared_db(op, "get_detection_mode")
        except Exception as e:
            logger.warning(f"Failed to read detection mode: {e}")
            return 'optimize'

    def _complete_detection_round(self):
        """Mark the current active round as completed after impulse."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE detection_rounds SET status = 'completed', completed_at = NOW() "
                "WHERE sender_id = %s AND status = 'active'",
                (self.breeder_id,)
            )
            cur.close()
        try:
            self._with_shared_db(op, "complete_detection_round")
            logger.info("Detection round completed")
        except Exception as e:
            logger.warning(f"Failed to complete detection round: {e}")

    def _get_last_successful_params(self) -> Optional[Dict[str, Any]]:
        """Get effectuation-format params from the last completed trial for hold mode.

        Reads the stashed effectuation_params user_attr which preserves the format
        that suggest_params returned and the effectuator expects.
        Falls back to flat trial.params if no stashed attr found.
        """
        for trial in reversed(self.study.trials):
            if trial.state == TrialState.COMPLETE:
                stashed = trial.user_attrs.get('effectuation_params')
                if stashed:
                    return json.loads(stashed) if isinstance(stashed, str) else stashed
                return dict(trial.params)
        return None

    def _generate_impulse_params(self, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate params pushed to upper bounds for impulse mode.

        Takes the last strain-formatted params as template (correct structure,
        per-zone lists, etc), then overrides the top-3 params by range to
        their upper bounds from the config constraints.
        """
        if not self.study or not self.study.trials:
            return None

        # Get the last strain-formatted params as template
        stashed = None
        for trial in reversed(self.study.trials):
            if trial.state == TrialState.COMPLETE:
                stashed = trial.user_attrs.get('effectuation_params')
                break

        if not stashed:
            return None

        params = json.loads(stashed) if isinstance(stashed, str) else dict(stashed)

        # Collect upper bounds from config and override top-3 by range
        upper_bounds = self._collect_upper_bounds(settings)
        upper_bounds.sort(key=lambda x: x.get('range', 0), reverse=True)

        for ub in upper_bounds[:3]:
            name = ub['name']
            value = ub['upper']
            if ub.get('is_int'):
                value = int(value)
            if name in params:
                if isinstance(params[name], list):
                    params[name] = [value] * len(params[name])
                else:
                    params[name] = value

        return params if params else None

    def _collect_upper_bounds(self, obj: Any, depth: int = 0) -> list:
        """Recursively walk config tree and collect params with upper bounds."""
        results = []
        if not isinstance(obj, dict) or depth > 5:
            return results
        for key, val in obj.items():
            if key == 'zones' or not isinstance(val, dict):
                continue
            constraints = val.get('constraints')
            if isinstance(constraints, list) and constraints:
                first = constraints[0]
                lower = first.get('lower')
                upper = first.get('upper')
                if upper is not None and lower is not None:
                    is_int = first.get('step', 1) == int(first.get('step', 1))
                    results.append({
                        'name': key,
                        'upper': upper,
                        'lower': lower,
                        'range': upper - lower,
                        'is_int': is_int,
                    })
            else:
                results.extend(self._collect_upper_bounds(val, depth + 1))
        return results

    def _get_calibrated_impulse_params(self) -> Optional[Dict[str, Any]]:
        """Get calibrated impulse params using AIMD approach.

        First call: generate from strain template with upper-bound overrides.
        Subsequent calls: reuse cached params, scaled down on FAIL.
        """
        if self._calibrated_impulse_params is not None:
            return self._calibrated_impulse_params

        # Need at least one trial to use as strain-formatted template
        if not self.study or not self.study.trials:
            return None

        # Try stashed effectuation params from a prior optimize trial
        template = None
        for trial in reversed(self.study.trials):
            if trial.state == TrialState.COMPLETE:
                stashed = trial.user_attrs.get('effectuation_params')
                if stashed:
                    template = json.loads(stashed) if isinstance(stashed, str) else dict(stashed)
                    break

        if template is None:
            return None

        # Override top-3 params by range to upper bounds * scale
        upper_bounds = self._collect_upper_bounds(self.config.get('settings', {}))
        upper_bounds.sort(key=lambda x: x.get('range', 0), reverse=True)

        params = dict(template)
        for ub in upper_bounds[:3]:
            name = ub['name']
            value = ub['upper'] * self._impulse_scale
            if ub.get('is_int'):
                value = int(value)
            if name in params:
                if isinstance(params[name], list):
                    params[name] = [value] * len(params[name])
                else:
                    params[name] = value

        self._calibrated_impulse_params = params
        self._impulse_base_params = dict(template)  # Save base for AIMD re-scaling
        logger.info(f"Calibrated impulse params at scale {self._impulse_scale:.2f}: {list(params.keys())}")
        return params

    def _get_calibrated_hold_params(self) -> Optional[Dict[str, Any]]:
        """Get calibrated hold params — median of last N successful optimize trials.

        Uses stashed effectuation_params to get correct format.
        Returns None if not enough trials exist yet.
        """
        if self._calibrated_hold_params is not None:
            return self._calibrated_hold_params

        if not self.study or not self.study.trials:
            return None

        # Collect last 3 successful stashed params
        successful = []
        for trial in reversed(self.study.trials):
            if trial.state == TrialState.COMPLETE and len(successful) < 3:
                stashed = trial.user_attrs.get('effectuation_params')
                if stashed:
                    p = json.loads(stashed) if isinstance(stashed, str) else stashed
                    successful.append(p)

        if not successful:
            return None

        if len(successful) == 1:
            self._calibrated_hold_params = successful[0]
            return self._calibrated_hold_params

        # Median of numeric values across the successful trials
        import statistics
        result = dict(successful[0])
        for key in result:
            if isinstance(result[key], list):
                lists = [s[key] for s in successful if key in s and isinstance(s[key], list)]
                if lists:
                    result[key] = [statistics.median(vals) for vals in zip(*lists)]
            elif isinstance(result[key], (int, float)):
                vals = [s[key] for s in successful if key in s and isinstance(s[key], (int, float))]
                if vals:
                    result[key] = statistics.median(vals)

        self._calibrated_hold_params = result
        logger.info(f"Calibrated hold params from {len(successful)} trials (median): {list(result.keys())}")
        return result

    def _impulse_aimd_backoff(self):
        """AIMD multiplicative decrease — called on impulse FAIL."""
        self._impulse_scale *= 0.5
        # Re-scale from the original base params, don't pick a new template
        if self._impulse_base_params is not None:
            upper_bounds = self._collect_upper_bounds(self.config.get('settings', {}))
            upper_bounds.sort(key=lambda x: x.get('range', 0), reverse=True)
            params = dict(self._impulse_base_params)
            for ub in upper_bounds[:3]:
                name = ub['name']
                value = ub['upper'] * self._impulse_scale
                if ub.get('is_int'):
                    value = int(value)
                if name in params:
                    if isinstance(params[name], list):
                        params[name] = [value] * len(params[name])
                    else:
                        params[name] = value
            self._calibrated_impulse_params = params
            logger.warning(f"Impulse FAIL — AIMD backoff to scale {self._impulse_scale:.2f}")
        else:
            self._calibrated_impulse_params = None
            logger.warning(f"Impulse FAIL — AIMD backoff to scale {self._impulse_scale:.2f}, no base params")

        if self._impulse_scale < 0.1:
            logger.error("Impulse scale below minimum, giving up on detection")
            return False
        return True

    def _load_or_create_study(self) -> optuna.Study:
        parallel_workers = self.config.get('run', {}).get('parallel', 1)
        if parallel_workers > 1:
            study_name = f"{self.breeder_id}_{self.sampler_type}_study"
        else:
            study_name = f"{self.breeder_id}_study"

        directions = [obj.get('direction').lower() for obj in self.config.get('objectives', [])]

        storage = optuna.storages.RDBStorage(url=self._get_db_url())
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            logger.info(f"Loaded existing study: {study_name} with {len(study.trials)} trials")
        except (KeyError, ValueError):
            sampler = None
            if parallel_workers > 1:
                sampler = self._create_sampler(self.sampler_type)
                logger.info(f"Created study {study_name} with {self.sampler_type} sampler")

            try:
                study = optuna.create_study(
                    study_name=study_name,
                    directions=directions,
                    storage=storage,
                    sampler=sampler
                )
                logger.info(f"Created new study: {study_name}")
            except (optuna.exceptions.StorageInternalError, Exception) as e:
                logger.warning(f"Study creation failed ({e}), retrying as load...")
                time.sleep(2)
                storage = optuna.storages.RDBStorage(url=self._get_db_url())
                study = optuna.load_study(study_name=study_name, storage=storage)
                logger.info(f"Loaded study after creation race: {study_name} with {len(study.trials)} trials")

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

    def _run_reconnaissance(self, settings: Dict[str, Any] = None) -> Dict[str, float]:
        import wmill

        targets = self.config.get('effectuation', {}).get('targets', [])
        recon_path = f"f/reconnaissance/{self.config.get('reconnaissance', {}).get('type', 'prometheus')}"

        recon_result = wmill.run_script_by_path(
            recon_path,
            args={"context": self.config, "targets": targets, "settings": settings or {}}
        )

        metrics = recon_result.get('metrics', {})
        if not metrics:
            logger.error("No metrics returned from reconnaissance")
            return {obj.get('name'): float('inf') for obj in self.config.get('objectives', [])}

        self._last_metric_noise = recon_result.get('metric_noise', {})

        return metrics

    def _execute_trial(self, settings: Dict[str, Any]) -> Dict[str, float]:
        import wmill

        targets = self.config.get('effectuation', {}).get('targets', [])
        effectuator_path = f"f/effectuation/{self.config.get('effectuation', {}).get('type', 'ssh')}"

        logger.info(f"Effectuating {len(targets)} targets via {effectuator_path} with settings: {list(settings.keys())}")

        try:
            eff_result = wmill.run_script_by_path(
                effectuator_path,
                args={"context": self.config, "targets": targets, "settings": settings}
            )

            logger.info(f"Effectuation completed: {eff_result.get('status')}")

            successful = eff_result.get('successful_changes', 0)
            failed = eff_result.get('failed_changes', 0)

            if successful == 0 and failed > 0:
                failed_targets = [r.get('target_id', 'unknown') for r in eff_result.get('results', []) if not r.get('success', False)]
                logger.error(f"Effectuation completely failed for all targets: {failed_targets}")
                raise RuntimeError(f"Effectuation failed for all targets: {failed_targets}")

            if failed > 0:
                failed_targets = [r.get('target_id', 'unknown') for r in eff_result.get('results', []) if not r.get('success', False)]
                logger.warning(f"Effectuation partially failed: {failed}/{successful + failed} targets succeeded. Failed: {failed_targets}")

            metrics = self._run_reconnaissance(settings)

            return metrics

        except RuntimeError as e:
            logger.error(f"Trial execution failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Trial execution failed: {e}", exc_info=True)
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

        initial_state = {
            'state': 'normal',
            'consecutive_failures': 0,
            'last_successful_params': None,
            'rollback_strategy': self.rollback_config.get('strategy', 'standard'),
            'version': 0
        }

        self._retry_op(
            lambda: self.study.set_user_attr(state_key, json.dumps(initial_state)),
            "init_rollback_state"
        )
        logger.info(f"Initialized rollback state for target {self.target_id}: {initial_state}")

    def _get_rollback_state(self) -> Dict[str, Any]:
        state_key = self._get_rollback_state_key()
        state_json = self.study.user_attrs.get(state_key)

        if not state_json:
            logger.warning(f"No rollback state found for target {self.target_id}, initializing")
            self._init_rollback_state()
            state_json = self.study.user_attrs.get(state_key)

        return json.loads(state_json)

    def _update_rollback_state(self, new_state: Dict[str, Any]) -> bool:
        state_key = self._get_rollback_state_key()

        new_state['version'] = new_state.get('version', 0) + 1

        self._retry_op(
            lambda: self.study.set_user_attr(state_key, json.dumps(new_state)),
            "update_rollback_state"
        )
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
            effectuator_path = f"f/effectuation/{self.config.get('effectuation', {}).get('type', 'ssh')}"

            logger.info(f"Executing rollback effectuation for target {self.target_id}")
            result = wmill.run_script_by_path(
                effectuator_path,
                args={"context": self.config, "targets": [self.target], "settings": params_to_restore}
            )

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

                trial = self._retry_op(
                    lambda: self.study.ask(),
                    f"study.ask"
                )
                logger.info(f"Trial {trial.number} started")

                trial_start_time = time.time()

                params = None

                try:
                    # Check detection mode — overrides normal sampling
                    detection_mode = self._get_detection_mode()

                    # Warmup gate: need at least 1 COMPLETE optimize trial before detection
                    has_warmup = False
                    if detection_mode in ('hold', 'impulse') and self.study and self.study.trials:
                        for t in self.study.trials:
                            if t.state == TrialState.COMPLETE:
                                has_warmup = True
                                break

                    if detection_mode == 'hold' and has_warmup:
                        logger.info(f"Trial {trial.number}: DETECTION HOLD mode")
                        params = self._get_calibrated_hold_params()
                        if not params:
                            logger.warning("No calibrated hold params yet, optimizing normally")
                            detection_mode = None  # Fall through to optimize
                        else:
                            trial.set_user_attr('detection_mode', 'hold')

                    elif detection_mode == 'impulse' and has_warmup:
                        logger.info(f"Trial {trial.number}: DETECTION IMPULSE mode")
                        params = self._get_calibrated_impulse_params()
                        if not params:
                            logger.warning("No calibrated impulse params yet, optimizing normally")
                            detection_mode = None  # Fall through to optimize
                        else:
                            trial.set_user_attr('detection_mode', 'impulse')

                    if not params:
                        params = self.strain.suggest_params(trial, self.config.get('settings', {}))

                    if not self.watermark and self._has_active_neighbors():
                        self.watermark = create_watermark(self.config, self.config.get('settings', {}), breeder_uuid=self.breeder_uuid)
                        if self.watermark:
                            logger.info(f"Watermarking activated (lazy): {self.watermark.metadata()}")

                    # Watermark and detection modes are mutually exclusive —
                    # coordinated detection replaces the old watermark system.
                    if self.watermark and detection_mode not in ('hold', 'impulse'):
                        wm_complete = hasattr(self.watermark, 'is_complete') and self.watermark.is_complete()
                        if not wm_complete:
                            wm_params = self.watermark.generate(self._watermark_trial_idx, params)
                            # Determine if this is an active impulse trial
                            is_impulse = hasattr(self.watermark, 'is_impulse_trial') and self.watermark.is_impulse_trial(self._watermark_trial_idx)
                            if wm_params:
                                params = wm_params
                                # Store corrected params for Optuna — we'll inject them via
                                # study.add_trial() after study.tell() to avoid the sampler's
                                # original values being recorded for watermarked params.
                                self._wm_corrected_params = {
                                    pname: pval for pname, pval in wm_params.items()
                                    if pname in trial.params and trial.params[pname] != pval
                                }
                                wm_meta = self.watermark.metadata()
                                if is_impulse:
                                    wm_meta['active'] = True
                                trial.set_user_attr('watermark', json.dumps(wm_meta))
                                trial.set_user_attr('watermark_trial_idx', self._watermark_trial_idx)
                            else:
                                trial.set_user_attr('watermark', 'off')
                                trial.set_user_attr('watermark_trial_idx', self._watermark_trial_idx)
                            self._watermark_trial_idx += 1
                            logger.info(f"Watermark trial {self._watermark_trial_idx}: impulse={'ON' if is_impulse else 'off'}")

                    if params:
                        metrics = self._execute_trial(params)
                    else:
                        logger.info(f"Watermark off phase — resetting to baseline then reconnaissance")
                        try:
                            self._execute_trial(self._watermark_baseline)
                        except Exception as e:
                            logger.warning(f"Baseline effectuation failed (non-fatal): {e}")
                        metrics = self._run_reconnaissance()

                    guardrails_violated, violations = self._check_guardrails(metrics)

                    guardrails_config = self.config.get('guardrails', [])
                    if guardrails_config:
                        guardrail_readings = {}
                        for g in guardrails_config:
                            gname = g.get('name', 'unknown')
                            gval = metrics.get(gname)
                            if gval is not None:
                                guardrail_readings[gname] = {
                                    'value': gval,
                                    'hard_limit': g.get('hard_limit'),
                                    'violated': gval > g.get('hard_limit', float('inf')) if isinstance(g.get('hard_limit'), (int, float)) else False
                                }
                        if guardrail_readings:
                            trial.set_user_attr('guardrails', json.dumps(guardrail_readings))

                    if guardrails_violated:
                        logger.error(f"Trial {trial.number} failed guardrails: {violations}")
                        self._retry_op(
                            lambda: self.study.tell(trial, state=TrialState.FAIL),
                            f"study.tell FAIL (trial {trial.number})"
                        )
                        logger.info(f"Trial {trial.number} marked as FAILED (guardrail violation)")

                        for violation_msg in violations:
                            guardrail_name = violation_msg.split(':')[0] if ':' in violation_msg else 'unknown'
                            self.metrics.inc_guardrail_violation(guardrail_name)

                        self._handle_guardrail_violation(params)

                        # AIMD backoff: if this was an impulse trial, scale down
                        if detection_mode == 'impulse':
                            self._impulse_aimd_backoff()

                        self.metrics.inc_trial('failed')
                        self.metrics.inc_effectuation('failure')
                    else:
                        values = [metrics.get(obj.get('name')) for obj in self.config.get('objectives', [])]
                        logger.info(f"Trial {trial.number} metrics: {metrics}, resolved values: {values}")
                        if any(v is None for v in values):
                            logger.warning(f"Trial {trial.number} has None values for objectives: {[obj.get('name') for obj, v in zip(self.config.get('objectives', []), values) if v is None]}")
                        if self._last_metric_noise:
                            trial.set_user_attr('metric_noise', json.dumps(self._last_metric_noise))
                        self._retry_op(
                            lambda: self.study.tell(trial, values),
                            f"study.tell (trial {trial.number})"
                        )

                        # If watermarking was active, inject corrected params into Optuna.
                        # The original trial has the sampler's values for watermarked params.
                        # We add a corrected trial with the actual values so the sampler
                        # learns the true relationship.
                        corrected = getattr(self, '_wm_corrected_params', None)
                        if corrected:
                            try:
                                from optuna.trial import create_trial
                                all_params = dict(trial.params)
                                all_params.update(corrected)
                                corrected_trial = create_trial(
                                    params=all_params,
                                    distributions=dict(trial.distributions),
                                    values=values,
                                )
                                self.study.add_trial(corrected_trial)
                                logger.info(f"Injected corrected trial with watermark params")
                            except Exception as e:
                                logger.warning(f"Failed to inject corrected trial: {e}")
                            self._wm_corrected_params = None

                        trial_duration = time.time() - trial_start_time

                        logger.info(f"Trial {trial.number} completed with values: {values}")

                        self.metrics.inc_trial('complete', value=values[0] if values else None)
                        self.metrics.observe_trial_duration(trial_duration)
                        self.metrics.inc_effectuation('success')

                        # Diagnostic: mark trial as reached success path
                        trial.set_user_attr('success_path', 'reached')

                        if self.study.best_trials and self.study.best_trials[0].number == trial.number:
                            self.metrics.set_best_value(values[0] if values else 0)

                        # Stash effectuation-format params for hold mode retrieval
                        try:
                            trial.set_user_attr('effectuation_params', json.dumps(params))
                            trial.set_user_attr('stash_status', 'ok')
                        except Exception as stash_err:
                            trial.set_user_attr('stash_status', f'fail:{stash_err}')
                        self._handle_successful_trial(params)

                        # Complete detection round if this was an impulse
                        if detection_mode == 'impulse':
                            self._complete_detection_round()

                        if self.communication_callback:
                            frozen_trial = self.study.trials[-1]
                            self.communication_callback(self.study, frozen_trial)

                            coop_config = self.config.get('cooperation', {})
                            share_strategy = coop_config.get('share_strategy', 'unknown')
                            self.metrics.inc_trial_shared(share_strategy)

                    trial_count += 1
                    if trial_count % 5 == 0:
                        self._heartbeat_interference()
                        self._update_state()
                        self.metrics.set_total_trials(len(self.study.trials))
                        self.metrics.push()

                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
                    try:
                        trial.set_user_attr('error', f"{type(e).__name__}: {str(e)[:500]}")
                    except Exception:
                        pass
                    try:
                        self._retry_op(
                            lambda: self.study.tell(trial, state=TrialState.FAIL),
                            f"study.tell FAIL recovery (trial {trial.number})"
                        )
                        logger.info(f"Trial {trial.number} marked as FAILED")
                        # AIMD backoff on effectuation failure
                        if detection_mode == 'impulse':
                            self._impulse_aimd_backoff()
                    except (ValueError, Exception) as tell_err:
                        logger.info(f"Trial {trial.number} tell failed: {tell_err}")

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
