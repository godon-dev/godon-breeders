#extra_requirements:
#opentelemetry-api
#opentelemetry-sdk
#opentelemetry-exporter-otlp

import optuna
import random
from typing import Dict, Any, Optional, List
from optuna.trial import TrialState
from scipy.stats import percentileofscore
from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


class CommunicationCallback:
    
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
        self.logger = get_logger('communication-callback')
    
    def _share_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        try:
            study_names = study.get_all_study_names(storage=self.storage)
            
            for study_name in study_names:
                if study_name != study.study_name:
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
        if self.share_strategy == "probabilistic":
            return random.random() < self.com_probability
        
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.values]
        if len(completed_trials) < self.min_trials_for_filtering:
            self.logger.debug(f"Insufficient trials ({len(completed_trials)}) for quality filtering, sharing all")
            return True
        
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
        if self._should_share_trial(study, trial):
            self.logger.debug(f"Sharing trial {trial.number} (strategy: {self.share_strategy})")
            self._share_trial(study, trial)
        else:
            self.logger.debug(f"Skipping trial sharing for {trial.number} (strategy: {self.share_strategy})")
