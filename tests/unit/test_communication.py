import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from engine.communication import CommunicationCallback


class TestProbabilisticStrategy:
    def test_shares_when_random_below_probability(self):
        cb = CommunicationCallback(storage="sqlite:///test.db", share_strategy="probabilistic", probability=0.8)
        study = MagicMock()
        trial = MagicMock()
        trial.number = 1
        trial.values = [0.5]

        with patch('engine.communication.random.random', return_value=0.5):
            assert cb._should_share_trial(study, trial) is True

    def test_skips_when_random_above_probability(self):
        cb = CommunicationCallback(storage="sqlite:///test.db", share_strategy="probabilistic", probability=0.8)
        study = MagicMock()
        trial = MagicMock()
        trial.number = 1
        trial.values = [0.5]

        with patch('engine.communication.random.random', return_value=0.9):
            assert cb._should_share_trial(study, trial) is False


class TestBestStrategy:
    def _setup_study_with_trials(self, trial_values):
        from optuna.trial import TrialState
        study = MagicMock()
        completed_trials = []
        for i, val in enumerate(trial_values):
            t = MagicMock()
            t.state = TrialState.COMPLETE
            t.values = [val]
            completed_trials.append(t)
        study.trials = completed_trials
        return study

    def test_shares_top_percentile_trial(self):
        cb = CommunicationCallback(
            storage="sqlite:///test.db",
            share_strategy="best",
            top_percentile=0.2,
            min_trials_for_filtering=5,
        )
        study = self._setup_study_with_trials([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        trial = MagicMock()
        trial.values = [1.5]

        with patch('engine.communication.percentileofscore', return_value=95.0):
            assert cb._should_share_trial(study, trial) is True

    def test_skips_low_percentile_trial(self):
        cb = CommunicationCallback(
            storage="sqlite:///test.db",
            share_strategy="best",
            top_percentile=0.2,
            min_trials_for_filtering=5,
        )
        study = self._setup_study_with_trials([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        trial = MagicMock()
        trial.values = [8.0]

        with patch('engine.communication.percentileofscore', return_value=30.0):
            assert cb._should_share_trial(study, trial) is False

    def test_shares_all_when_insufficient_trials(self):
        cb = CommunicationCallback(
            storage="sqlite:///test.db",
            share_strategy="best",
            min_trials_for_filtering=10,
        )
        study = MagicMock()
        study.trials = []
        trial = MagicMock()
        trial.values = [5.0]

        assert cb._should_share_trial(study, trial) is True


class TestWorstStrategy:
    def test_shares_bottom_percentile_trial(self):
        cb = CommunicationCallback(
            storage="sqlite:///test.db",
            share_strategy="worst",
            bottom_percentile=0.2,
            min_trials_for_filtering=5,
        )
        from optuna.trial import TrialState
        study = MagicMock()
        mock_trials = []
        for _ in range(10):
            t = MagicMock()
            t.state = TrialState.COMPLETE
            t.values = [1.0]
            mock_trials.append(t)
        study.trials = mock_trials
        trial = MagicMock()
        trial.values = [0.1]

        with patch('engine.communication.percentileofscore', return_value=5.0):
            assert cb._should_share_trial(study, trial) is True

    def test_skips_high_percentile_trial(self):
        cb = CommunicationCallback(
            storage="sqlite:///test.db",
            share_strategy="worst",
            bottom_percentile=0.2,
            min_trials_for_filtering=5,
        )
        from optuna.trial import TrialState
        study = MagicMock()
        mock_trials = []
        for _ in range(10):
            t = MagicMock()
            t.state = TrialState.COMPLETE
            t.values = [1.0]
            mock_trials.append(t)
        study.trials = mock_trials
        trial = MagicMock()
        trial.values = [0.5]

        with patch('engine.communication.percentileofscore', return_value=50.0):
            assert cb._should_share_trial(study, trial) is False


class TestExtremesStrategy:
    def test_shares_top_extreme(self):
        cb = CommunicationCallback(
            storage="sqlite:///test.db",
            share_strategy="extremes",
            top_percentile=0.2,
            bottom_percentile=0.2,
            min_trials_for_filtering=5,
        )
        from optuna.trial import TrialState
        study = MagicMock()
        mock_trials = []
        for _ in range(10):
            t = MagicMock()
            t.state = TrialState.COMPLETE
            t.values = [1.0]
            mock_trials.append(t)
        study.trials = mock_trials
        trial = MagicMock()
        trial.values = [0.1]

        with patch('engine.communication.percentileofscore', return_value=95.0):
            assert cb._should_share_trial(study, trial) is True

    def test_shares_bottom_extreme(self):
        cb = CommunicationCallback(
            storage="sqlite:///test.db",
            share_strategy="extremes",
            top_percentile=0.2,
            bottom_percentile=0.2,
            min_trials_for_filtering=5,
        )
        from optuna.trial import TrialState
        study = MagicMock()
        mock_trials = []
        for _ in range(10):
            t = MagicMock()
            t.state = TrialState.COMPLETE
            t.values = [1.0]
            mock_trials.append(t)
        study.trials = mock_trials
        trial = MagicMock()
        trial.values = [0.5]

        with patch('engine.communication.percentileofscore', return_value=5.0):
            assert cb._should_share_trial(study, trial) is True

    def test_skips_middle_percentile(self):
        cb = CommunicationCallback(
            storage="sqlite:///test.db",
            share_strategy="extremes",
            top_percentile=0.2,
            bottom_percentile=0.2,
            min_trials_for_filtering=5,
        )
        from optuna.trial import TrialState
        study = MagicMock()
        mock_trials = []
        for _ in range(10):
            t = MagicMock()
            t.state = TrialState.COMPLETE
            t.values = [1.0]
            mock_trials.append(t)
        study.trials = mock_trials
        trial = MagicMock()
        trial.values = [0.5]

        with patch('engine.communication.percentileofscore', return_value=50.0):
            assert cb._should_share_trial(study, trial) is False


class TestUnknownStrategy:
    def test_unknown_strategy_defaults_to_share(self):
        cb = CommunicationCallback(storage="sqlite:///test.db", share_strategy="unknown")
        from optuna.trial import TrialState
        study = MagicMock()
        mock_trials = []
        for _ in range(10):
            t = MagicMock()
            t.state = TrialState.COMPLETE
            t.values = [1.0]
            mock_trials.append(t)
        study.trials = mock_trials
        trial = MagicMock()
        trial.values = [5.0]

        assert cb._should_share_trial(study, trial) is True


class TestShareTrial:
    def test_shares_to_other_studies(self):
        cb = CommunicationCallback(storage="sqlite:///test.db", share_within_breeder=True)
        study = MagicMock()
        study.study_name = 'breeder_a_tpe_study'
        study.get_all_study_names.return_value = ['breeder_a_tpe_study', 'breeder_a_nsga2_study']

        cooperating = MagicMock()
        trial = MagicMock()
        trial.number = 42

        with patch('engine.communication.optuna.load_study', return_value=cooperating):
            cb._share_trial(study, trial)
            cooperating.add_trial.assert_called_once_with(trial)

    def test_skips_own_study(self):
        cb = CommunicationCallback(storage="sqlite:///test.db", share_within_breeder=True)
        study = MagicMock()
        study.study_name = 'breeder_a_tpe_study'
        study.get_all_study_names.return_value = ['breeder_a_tpe_study']

        trial = MagicMock()
        trial.number = 1

        with patch('engine.communication.optuna') as mock_optuna:
            cb._share_trial(study, trial)
            mock_optuna.load_study.assert_not_called()

    def test_share_within_breeder_false_skips_same_breeder(self):
        cb = CommunicationCallback(storage="sqlite:///test.db", share_within_breeder=False)
        study = MagicMock()
        study.study_name = 'alpha_tpe_study'
        study.get_all_study_names.return_value = ['alpha_tpe_study', 'alpha_nsga2_study', 'beta_tpe_study']

        cooperating = MagicMock()
        trial = MagicMock()
        trial.number = 1

        with patch('engine.communication.optuna.load_study', return_value=cooperating) as mock_load:
            cb._share_trial(study, trial)
            mock_load.assert_called_once()
            called_study_name = mock_load.call_args[1]['study_name']
            assert called_study_name == 'beta_tpe_study'

    def test_handles_share_failure_gracefully(self):
        cb = CommunicationCallback(storage="sqlite:///test.db", share_within_breeder=True)
        study = MagicMock()
        study.study_name = 'breeder_a_tpe_study'
        study.get_all_study_names.return_value = ['breeder_a_tpe_study', 'breeder_b_study']

        trial = MagicMock()
        trial.number = 1

        with patch('engine.communication.optuna.load_study', side_effect=Exception("DB error")):
            cb._share_trial(study, trial)

    def test_handles_study_list_failure(self):
        cb = CommunicationCallback(storage="sqlite:///test.db")
        study = MagicMock()
        study.get_all_study_names.side_effect = Exception("Connection error")
        trial = MagicMock()

        cb._share_trial(study, trial)


class TestCallbackInvocation:
    def test_call_delegates_to_share_when_should_share(self):
        cb = CommunicationCallback(storage="sqlite:///test.db", share_strategy="probabilistic", probability=1.0)
        study = MagicMock()
        trial = MagicMock()
        trial.number = 1

        with patch.object(cb, '_share_trial') as mock_share:
            cb(study, trial)
            mock_share.assert_called_once_with(study, trial)

    def test_call_skips_share_when_should_not_share(self):
        cb = CommunicationCallback(storage="sqlite:///test.db", share_strategy="probabilistic", probability=0.0)
        study = MagicMock()
        trial = MagicMock()
        trial.number = 1

        with patch.object(cb, '_share_trial') as mock_share:
            cb(study, trial)
            mock_share.assert_not_called()
