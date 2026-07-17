
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
import sys
import os
import types
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from engine.detection_coordinator import DetectionCoordinator


def _mock_db(fn, desc):
    """Mock shared_db_fn — provides a mock connection."""
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value = cur
    cur.close = MagicMock()
    return fn(conn)


def _make_config(hold_params=None):
    cfg = {
        'objectives': [{'name': 'growth_rate', 'direction': 'maximize'}],
        'settings': {},
        'interference_detection': {
            'mode': 'active',
            'min_optimize_trials': 5,
            'hold_calib_trials': 3,
            'push_block_size': 5,
            'pause_block_size': 5,
            'cooldown_trials': 3,
        },
    }
    if hold_params:
        cfg['interference_detection']['hold_params'] = hold_params
    return cfg


def _make_coord(breeder_id='sender-1', hold_params=None):
    coord = DetectionCoordinator(
        breeder_id=breeder_id,
        config=_make_config(hold_params),
        shared_db_fn=_mock_db,
        collect_upper_bounds_fn=lambda cfg: [
            {'name': 'heating', 'lower': 5.0, 'upper': 40.0, 'range': 35.0, 'is_int': False},
            {'name': 'light', 'lower': 0.0, 'upper': 1000.0, 'range': 1000.0, 'is_int': False},
            {'name': 'co2', 'lower': 0.0, 'upper': 20.0, 'range': 20.0, 'is_int': False},
        ],
    )
    coord._initialized = True
    return coord


class MockTrial:
    def __init__(self, number):
        self.number = number
        self._attrs = {}

    def set_user_attr(self, key, value):
        self._attrs[key] = value

    @property
    def user_attrs(self):
        return self._attrs


class TestHoldParamsSkipCalib:
    """When hold_params come from config, skip flatness search entirely."""

    def test_locks_on_first_trial(self):
        """Params from config must be locked immediately — no MIN_CALIB_SAMPLES wait."""
        coord = _make_coord(hold_params={'heating': 20.0, 'light': 300.0})
        coord.state = coord.HOLD_CALIB
        coord._hold_calib_count = 0

        with patch.object(coord, '_set_lease_phase'), \
             patch.object(coord, '_signal_ready'), \
             patch.object(coord, '_check_all_ready', return_value=False), \
             patch.object(coord, '_evaluate_hold_flatness') as eval_mock:
            trial = MockTrial(0)
            result = coord._handle_hold_calib(trial)

            assert coord._calib_params_locked is True, \
                "Params from config should be locked on first trial"
            assert result['mode'] == 'hold'
            assert result['params'] == {'heating': 20.0, 'light': 300.0}
            eval_mock.assert_not_called(), \
                "Flatness evaluation should NOT run for config-specified params"

    def test_does_not_modify_params(self):
        """Config hold_params must never be adjusted by _adjust_hold_params."""
        hold = {'heating': 20.0, 'light': 300.0, 'co2': 10.0}
        coord = _make_coord(hold_params=hold)
        coord.state = coord.HOLD_CALIB
        coord._hold_calib_count = 0

        with patch.object(coord, '_set_lease_phase'), \
             patch.object(coord, '_signal_ready'), \
             patch.object(coord, '_check_all_ready', return_value=False):
            trial = MockTrial(0)
            coord._handle_hold_calib(trial)

            # Run several more trials — params should never change
            for i in range(1, 10):
                trial = MockTrial(i)
                coord._handle_hold_calib(trial)

            assert coord._get_neutral_params() == hold, \
                "Config hold_params were modified during hold_calib"

    def test_skips_flatness_eval_completely(self):
        """Even after MIN_CALIB_SAMPLES, flatness check should not fire."""
        coord = _make_coord(hold_params={'heating': 20.0})
        coord.state = coord.HOLD_CALIB

        with patch.object(coord, '_set_lease_phase'), \
             patch.object(coord, '_signal_ready'), \
             patch.object(coord, '_check_all_ready', return_value=False), \
             patch.object(coord, '_evaluate_hold_flatness') as eval_mock, \
             patch.object(coord, '_adjust_hold_params') as adjust_mock:
            for i in range(coord.MIN_CALIB_SAMPLES + 5):
                trial = MockTrial(i)
                coord._handle_hold_calib(trial)

            eval_mock.assert_not_called()
            adjust_mock.assert_not_called()

    def test_signals_ready_immediately(self):
        """Readiness should be signaled on first trial, not after flatness pass."""
        coord = _make_coord(hold_params={'heating': 20.0})
        coord.state = coord.HOLD_CALIB
        coord._hold_calib_count = 0

        with patch.object(coord, '_set_lease_phase'), \
             patch.object(coord, '_check_all_ready', return_value=False):
            with patch.object(coord, '_signal_ready') as signal_mock:
                trial = MockTrial(0)
                coord._handle_hold_calib(trial)

                signal_mock.assert_called_once_with(coord.HOLD_CALIB)

    def test_proceeds_to_impulse_when_partner_ready(self):
        """When config hold_params + partner ready, go straight to impulse on trial 2."""
        coord = _make_coord(hold_params={'heating': 20.0})
        coord.state = coord.HOLD_CALIB
        coord._hold_calib_count = 0

        with patch.object(coord, '_set_lease_phase'), \
             patch.object(coord, '_signal_ready'), \
             patch.object(coord, '_check_all_ready', return_value=True):
            # Trial 1: locks params, signals ready
            trial1 = MockTrial(0)
            coord._handle_hold_calib(trial1)
            assert coord._calib_params_locked is True
            assert coord.state == coord.HOLD_CALIB  # not enough yet

            # Trial 2: barrier passes → should transition to IMPULSE_CALIB
            trial2 = MockTrial(1)
            coord._handle_hold_calib(trial2)

            assert coord.state == coord.IMPULSE_CALIB, \
                f"Should be in IMPULSE_CALIB, got {coord.state}"


class TestComputedParamsStillCalibrate:
    """Without config hold_params, calibration search runs normally."""

    def test_flatness_eval_runs(self):
        """When no config hold_params, flatness evaluation must run after MIN_CALIB_SAMPLES."""
        coord = _make_coord(hold_params=None)
        coord.state = coord.HOLD_CALIB

        # Set up neutral params via callback (simulates midpoint computation)
        coord._neutral_params = {'heating': 22.5, 'light': 500.0, 'co2': 10.0}

        with patch.object(coord, '_set_lease_phase'), \
             patch.object(coord, '_signal_ready'), \
             patch.object(coord, '_check_all_ready', return_value=False), \
             patch.object(coord, '_evaluate_hold_flatness', return_value=False) as eval_mock, \
             patch.object(coord, '_adjust_hold_params'):
            for i in range(coord.MIN_CALIB_SAMPLES):
                trial = MockTrial(i)
                coord._handle_hold_calib(trial)

            eval_mock.assert_called(), \
                "Flatness eval should run when params are NOT from config"

    def test_adjust_runs_on_failure(self):
        """When flatness fails and params are computed, _adjust_hold_params should run."""
        coord = _make_coord(hold_params=None)
        coord.state = coord.HOLD_CALIB
        coord._neutral_params = {'heating': 22.5, 'light': 500.0, 'co2': 10.0}

        with patch.object(coord, '_set_lease_phase'), \
             patch.object(coord, '_signal_ready'), \
             patch.object(coord, '_check_all_ready', return_value=False), \
             patch.object(coord, '_evaluate_hold_flatness', return_value=False), \
             patch.object(coord, '_adjust_hold_params') as adjust_mock:
            for i in range(coord.MIN_CALIB_SAMPLES):
                trial = MockTrial(i)
                coord._handle_hold_calib(trial)

            adjust_mock.assert_called(), \
                "_adjust_hold_params should run when flatness fails for computed params"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
