#
# Copyright (c) 2019 Matthias Tafelmeier.
#
# This file is part of godon
#
# godon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# godon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this godon. If not, see <http://www.gnu.org/licenses/>.
#

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

sys.modules['wmill'] = MagicMock()
sys.modules['optuna'] = MagicMock()
sys.modules['optuna.storages'] = MagicMock()
sys.modules['optuna.trial'] = MagicMock()
from optuna.trial import TrialState
sys.modules['optuna.samplers'] = MagicMock()
sys.modules['prometheus_api_client'] = MagicMock()
sys.modules['prometheus_api_client.exceptions'] = MagicMock()
sys.modules['prometheus_client'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()

from engine.breeder_worker import BreederWorker


def _base_config(**overrides):
    config = {
        'breeder': {
            'name': 'test_breeder',
            'uuid': 'test-uuid-123',
            'type': 'linux_performance',
        },
        'creation_ts': '2025-01-15T10:30:00Z',
        'run': {'parallel': 1},
        'objectives': [{'name': 'throughput', 'direction': 'minimize'}],
        'effectuation': {'targets': [], 'type': 'ssh'},
        'reconnaissance': {'type': 'prometheus'},
    }
    config.update(overrides)
    return config


def _mock_study(trials=None):
    study = MagicMock()
    study.trials = trials or []
    study.study_name = 'test_study'
    study.user_attrs = {}

    def set_attr(key, value):
        study.user_attrs[key] = value

    study.set_user_attr = set_attr
    return study


def _create_worker(breeder_id='test-uuid-123', **config_overrides):
    config = _base_config(**config_overrides)
    config['breeder']['uuid'] = breeder_id
    study = _mock_study()

    with patch.object(BreederWorker, '_load_or_create_study', return_value=study), \
         patch.object(BreederWorker, '_setup_communication', return_value=None), \
         patch.object(BreederWorker, '_update_state'), \
         patch.object(BreederWorker, '_register_interference_breeder'), \
         patch('engine.breeder_worker.load_strain', return_value=MagicMock()):
        worker = BreederWorker(config)
    return worker


class TestGetDetectionMode:
    """Test that breeder reads detection mode from the coordination table"""

    def test_returns_optimize_when_no_active_rounds(self):
        """No active rounds means normal optimization mode"""
        worker = _create_worker()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(worker, '_with_shared_db', return_value='optimize') as mock_db:
            # Simulate the op function behavior
            def side_effect(fn, desc):
                return fn(mock_conn)
            mock_db.side_effect = side_effect

            result = worker._get_detection_mode()

        assert result == 'optimize'

    def test_returns_impulse_when_this_breeder_is_sender(self):
        """This breeder is the active sender — should impulse"""
        worker = _create_worker(breeder_id='sender-abc')

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # First query (own round check): found — this breeder has an active round
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(worker, '_with_shared_db') as mock_db:
            def side_effect(fn, desc):
                return fn(mock_conn)
            mock_db.side_effect = side_effect

            result = worker._get_detection_mode()

        assert result == 'impulse'

    def test_returns_hold_when_another_breeder_is_sender(self):
        """Another breeder is sender — this one holds still"""
        worker = _create_worker(breeder_id='receiver-xyz')

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # First query (own round check): None — this breeder has no active round
        # Second query (any sender check): another breeder is sender
        mock_cursor.fetchone.side_effect = [None, ('sender-abc',)]
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(worker, '_with_shared_db') as mock_db:
            def side_effect(fn, desc):
                return fn(mock_conn)
            mock_db.side_effect = side_effect

            result = worker._get_detection_mode()

        assert result == 'hold'

    def test_returns_optimize_on_db_error(self):
        """DB failure falls back to optimize (safe default)"""
        worker = _create_worker()

        with patch.object(worker, '_with_shared_db', side_effect=Exception("connection refused")):
            result = worker._get_detection_mode()

        assert result == 'optimize'


class TestCompleteDetectionRound:
    """Test that breeder marks its detection round as completed"""

    def test_completes_active_round_for_sender(self):
        """After impulse, sender marks its round completed"""
        worker = _create_worker(breeder_id='sender-abc')

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(worker, '_with_shared_db') as mock_db:
            def side_effect(fn, desc):
                return fn(mock_conn)
            mock_db.side_effect = side_effect

            worker._complete_detection_round()

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'completed'" in sql
        assert "sender_id = %s" in sql
        assert "status = 'active'" in sql

    def test_complete_round_handles_db_error_gracefully(self):
        """DB failure during round completion should not crash"""
        worker = _create_worker(breeder_id='sender-abc')

        with patch.object(worker, '_with_shared_db', side_effect=Exception("timeout")):
            # Should not raise
            worker._complete_detection_round()


class TestNoTableCreation:
    """Verify that the breeder does NOT create or modify the detection_rounds table"""

    def test_no_ensure_detection_rounds_table_method(self):
        """The breeder should not have _ensure_detection_rounds_table"""
        worker = _create_worker()
        assert not hasattr(worker, '_ensure_detection_rounds_table'), \
            "Breeder should not have _ensure_detection_rounds_table — controller owns table creation"

    def test_detection_mode_only_reads(self):
        """_get_detection_mode only SELECTs, never CREATEs or INSERTs"""
        worker = _create_worker()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(worker, '_with_shared_db') as mock_db:
            def side_effect(fn, desc):
                return fn(mock_conn)
            mock_db.side_effect = side_effect

            worker._get_detection_mode()

        sql = mock_cursor.execute.call_args[0][0]
        assert sql.strip().startswith('SELECT')
        assert 'CREATE' not in sql.upper()
        assert 'INSERT' not in sql.upper()
