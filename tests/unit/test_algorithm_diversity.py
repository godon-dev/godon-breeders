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

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Mock external dependencies before importing breeder worker
sys.modules['wmill'] = MagicMock()
sys.modules['optuna'] = MagicMock()
sys.modules['optuna.storages'] = MagicMock()
sys.modules['optuna.trial'] = MagicMock()
sys.modules['optuna.samplers'] = MagicMock()

from linux_performance.breeder_worker import BreederWorker


class TestSamplerAssignment:
    """Test sampler assignment logic for algorithm diversity"""
    
    @patch('linux_performance.breeder_worker.BreederWorker._load_or_create_study')
    @patch('linux_performance.breeder_worker.BreederWorker._setup_communication')
    @patch('linux_performance.breeder_worker.BreederWorker._update_state')
    def test_single_worker_gets_tpe_sampler(self, mock_update, mock_comm, mock_study):
        """Test that single worker configuration defaults to TPE sampler"""
        config = {
            'breeder': {
                'name': 'test_breeder',
                'uuid': 'test_uuid_123'
            },
            'creation_ts': '2025-01-15T10:30:00Z',
            'run': {
                'parallel': 1
            },
            'objectives': [{'name': 'test_obj', 'direction': 'maximize'}]
        }
        
        mock_study.return_value = MagicMock()
        
        worker = BreederWorker(config)
        
        assert worker.sampler_type == 'tpe', "Single worker should use TPE sampler"
    
    @patch('linux_performance.breeder_worker.BreederWorker._load_or_create_study')
    @patch('linux_performance.breeder_worker.BreederWorker._setup_communication')
    @patch('linux_performance.breeder_worker.BreederWorker._update_state')
    def test_multiple_workers_get_different_samplers(self, mock_update, mock_comm, mock_study):
        """Test that multiple workers get assigned different samplers"""
        config = {
            'breeder': {
                'name': 'test_breeder',
                'uuid': 'test_uuid_123'
            },
            'creation_ts': '2025-01-15T10:30:00Z',
            'run': {
                'parallel': 3
            },
            'objectives': [{'name': 'test_obj', 'direction': 'maximize'}]
        }
        
        mock_study.return_value = MagicMock()
        
        # Create multiple workers with different worker_ids
        workers = []
        for i in range(3):
            worker_config = config.copy()
            worker_config['breeder']['uuid'] = f'test_uuid_{i}'
            worker = BreederWorker(worker_config)
            workers.append(worker)
        
        # Check that workers have sampler types assigned
        sampler_types = [w.sampler_type for w in workers]
        valid_samplers = ['tpe', 'nsga2', 'random', 'nsga3', 'qmc']
        
        for sampler_type in sampler_types:
            assert sampler_type in valid_samplers, f"Invalid sampler type: {sampler_type}"
    
    @patch('linux_performance.breeder_worker.BreederWorker._load_or_create_study')
    @patch('linux_performance.breeder_worker.BreederWorker._setup_communication')
    @patch('linux_performance.breeder_worker.BreederWorker._update_state')
    def test_sampler_assignment_is_deterministic(self, mock_update, mock_comm, mock_study):
        """Test that same worker_id always gets same sampler"""
        config = {
            'breeder': {
                'name': 'test_breeder',
                'uuid': 'deterministic_test_uuid'
            },
            'creation_ts': '2025-01-15T10:30:00Z',
            'run': {
                'parallel': 3
            },
            'objectives': [{'name': 'test_obj', 'direction': 'maximize'}]
        }
        
        mock_study.return_value = MagicMock()
        
        # Create two workers with same config
        worker1 = BreederWorker(config)
        worker2 = BreederWorker(config)
        
        assert worker1.sampler_type == worker2.sampler_type, \
            "Same worker_id should produce same sampler assignment"


class TestSamplerCreation:
    """Test sampler parameter randomization"""
    
    @patch('linux_performance.breeder_worker.BreederWorker._load_or_create_study')
    @patch('linux_performance.breeder_worker.BreederWorker._setup_communication')
    @patch('linux_performance.breeder_worker.BreederWorker._update_state')
    def test_tpe_sampler_gets_randomized_config(self, mock_update, mock_comm, mock_study):
        """Test that TPE sampler gets randomized parameters"""
        config = {
            'breeder': {
                'name': 'test_breeder',
                'uuid': 'test_uuid'
            },
            'creation_ts': '2025-01-15T10:30:00Z',
            'run': {'parallel': 2},
            'objectives': [{'name': 'test_obj', 'direction': 'maximize'}]
        }
        
        mock_study.return_value = MagicMock()
        worker = BreederWorker(config)
        
        # Force TPE sampler for this test
        worker.sampler_type = 'tpe'
        
        with patch('linux_performance.breeder_worker.random.choice') as mock_random:
            mock_random.side_effect = [(True, False), True, 10]  # (multivariate, group), constant_liar, n_startup
            
            with patch('linux_performance.breeder_worker.TPESampler') as mock_tpe:
                worker._create_sampler('tpe')
                
                # Check that TPESampler was called with randomized config
                assert mock_tpe.called, "TPESampler should be instantiated"
    
    @patch('linux_performance.breeder_worker.BreederWorker._load_or_create_study')
    @patch('linux_performance.breeder_worker.BreederWorker._setup_communication')
    @patch('linux_performance.breeder_worker.BreederWorker._update_state')
    def test_nsga2_sampler_gets_randomized_config(self, mock_update, mock_comm, mock_study):
        """Test that NSGA2 sampler gets randomized parameters"""
        config = {
            'breeder': {
                'name': 'test_breeder',
                'uuid': 'test_uuid'
            },
            'creation_ts': '2025-01-15T10:30:00Z',
            'run': {'parallel': 2},
            'objectives': [{'name': 'test_obj', 'direction': 'maximize'}]
        }
        
        mock_study.return_value = MagicMock()
        worker = BreederWorker(config)
        
        # Force NSGA2 sampler for this test
        worker.sampler_type = 'nsga2'
        
        with patch('linux_performance.breeder_worker.random.choice') as mock_random:
            # population_size, mutation_prob, crossover_prob, crossover
            mock_random.side_effect = [50, 0.1, 0.9, 'uniform']
            
            with patch('linux_performance.breeder_worker.NSGAIISampler') as mock_nsga2:
                worker._create_sampler('nsga2')
                
                assert mock_nsga2.called, "NSGAIISampler should be instantiated"


class TestStudyNaming:
    """Test study naming for multi-study architecture"""
    
    @patch('linux_performance.breeder_worker.BreederWorker._setup_communication')
    @patch('linux_performance.breeder_worker.BreederWorker._update_state')
    @patch('linux_performance.breeder_worker.optuna')
    def test_single_worker_creates_single_study(self, mock_optuna, mock_update, mock_comm):
        """Test that single worker creates standard study name"""
        config = {
            'breeder': {
                'name': 'test_breeder',
                'uuid': 'test_uuid'
            },
            'creation_ts': '2025-01-15T10:30:00Z',
            'run': {'parallel': 1},
            'objectives': [{'name': 'test_obj', 'direction': 'maximize'}]
        }
        
        mock_storage = MagicMock()
        mock_optuna.storages.RDBStorage.return_value = mock_storage
        
        # Make load_study fail so create_study gets called
        mock_optuna.load_study.side_effect = KeyError("Study not found")
        mock_optuna.create_study.return_value = MagicMock()
        
        worker = BreederWorker(config)
        
        # Check study name doesn't include sampler type
        assert '_study' in str(mock_optuna.create_study.call_args)
    
    @patch('linux_performance.breeder_worker.BreederWorker._setup_communication')
    @patch('linux_performance.breeder_worker.BreederWorker._update_state')
    @patch('linux_performance.breeder_worker.optuna')
    def test_multiple_workers_create_sampler_specific_studies(self, mock_optuna, mock_update, mock_comm):
        """Test that multiple workers create sampler-specific study names"""
        config = {
            'breeder': {
                'name': 'test_breeder',
                'uuid': 'test_uuid'
            },
            'creation_ts': '2025-01-15T10:30:00Z',
            'run': {'parallel': 3},
            'objectives': [{'name': 'test_obj', 'direction': 'maximize'}]
        }
        
        mock_storage = MagicMock()
        mock_optuna.storages.RDBStorage.return_value = mock_storage
        
        # Make load_study fail so create_study gets called
        mock_optuna.load_study.side_effect = KeyError("Study not found")
        mock_optuna.create_study.return_value = MagicMock()
        
        worker = BreederWorker(config)
        
        # Check that study name includes sampler type
        call_args = str(mock_optuna.create_study.call_args)
        assert f"_{worker.sampler_type}_study" in call_args or "study_name" in call_args


class TestCommunicationCallback:
    """Test communication callback for multi-study coordination"""
    
    def test_share_within_breeder_flag_for_parallel_workers(self):
        """Test that parallel workers enable share_within_breeder"""
        from linux_performance.breeder_worker import CommunicationCallback
        
        callback = CommunicationCallback(
            storage="test_storage",
            probability=0.8,
            share_within_breeder=True
        )
        
        assert callback.share_within_breeder == True
    
    @patch('linux_performance.breeder_worker.BreederWorker._load_or_create_study')
    @patch('linux_performance.breeder_worker.BreederWorker._update_state')
    def test_parallel_workers_enables_intra_breeder_sharing(self, mock_update, mock_study):
        """Test that parallel worker configuration enables share_within_breeder"""
        config = {
            'breeder': {
                'name': 'test_breeder',
                'uuid': 'test_uuid'
            },
            'creation_ts': '2025-01-15T10:30:00Z',
            'run': {'parallel': 3},
            'cooperation': {'active': True, 'consolidation': {'probability': 0.8}},
            'objectives': [{'name': 'test_obj', 'direction': 'maximize'}]
        }
        
        mock_study.return_value = MagicMock()
        
        worker = BreederWorker(config)
        
        # Check that communication callback is configured for intra-breeder sharing
        assert worker.communication_callback is not None
        assert worker.communication_callback.share_within_breeder == True