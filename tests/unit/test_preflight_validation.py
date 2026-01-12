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
# along with this godon.  If not, see <http://www.gnu.org/licenses/>.
#

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Mock wmill before any imports
from unittest.mock import MagicMock
sys.modules['wmill'] = MagicMock()


class TestPreflightValidation:
    """Test preflight validation logic"""

    def test_missing_config_returns_failure(self):
        """Test that missing config parameter returns failure"""
        # Import after mocking
        from linux_performance import preflight
        
        result = preflight.main(config=None)
        
        assert result['result'] == 'FAILURE'
        assert 'Missing config parameter' in result['error']

    def test_valid_sysctl_config_passes(self):
        """Test that valid sysctl configuration passes preflight"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysctl': {
                    'net.ipv4.tcp_rmem': {
                        'constraints': [
                            {'step': 100, 'lower': 4096, 'upper': 131072}
                        ]
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'SUCCESS'
        assert 'Preflight validation passed' in result['data']['message']

    def test_valid_sysfs_config_passes(self):
        """Test that valid sysfs configuration passes preflight"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysfs': {
                    'cpu_governor': {
                        'constraints': [
                            {'values': ['performance', 'powersave', 'ondemand']}
                        ]
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'SUCCESS'

    def test_valid_cpufreq_config_passes(self):
        """Test that valid cpufreq configuration passes preflight"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'cpufreq': {
                    'min_freq_ghz': {
                        'constraints': [
                            {'step': 0.1, 'lower': 1.0, 'upper': 3.0}
                        ]
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'SUCCESS'

    def test_valid_ethtool_config_passes(self):
        """Test that valid ethtool configuration passes preflight"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'ethtool': {
                    'eth0': {
                        'tso': {
                            'constraints': [
                                {'values': ['on', 'off']}
                            ]
                        },
                        'rx_ring': {
                            'constraints': [
                                {'step': 64, 'lower': 256, 'upper': 4096}
                            ]
                        }
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'SUCCESS'

    def test_unsupported_sysctl_parameter_fails(self):
        """Test that unsupported sysctl parameter fails preflight"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysctl': {
                    'unsupported.param': {
                        'constraints': [
                            {'step': 1, 'lower': 0, 'upper': 100}
                        ]
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'FAILURE'
        assert 'unsupported parameter' in result['error'].lower()

    def test_unsupported_sysfs_parameter_fails(self):
        """Test that unsupported sysfs parameter fails preflight"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysfs': {
                    'unknown_sysfs_param': {
                        'constraints': [
                            {'values': ['a', 'b']}
                        ]
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'FAILURE'
        assert 'unsupported parameter' in result['error'].lower()

    def test_unsupported_ethtool_parameter_fails(self):
        """Test that unsupported ethtool parameter fails preflight"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'ethtool': {
                    'eth0': {
                        'unsupported_ethtool_param': {
                            'constraints': [
                                {'values': ['on', 'off']}
                            ]
                        }
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'FAILURE'
        assert 'unsupported ethtool parameter' in result['error'].lower()

    def test_categorical_parameter_with_wrong_constraint_type_fails(self):
        """Test that categorical parameter with range constraints fails"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysfs': {
                    'cpu_governor': {
                        'constraints': [
                            {'step': 1, 'lower': 0, 'upper': 100}  # Wrong: should be 'values'
                        ]
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'FAILURE'
        assert "constraints don't have 'values'" in result['error']

    def test_integer_parameter_with_wrong_constraint_type_fails(self):
        """Test that integer parameter with categorical constraints fails"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysctl': {
                    'net.core.netdev_budget': {
                        'constraints': [
                            {'values': ['100', '200', '300']}  # Wrong: should be range
                        ]
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'FAILURE'
        assert "constraints don't have step/lower/upper" in result['error']

    def test_missing_constraints_fails(self):
        """Test that missing constraints field fails preflight"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysctl': {
                    'net.ipv4.tcp_rmem': {
                        # Missing 'constraints' key
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'FAILURE'
        assert "missing 'constraints'" in result['error']

    def test_constraints_not_list_fails(self):
        """Test that non-list constraints fail preflight"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysctl': {
                    'net.ipv4.tcp_rmem': {
                        'constraints': {'lower': 4096, 'upper': 131072}  # Dict, not list
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'FAILURE'
        assert 'constraints must be a list' in result['error']

    def test_multiple_categories_all_valid(self):
        """Test that config with multiple categories all valid passes"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysctl': {
                    'net.ipv4.tcp_rmem': {
                        'constraints': [{'step': 100, 'lower': 4096, 'upper': 131072}]
                    }
                },
                'sysfs': {
                    'cpu_governor': {
                        'constraints': [{'values': ['performance', 'powersave']}]
                    }
                },
                'cpufreq': {
                    'min_freq_ghz': {
                        'constraints': [{'step': 0.1, 'lower': 1.0, 'upper': 2.0}]
                    }
                },
                'ethtool': {
                    'eth0': {
                        'tso': {
                            'constraints': [{'values': ['on', 'off']}]
                        }
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'SUCCESS'

    def test_multiple_errors_aggregated(self):
        """Test that multiple validation errors are aggregated"""
        from linux_performance import preflight
        
        config = {
            'settings': {
                'sysctl': {
                    'unsupported_param_1': {
                        'constraints': [{'step': 1, 'lower': 0, 'upper': 100}]
                    },
                    'unsupported_param_2': {
                        'constraints': [{'step': 1, 'lower': 0, 'upper': 100}]
                    }
                }
            }
        }
        
        result = preflight.main(config=config)
        
        assert result['result'] == 'FAILURE'
        # Should have multiple errors
        error_lines = result['error'].split('\n')
        assert len(error_lines) > 2  # At least header + 2 errors
