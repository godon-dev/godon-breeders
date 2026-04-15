import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from strains.bench_greenhouse import preflight


class TestPreflightValidation:
    def test_missing_config_returns_failure(self):
        result = preflight.main(config=None)
        assert result['result'] == 'FAILURE'
        assert 'Missing config parameter' in result['error']

    def test_valid_greenhouse_config_passes(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 2,
                    'heating_setpoints': {
                        'constraints': [{'step': 0.5, 'lower': 5.0, 'upper': 40.0}]
                    },
                    'shading': {
                        'constraints': [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}]
                    }
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'SUCCESS'

    def test_valid_all_params(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 4,
                    'heating_setpoints': {
                        'constraints': [{'step': 0.5, 'lower': 5.0, 'upper': 40.0}]
                    },
                    'vent_openings': {
                        'constraints': [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}]
                    },
                    'shading': {
                        'constraints': [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}]
                    },
                    'co2_injection': {
                        'constraints': [{'step': 0.5, 'lower': 0.0, 'upper': 20.0}]
                    },
                    'light_intensity': {
                        'constraints': [{'step': 10.0, 'lower': 0.0, 'upper': 1000.0}]
                    },
                    'irrigation': {
                        'constraints': [{'step': 0.1, 'lower': 0.0, 'upper': 3.0}]
                    },
                    'sim_steps': {
                        'constraints': [{'step': 10, 'lower': 10, 'upper': 200}]
                    }
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'SUCCESS'

    def test_unsupported_parameter_fails_strict(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 2,
                    'nonexistent_param': {
                        'constraints': [{'step': 1, 'lower': 0, 'upper': 100}]
                    }
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'FAILURE'
        assert 'unsupported parameter' in result['error'].lower()

    def test_unsupported_parameter_warns_non_strict(self):
        config = {
            'meta': {'strict_validation': False},
            'settings': {
                'greenhouse': {
                    'zones': 2,
                    'nonexistent_param': {
                        'constraints': [{'step': 1, 'lower': 0, 'upper': 100}]
                    }
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'SUCCESS'
        assert 'warnings' in result['data']

    def test_missing_constraints_fails(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 2,
                    'shading': {}
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'FAILURE'
        assert "missing 'constraints'" in result['error']

    def test_invalid_zones_value_fails(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 0,
                    'shading': {
                        'constraints': [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}]
                    }
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'FAILURE'
        assert 'zones' in result['error'].lower()

    def test_categorical_param_wrong_constraints_fails(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 2,
                    'shading': {
                        'constraints': [{'step': 0.1, 'lower': 0.0, 'upper': 1.0}]
                    }
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'FAILURE'
        assert "step/lower/upper" in result['error']

    def test_float_param_wrong_constraints_fails(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 2,
                    'heating_setpoints': {
                        'constraints': [{'values': ['a', 'b']}]
                    }
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'FAILURE'
        assert "'values'" in result['error']

    def test_non_dict_settings_fails(self):
        config = {
            'settings': {
                'greenhouse': 'not_a_dict'
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'FAILURE'
        assert 'must be a dict' in result['error'].lower()

    def test_non_dict_param_config_fails(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 2,
                    'shading': 'not_a_dict'
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'FAILURE'

    def test_constraints_dict_with_values_succeeds(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 2,
                    'heating_setpoints': {
                        'constraints': {'step': 0.5, 'lower': 5.0, 'upper': 40.0}
                    }
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'FAILURE'

    def test_multiple_errors_aggregated(self):
        config = {
            'settings': {
                'greenhouse': {
                    'zones': 2,
                    'bad_param_1': {
                        'constraints': [{'step': 1, 'lower': 0, 'upper': 100}]
                    },
                    'bad_param_2': {
                        'constraints': [{'step': 1, 'lower': 0, 'upper': 100}]
                    }
                }
            }
        }

        result = preflight.main(config=config)
        assert result['result'] == 'FAILURE'
        error_lines = result['error'].split('\n')
        assert len(error_lines) > 2
