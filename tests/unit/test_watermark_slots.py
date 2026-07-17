
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
"""
Tests for watermark creation and impulse watermark behavior.

Verifies that:
- Inactive detection mode returns None
- Impulse watermark is created with correct params
- Duty cycle and direction are configured properly
- Top params by range are selected for impulse
"""
import pytest
import sys
import types

# Mock the f.breeder.shared.otel_logging module
otel_mock = types.ModuleType('f.breeder.shared.otel_logging')
otel_mock.get_logger = lambda name: type('Logger', (), {'info': lambda *a, **kw: None, 'warning': lambda *a, **kw: None, 'error': lambda *a, **kw: None})()
sys.modules['f.breeder.shared.otel_logging'] = otel_mock

from engine.watermark import create_watermark, Impulse


def _base_config(slot=None):
    """Create a minimal breeder config with optional watermark_slot."""
    config = {
        'interference_detection': {'mode': 'active'},
        'settings': {
            'microgrid': {
                'power_draw': {
                    'constraints': [{'lower': 0.0, 'upper': 1000.0, 'step': 10.0}]
                },
                'storage_dispatch': {
                    'constraints': [{'lower': -500.0, 'upper': 500.0, 'step': 10.0}]
                },
            }
        },
        'breeder': {'type': 'bench_microgrid'},
    }
    if slot is not None:
        config['breeder']['watermark_slot'] = slot
    return config


class TestImpulseCreation:
    """Impulse watermark is created with correct configuration."""

    def test_active_detection_creates_impulse(self):
        wm = create_watermark(_base_config(), _base_config()['settings'])
        assert wm is not None
        assert isinstance(wm, Impulse)

    def test_impulse_selects_top_params_by_range(self):
        """Largest-range params are selected for impulse."""
        wm = create_watermark(_base_config(), _base_config()['settings'])
        assert wm is not None
        assert len(wm._impulse_params) == 2

    def test_impulse_default_duty_cycle(self):
        wm = create_watermark(_base_config(), _base_config()['settings'])
        assert wm is not None
        assert wm.duty_cycle == 0.02

    def test_impulse_custom_duty_cycle(self):
        config = _base_config()
        config['interference_detection']['impulse'] = {'duty_cycle': 0.05}
        wm = create_watermark(config, config['settings'])
        assert wm is not None
        assert wm.duty_cycle == 0.05

    def test_impulse_default_direction(self):
        wm = create_watermark(_base_config(), _base_config()['settings'])
        assert wm is not None
        assert wm.direction == 'random'

    def test_impulse_custom_direction(self):
        config = _base_config()
        config['interference_detection']['impulse'] = {'direction': 'upper'}
        wm = create_watermark(config, config['settings'])
        assert wm is not None
        assert wm.direction == 'upper'

    def test_impulse_has_period(self):
        """Impulse watermark has an internal period for duty cycle."""
        wm = create_watermark(_base_config(), _base_config()['settings'])
        assert wm is not None
        assert hasattr(wm, '_period')
        assert wm._period > 0


class TestInactiveDetection:
    """When detection is inactive, no watermark is created regardless of slot."""

    def test_inactive_returns_none(self):
        config = _base_config(slot=0)
        config['interference_detection']['mode'] = 'inactive'
        wm = create_watermark(config, config['settings'])
        assert wm is None

    def test_inactive_without_slot_returns_none(self):
        config = _base_config()
        config['interference_detection']['mode'] = 'inactive'
        wm = create_watermark(config, config['settings'])
        assert wm is None


class TestNoParamsWithRanges:
    """When no params have ranges, no watermark is created."""

    def test_no_range_params_returns_none(self):
        config = {
            'interference_detection': {'mode': 'active'},
            'settings': {
                'microgrid': {
                    'fixed_param': {'value': 42},
                }
            },
            'breeder': {'type': 'bench_microgrid'},
        }
        wm = create_watermark(config, config['settings'])
        assert wm is None
