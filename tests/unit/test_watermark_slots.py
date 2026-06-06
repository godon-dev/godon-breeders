"""
Tests for collision-free watermark slot assignment.

Verifies that:
- Explicit watermark_slot from config is used correctly
- Missing watermark_slot raises ValueError
- Invalid slot values raise ValueError
- All 6 slots map to correct prime pairs
"""
import pytest
import sys
import types

# Mock the f.breeder.shared.otel_logging module
otel_mock = types.ModuleType('f.breeder.shared.otel_logging')
otel_mock.get_logger = lambda name: type('Logger', (), {'info': lambda *a, **kw: None, 'warning': lambda *a, **kw: None, 'error': lambda *a, **kw: None})()
sys.modules['f.breeder.shared.otel_logging'] = otel_mock

from engine.watermark import create_watermark

PRIME_POOL = [17, 23, 29, 37, 41, 43, 47, 53, 59, 61, 67, 71]


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


class TestExplicitSlotAssignment:
    """Controller-assigned slots are used without fallback."""

    def test_slot_0_maps_to_first_prime_pair(self):
        wm = create_watermark(_base_config(slot=0), _base_config()['settings'])
        assert wm is not None
        assert wm.periods == [17, 23]

    def test_slot_5_maps_to_last_prime_pair(self):
        wm = create_watermark(_base_config(slot=5), _base_config()['settings'])
        assert wm is not None
        assert wm.periods == [67, 71]

    @pytest.mark.parametrize("slot,expected", [
        (0, [17, 23]),
        (1, [29, 37]),
        (2, [41, 43]),
        (3, [47, 53]),
        (4, [59, 61]),
        (5, [67, 71]),
    ])
    def test_all_slots_map_correctly(self, slot, expected):
        wm = create_watermark(_base_config(slot=slot), _base_config()['settings'])
        assert wm.periods == expected

    def test_no_two_slots_share_primes(self):
        all_periods = set()
        for slot in range(6):
            wm = create_watermark(_base_config(slot=slot), _base_config()['settings'])
            periods = tuple(wm.periods)
            assert periods not in all_periods, f"Slot {slot} shares periods with another slot"
            all_periods.add(periods)


class TestMissingSlotRaises:
    """Missing watermark_slot is a fatal error."""

    def test_missing_slot_raises_value_error(self):
        with pytest.raises(ValueError, match="No watermark_slot assigned"):
            create_watermark(_base_config(), _base_config()['settings'])

    def test_no_fallback_to_hash(self):
        """Even with a UUID, missing slot still raises."""
        config = _base_config()
        config['breeder']['uuid'] = 'some-uuid-here'
        with pytest.raises(ValueError, match="No watermark_slot assigned"):
            create_watermark(config, _base_config()['settings'])


class TestInvalidSlotRaises:
    """Out-of-range slot values are rejected."""

    def test_negative_slot_raises(self):
        with pytest.raises(ValueError, match="Invalid watermark_slot"):
            create_watermark(_base_config(slot=-1), _base_config()['settings'])

    def test_slot_too_large_raises(self):
        with pytest.raises(ValueError, match="Invalid watermark_slot"):
            create_watermark(_base_config(slot=6), _base_config()['settings'])

    def test_slot_100_raises(self):
        with pytest.raises(ValueError, match="Invalid watermark_slot"):
            create_watermark(_base_config(slot=100), _base_config()['settings'])


class TestInactiveDetection:
    """When detection is inactive, no watermark is created regardless of slot."""

    def test_inactive_returns_none(self):
        config = _base_config(slot=0)
        config['interference_detection']['mode'] = 'inactive'
        wm = create_watermark(config, config['settings'])
        assert wm is None
