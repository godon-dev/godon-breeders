
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
#!/usr/bin/env python3
"""
Integration tests for DetectionCoordinator state machine (count-based budget).

Tests the state machine behavior end-to-end through decide_trial(),
verifying:
1. Push block always completes (fixed count, even with guardrail FAILs)
2. on_guardrail_fail does NOT reset push counter or scale
3. Every state has a bounded escape hatch
4. Full sender round: PUSH -> PAUSE -> DONE -> COOLDOWN -> OPTIMIZE
5. OPTIMIZE stays put when not enough trials or breeders
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from unittest.mock import MagicMock, patch
from engine.detection_coordinator import DetectionCoordinator


def _base_config(**overrides):
    config = {
        'interference_detection': {
            'min_optimize_trials': 5,
            'hold_calib_trials': 3,
            'push_block_size': 3,
            'pause_block_size': 3,
            'cooldown_trials': 2,
            'hold_params': {'heating': 20.0, 'light': 500.0, 'co2': 10.0},
        },
        'objectives': [{'name': 'growth_rate', 'direction': 'maximize'}],
        'settings': {},
    }
    config.update(overrides)
    return config


def _create_coordinator(breeder_id='test-1', config=None):
    config = config or _base_config()
    coord = DetectionCoordinator(
        breeder_id=breeder_id,
        config=config,
        shared_db_fn=lambda fn, desc: _noop_db(fn, desc),
        collect_upper_bounds_fn=lambda cfg: [
            {'name': 'heating', 'upper': 40.0, 'lower': 0.0, 'range': 40.0, 'is_int': False},
            {'name': 'light', 'upper': 1000.0, 'lower': 0.0, 'range': 1000.0, 'is_int': False},
            {'name': 'co2', 'upper': 20.0, 'lower': 0.0, 'range': 20.0, 'is_int': False},
        ],
    )
    return coord


def _noop_db(fn, desc):
    """Minimal DB mock — most DB methods are patched per-test."""
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (0,)
    cur.rowcount = 0
    conn.cursor.return_value = cur
    try:
        return fn(conn)
    except Exception:
        return None


def _patch_sender_db(coord):
    """Patch all DB-dependent methods so the sender state machine runs cleanly.

    Returns a list of patchers to stop later.
    """
    patchers = [
        patch.object(coord, '_ensure_tables'),
        patch.object(coord, '_cleanup_stale_state'),
        patch.object(coord, '_heartbeat', return_value=True),
        patch.object(coord, '_set_lease_phase'),
        patch.object(coord, '_decrement_budget'),
        patch.object(coord, '_release_lease'),
        patch.object(coord, '_signal_ready'),
        patch.object(coord, '_clear_ready'),
        patch.object(coord, '_check_all_ready', return_value=True),
        patch.object(coord, '_count_active_breeders', return_value=2),
        patch.object(coord, '_count_complete_trials_db', return_value=100),
        patch.object(coord, '_has_active_sender', return_value=False),
    ]
    for p in patchers:
        p.start()
    return patchers


class MockTrial:
    def __init__(self, number):
        self.number = number
        self._attrs = {}

    def set_user_attr(self, key, value):
        self._attrs[key] = value

    @property
    def user_attrs(self):
        return self._attrs


class TestPushCompletes:
    """Push block must complete even when every trial FAILs guardrails."""

    def test_push_completes_with_all_fails(self):
        coord = _create_coordinator()
        coord._initialized = True
        coord.state = DetectionCoordinator.PUSH
        coord._push_count = 0
        coord._locked_scale = 1.0

        patchers = _patch_sender_db(coord)
        try:
            for i in range(coord.push_block_size + 5):
                trial = MockTrial(i + 1)
                decision = coord.decide_trial(trial)

                if coord.state == DetectionCoordinator.PUSH:
                    assert decision['mode'] == 'impulse', \
                        f"Trial {i}: expected impulse, got {decision['mode']}"
                    assert decision['impulse_phase'] == 'push', \
                        f"Trial {i}: expected push, got {decision.get('impulse_phase')}"
                    # Simulate FAIL
                    coord.on_guardrail_fail({'heating': 40.0})
                else:
                    break

            assert coord.state != DetectionCoordinator.PUSH, \
                f"STUCK in PUSH after {coord.push_block_size + 5} trials"

            print(f"PASS: Push completed after {coord._push_count} pushes "
                  f"(push_block_size={coord.push_block_size})")
        finally:
            for p in patchers:
                p.stop()


class TestGuardrailFail:
    """on_guardrail_fail must NOT decrement push counter or change locked scale."""

    def test_push_not_reset_on_fail(self):
        coord = _create_coordinator()
        coord.state = DetectionCoordinator.PUSH
        coord._push_count = 3
        coord._locked_scale = 1.0

        coord.on_guardrail_fail({'heating': 40.0})

        assert coord._push_count == 3, \
            f"Push counter was decremented on FAIL: {coord._push_count}"
        assert coord._locked_scale == 1.0, \
            f"Locked scale should not change during PUSH: {coord._locked_scale}"

        print(f"PASS: Push counter not reset on FAIL "
              f"(stayed at {coord._push_count}, scale={coord._locked_scale:.2f})")


class TestEscapeHatches:
    """Every state must have a bounded escape hatch (max trials or floor)."""

    def test_all_states_have_escape_hatch(self):
        coord = _create_coordinator()

        # Fixed-count phases: always complete in exactly N trials
        assert coord.push_block_size > 0, "push_block_size must be positive"
        assert coord.push_block_size < 200, \
            f"push_block_size too high: {coord.push_block_size}"
        assert coord.pause_block_size > 0, "pause_block_size must be positive"
        assert coord.pause_block_size < 200, \
            f"pause_block_size too high: {coord.pause_block_size}"
        assert coord.cooldown_trials > 0, "cooldown_trials must be positive"

        # Bounded search phases
        assert coord.MAX_HOLD_TRIALS > 0, "MAX_HOLD_TRIALS missing"
        assert coord.MAX_HOLD_TRIALS < 200, \
            f"MAX_HOLD_TRIALS too high: {coord.MAX_HOLD_TRIALS}"
        assert coord.MAX_HOLD_CALIB_SEARCH > 0, "MAX_HOLD_CALIB_SEARCH missing"

        # Scale floor: IMPULSE_CALIB aborts when scale drops below this
        assert coord.MIN_IMPULSE_SCALE > 0, "MIN_IMPULSE_SCALE missing"

        # Partner wait timeout
        assert coord.MAX_CALIB_WAIT > 0, "MAX_CALIB_WAIT missing"

        print("PASS: All states have escape hatches")


class TestFullSenderRound:
    """Full sender round: PUSH -> PAUSE -> DONE -> COOLDOWN -> OPTIMIZE."""

    def test_full_round_completes(self):
        coord = _create_coordinator()
        coord._initialized = True
        coord.state = DetectionCoordinator.PUSH
        coord._push_count = 0
        coord._locked_scale = 1.0

        patchers = _patch_sender_db(coord)
        try:
            states_seen = []
            total_trials = 0

            for i in range(coord.push_block_size + coord.pause_block_size + 10):
                trial = MockTrial(i + 1)
                coord.decide_trial(trial)
                total_trials += 1

                if coord.state not in states_seen:
                    states_seen.append(coord.state)

                if coord.state == DetectionCoordinator.OPTIMIZE:
                    break

            assert DetectionCoordinator.PAUSE in states_seen, \
                f"Never entered PAUSE. States seen: {states_seen}"
            assert DetectionCoordinator.COOLDOWN in states_seen, \
                f"Never entered COOLDOWN. States seen: {states_seen}"
            assert coord.state == DetectionCoordinator.OPTIMIZE, \
                f"Did not return to OPTIMIZE. Final state: {coord.state}"

            print(f"PASS: Full round completed in {total_trials} trials. "
                  f"States: {' -> '.join(states_seen)}")
        finally:
            for p in patchers:
                p.stop()


class TestOptimizeGating:
    """OPTIMIZE must stay in OPTIMIZE when conditions aren't met."""

    def test_stays_optimize_when_not_enough_trials(self):
        coord = _create_coordinator()
        coord._initialized = True
        coord.state = DetectionCoordinator.OPTIMIZE

        with patch.object(coord, '_count_complete_trials_db', return_value=2), \
             patch.object(coord, '_count_active_breeders', return_value=5), \
             patch.object(coord, '_try_acquire_lease') as mock_acquire, \
             patch.object(coord, '_has_active_sender') as mock_has_sender:
            trial = MockTrial(0)
            decision = coord.decide_trial(trial)

            assert coord.state == DetectionCoordinator.OPTIMIZE, \
                f"Left OPTIMIZE with too few trials. State: {coord.state}"
            assert decision['mode'] == 'optimize', \
                f"Expected optimize mode, got {decision['mode']}"
            mock_acquire.assert_not_called()

        print("PASS: Stays in OPTIMIZE when not enough trials")

    def test_stays_optimize_when_alone(self):
        coord = _create_coordinator()
        coord._initialized = True
        coord.state = DetectionCoordinator.OPTIMIZE

        with patch.object(coord, '_count_complete_trials_db', return_value=100), \
             patch.object(coord, '_count_active_breeders', return_value=1), \
             patch.object(coord, '_try_acquire_lease') as mock_acquire, \
             patch.object(coord, '_has_active_sender') as mock_has_sender:
            trial = MockTrial(0)
            decision = coord.decide_trial(trial)

            assert coord.state == DetectionCoordinator.OPTIMIZE, \
                f"Left OPTIMIZE when alone. State: {coord.state}"
            assert decision['mode'] == 'optimize'
            mock_acquire.assert_not_called()

        print("PASS: Stays in OPTIMIZE when only 1 breeder active")


if __name__ == '__main__':
    TestGuardrailFail().test_push_not_reset_on_fail()
    TestPushCompletes().test_push_completes_with_all_fails()
    TestEscapeHatches().test_all_states_have_escape_hatch()
    TestFullSenderRound().test_full_round_completes()
    TestOptimizeGating().test_stays_optimize_when_not_enough_trials()
    TestOptimizeGating().test_stays_optimize_when_alone()
    print("\n=== ALL TESTS PASSED ===")
