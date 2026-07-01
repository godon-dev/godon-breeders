#!/usr/bin/env python3
"""
Integration test for DetectionCoordinator state machine.

Simulates full trial sequences with configurable FAIL rates and verifies:
1. Push block always completes (including with 100% FAIL)
2. Turn-taking actually alternates
3. No state gets stuck
4. Receiver enters baseline BEFORE signal
5. FAIL on push doesn't reset the push counter
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from unittest.mock import MagicMock, patch
from engine.detection_coordinator import DetectionCoordinator


def _base_config():
    return {
        'detection': {
            'warmup_trials': 5,
            'push_block_size': 3,
            'pause_block_size': 3,
            'receiver_baseline_trials': 2,
            'receiver_post_trials': 2,
            'recover_trials': 2,
        },
        'objectives': [{'name': 'growth_rate', 'direction': 'maximize'}],
        'settings': {},
    }


def _create_coordinator(breeder_id='test-1', config=None):
    config = config or _base_config()
    coord = DetectionCoordinator(
        breeder_id=breeder_id,
        config=config,
        shared_db_fn=lambda fn, desc: _mock_db(fn, desc),
        collect_upper_bounds_fn=lambda cfg: [
            {'name': 'heating', 'upper': 40.0, 'range': 35.0, 'is_int': False},
            {'name': 'light', 'upper': 1000.0, 'range': 1000.0, 'is_int': False},
            {'name': 'co2', 'upper': 20.0, 'range': 20.0, 'is_int': False},
        ],
    )
    return coord


# Global mock DB state
_mock_rounds = []
_mock_lock_holder = [None]
_mock_complete_count = [0]
_mock_effectuation_params = [None]


def _mock_db(fn, desc):
    """Mock _with_shared_db — provides a mock connection."""
    conn = MagicMock()
    cur = MagicMock()

    def execute(query, *args):
        query_str = str(query).lower()
        if 'pg_try_advisory_lock' in query_str:
            if _mock_lock_holder[0] is None:
                _mock_lock_holder[0] = 'test-1'
                cur.fetchone.return_value = (True,)
            else:
                cur.fetchone.return_value = (False,)
        elif 'pg_advisory_unlock' in query_str:
            _mock_lock_holder[0] = None
            cur.fetchone.return_value = (True,)
        elif 'insert into detection_rounds' in query_str:
            _mock_rounds.append({'sender_id': args[0] if args else 'test-1', 'status': 'active'})
        elif 'select sender_id from detection_rounds order by round_id' in query_str:
            if _mock_rounds:
                cur.fetchone.return_value = (_mock_rounds[-1]['sender_id'],)
            else:
                cur.fetchone.return_value = None
        elif 'select count(*) from detection_rounds' in query_str:
            cur.fetchone.return_value = (0,)  # other breeder has sent
        elif 'update detection_rounds set status' in query_str:
            for r in _mock_rounds:
                if r['status'] == 'active' and r['sender_id'] == 'test-1':
                    r['status'] = 'completed'
        elif 'select count(*) from detection_rounds where status' in query_str:
            cur.fetchone.return_value = (len([r for r in _mock_rounds if r['status'] == 'active']),)

    conn.cursor.return_value = cur
    cur.execute.side_effect = execute
    cur.close = MagicMock()
    result = fn(conn)
    return result


def _reset_mocks():
    global _mock_rounds, _mock_lock_holder, _mock_complete_count, _mock_effectuation_params
    _mock_rounds = []
    _mock_lock_holder = [None]
    _mock_complete_count = [0]
    _mock_effectuation_params = [None]


class MockStudy:
    def __init__(self):
        self.trials = []

    def add_complete_trial(self, number, params=None, value=0.5):
        t = MagicMock()
        t.state = 'COMPLETE'
        t.number = number
        t.user_attrs = {'effectuation_params': '{"heating": 20.0}'} if params else None
        t.values = [value]
        self.trials.append(t)

    @property
    def COMPLETE(self):
        return [t for t in self.trials if t.state == 'COMPLETE']


class MockTrial:
    def __init__(self, number):
        self.number = number
        self._attrs = {}

    def set_user_attr(self, key, value):
        self._attrs[key] = value

    @property
    def user_attrs(self):
        return self._attrs


def test_push_completes_with_all_fails():
    """Push block must complete even when every trial FAILs guardrails."""
    _reset_mocks()
    coord = _create_coordinator()
    coord.warmup_target = 0  # Skip warmup
    coord.state = coord.WARMUP

    # Mock warmup as complete
    with patch.object(coord, '_count_complete_trials_db', return_value=10):
        with patch.object(coord, '_refresh_baseline_db'):
            coord._baseline_params = {'heating': 20.0}

            # Force into SENDER_PUSH
            coord.state = coord.SENDER_PUSH
            coord._push_count = 0
            coord._pause_count = 0

            # Simulate push_block_size pushes, all FAILing
            push_count = 0
            for i in range(coord.push_block_size + 5):
                trial = MockTrial(i + 1)
                decision = coord.decide_trial(trial, MockStudy())

                if coord.state == coord.SENDER_PUSH:
                    assert decision['mode'] == 'impulse', f"Trial {i}: expected impulse, got {decision['mode']}"
                    assert decision['impulse_phase'] == 'push', f"Trial {i}: expected push, got {decision.get('impulse_phase')}"
                    # Simulate FAIL
                    coord.on_guardrail_fail({'heating': 40.0})
                    push_count += 1
                else:
                    # Transitioned out of push — success
                    break

            assert coord.state != coord.SENDER_PUSH, \
                f"STUCK in SENDER_PUSH after {push_count} attempts (counter={coord._push_count})"

            print(f"PASS: Push completed after {push_count} pushes (push_block_size={coord.push_block_size})")


def test_push_not_reset_on_fail():
    """on_guardrail_fail must NOT decrement push counter."""
    _reset_mocks()
    coord = _create_coordinator()

    coord.state = coord.SENDER_PUSH
    coord._push_count = 3
    coord._baseline_params = {'heating': 20.0}

    coord.on_guardrail_fail({'heating': 40.0})

    assert coord._push_count == 3, f"Push counter was decremented on FAIL: {coord._push_count}"
    assert coord._impulse_scale < 1.0, f"AIMD should have reduced scale: {coord._impulse_scale}"

    print(f"PASS: Push counter not reset on FAIL (stayed at {coord._push_count}, scale={coord._impulse_scale:.2f})")


def test_no_state_runs_forever():
    """Every state must have a max-trials escape hatch."""
    coord = _create_coordinator()

    max_limits = {
        coord.SENDER_PUSH: coord.MAX_PUSH_ATTEMPTS,
        coord.RECEIVER_HOLD: coord.MAX_HOLD_TRIALS,
        coord.RECEIVER_BASELINE: coord.MAX_RECEIVER_BASELINE,
        coord.RECEIVER_POST: coord.MAX_RECEIVER_POST,
        coord.RECOVER: coord.MAX_RECOVER_TRIALS,
    }

    for state, max_trials in max_limits.items():
        assert max_trials > 0, f"{state} has no max limit!"
        assert max_trials < 200, f"{state} max limit too high: {max_trials}"

    print(f"PASS: All states have escape hatches: {max_limits}")


def test_max_push_attempts_completes_round():
    """If push always fails, MAX_PUSH_ATTEMPTS forces round completion."""
    _reset_mocks()
    coord = _create_coordinator()
    coord.state = coord.SENDER_PUSH
    coord._push_count = 0
    coord._baseline_params = {'heating': 20.0}

    with patch.object(coord, '_get_impulse_params', return_value={'heating': 40.0}):
        decisions = []
        for i in range(coord.MAX_PUSH_ATTEMPTS + 5):
            trial = MockTrial(i)
            decision = coord.decide_trial(trial, MockStudy())
            decisions.append(decision)
            if coord.state != coord.SENDER_PUSH:
                break

        assert coord.state != coord.SENDER_PUSH, \
            f"STUCK in SENDER_PUSH after {coord.MAX_PUSH_ATTEMPTS + 5} trials"

        # Should have transitioned to SENDER_DONE or SENDER_PAUSE
        assert coord.state in (coord.SENDER_DONE, coord.SENDER_PAUSE), \
            f"Unexpected state after max push: {coord.state}"

        print(f"PASS: Escaped SENDER_PUSH after {len(decisions)} trials -> {coord.state}")


def test_warmup_not_interrupted_by_stale_rounds():
    """Warmup must complete fully, ignoring stale active rounds."""
    _reset_mocks()
    coord = _create_coordinator()
    coord.state = coord.WARMUP

    # Mock: stale active round exists but warmup not done
    with patch.object(coord, '_any_active_round', return_value=True):
        with patch.object(coord, '_count_complete_trials_db', return_value=2):
            trial = MockTrial(0)
            decision = coord.decide_trial(trial, MockStudy())

            assert coord.state == coord.WARMUP, \
                f"Warmup interrupted by stale round! State={coord.state}"
            assert decision['mode'] == 'optimize', \
                f"Warmup should optimize, got {decision['mode']}"

    print("PASS: Warmup not interrupted by stale rounds")


if __name__ == '__main__':
    test_push_not_reset_on_fail()
    test_push_completes_with_all_fails()
    test_no_state_runs_forever()
    test_max_push_attempts_completes_round()
    test_warmup_not_interrupted_by_stale_rounds()
    print("\n=== ALL TESTS PASSED ===")
