"""
Detection Coordinator — Clean state machine for coordinated interference detection.

Replaces scattered detection logic in breeder_worker.py with an explicit
state machine that manages warmup, sender/receiver roles, impulse pulsing,
baseline refresh, and turn-taking through the shared detection_rounds table.

States:
    WARMUP         — Free optimization, no detection
    SENDER_PING    — Apply extreme params (the ping)
    SENDER_LISTEN  — Apply baseline params (the listen)
    SENDER_DONE    — Round complete, mark in DB
    RECEIVER_HOLD  — Apply baseline params, measure coupling
    RECOVER        — Brief optimization to refresh baseline for drifted greenhouse

Cycle:
    WARMUP → (become sender or receiver)
    SENDER: PING → LISTEN → PING → LISTEN → ... → DONE → RECOVER → (swap)
    RECEIVER: HOLD → HOLD → ... → DONE → RECOVER → (swap)
"""

import json
import logging
from typing import Dict, Any, Optional, Callable
from optuna.trial import TrialState

logger = logging.getLogger(__name__)


class DetectionCoordinator:
    """Manages the coordinated detection lifecycle for a single breeder.

    The coordinator is called once per trial via decide_trial() and returns
    what mode the breeder should operate in for that trial.

    Coordination between breeders happens through the shared detection_rounds
    table in YugaByte. Each breeder checks this table to determine its role.
    """

    # States
    WARMUP = "warmup"
    SENDER_PING = "sender_ping"
    SENDER_LISTEN = "sender_listen"
    SENDER_DONE = "sender_done"
    RECEIVER_HOLD = "receiver_hold"
    RECOVER = "recover"

    def __init__(
        self,
        breeder_id: str,
        config: Dict[str, Any],
        shared_db_fn: Callable,
        collect_upper_bounds_fn: Callable,
    ):
        self.breeder_id = breeder_id
        self.config = config
        self._db = shared_db_fn  # _with_shared_db callback
        self._collect_upper_bounds = collect_upper_bounds_fn

        # Config
        det_cfg = config.get('detection', {})
        self.warmup_target = det_cfg.get('warmup_trials', 15)
        self.impulses_per_round = det_cfg.get('impulses_per_round', 5)
        self.recover_trials = det_cfg.get('recover_trials', 3)

        # State
        self.state = self.WARMUP
        self._initialized = False

        # Counters
        self._ping_count = 0       # Pings sent in current round
        self._recover_count = 0    # Optimize trials in current RECOVER phase

        # Params
        self._baseline_params = None    # Current safe operating point (best trial)
        self._impulse_params = None     # Extreme params (cached after first generation)
        self._impulse_scale = 1.0       # AIMD scale factor
        self._impulse_base_params = None  # Original baseline for AIMD re-scaling

    def _cleanup_stale_rounds(self):
        """Complete all active rounds to reset coordination at startup."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE detection_rounds SET status = 'completed', completed_at = NOW() "
                "WHERE status = 'active'"
            )
            cur.close()
        try:
            self._db(op, "cleanup_stale_rounds")
            logger.info("Cleaned up stale detection rounds")
        except Exception as e:
            logger.warning(f"Failed to cleanup stale rounds: {e}")

    def _try_start_round(self) -> bool:
        """Try to start a new detection round as sender.
        Only starts if no other active rounds exist AND this breeder
        wasn't the most recent sender (fair turn-taking)."""
        def op(conn):
            cur = conn.cursor()
            # No active rounds allowed
            cur.execute("SELECT count(*) FROM detection_rounds WHERE status = 'active'")
            if cur.fetchone()[0] > 0:
                cur.close()
                return False
            # Don't start if we were the most recent sender
            cur.execute(
                "SELECT sender_id FROM detection_rounds "
                "ORDER BY round_id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row and row[0] == self.breeder_id:
                cur.close()
                return False
            # Start our round
            cur.execute(
                "INSERT INTO detection_rounds (sender_id) VALUES (%s)",
                (self.breeder_id,)
            )
            cur.close()
            return True
        try:
            return self._db(op, "try_start_round")
        except Exception as e:
            logger.warning(f"Failed to start round: {e}")
            return False

    def _has_own_active_round(self) -> bool:
        """Check if this breeder has an active round (is the current sender)."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT count(*) FROM detection_rounds "
                "WHERE status = 'active' AND sender_id = %s",
                (self.breeder_id,)
            )
            result = cur.fetchone()[0] > 0
            cur.close()
            return result
        try:
            return self._db(op, "has_own_active_round")
        except Exception:
            return False

    def _any_active_round(self) -> bool:
        """Check if any breeder has an active round."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT count(*) FROM detection_rounds WHERE status = 'active'"
            )
            result = cur.fetchone()[0] > 0
            cur.close()
            return result
        try:
            return self._db(op, "any_active_round")
        except Exception:
            return False

    def _complete_my_round(self):
        """Mark this breeder's active round as completed."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE detection_rounds SET status = 'completed', completed_at = NOW() "
                "WHERE status = 'active' AND sender_id = %s",
                (self.breeder_id,)
            )
            cur.close()
        try:
            self._db(op, "complete_my_round")
            logger.info("Detection round completed")
        except Exception as e:
            logger.warning(f"Failed to complete round: {e}")

    def _refresh_baseline(self, study):
        """Find the best complete trial and use its params as the current baseline.
        Called after warmup and after each RECOVER phase."""
        if not study or not study.trials:
            return

        objectives = self.config.get('objectives', [])
        minimize = objectives and objectives[0].get('direction', '').lower() == 'minimize'

        best_trial_params = None
        best_value = None

        for trial in study.trials:
            if trial.state != TrialState.COMPLETE:
                continue
            stashed = trial.user_attrs.get('effectuation_params')
            if not stashed:
                continue
            values = trial.values if hasattr(trial, 'values') else None
            if not values:
                continue
            v = values[0]
            if best_value is None:
                best_value = v
                best_trial_params = stashed
            elif minimize and v < best_value:
                best_value = v
                best_trial_params = stashed
            elif not minimize and v > best_value:
                best_value = v
                best_trial_params = stashed

        if best_trial_params:
            self._baseline_params = json.loads(best_trial_params) \
                if isinstance(best_trial_params, str) else dict(best_trial_params)
            # Invalidate cached impulse params so they regenerate from new baseline
            self._impulse_params = None
            self._impulse_base_params = None
            logger.info(f"Refreshed baseline params (best value: {best_value:.4f})")

    def _get_impulse_params(self) -> Optional[Dict[str, Any]]:
        """Generate extreme params from current baseline.
        Caches after first generation. AIMD-scaled on guardrail FAIL."""
        if self._impulse_params is not None:
            return self._impulse_params

        if self._baseline_params is None:
            return None

        upper_bounds = self._collect_upper_bounds(self.config.get('settings', {}))
        upper_bounds.sort(key=lambda x: x.get('range', 0), reverse=True)

        params = dict(self._baseline_params)
        for ub in upper_bounds[:3]:
            name = ub['name']
            value = ub['upper'] * self._impulse_scale
            if ub.get('is_int'):
                value = int(value)
            if name in params:
                if isinstance(params[name], list):
                    params[name] = [value] * len(params[name])
                else:
                    params[name] = value

        self._impulse_params = params
        self._impulse_base_params = dict(self._baseline_params)
        logger.info(f"Generated impulse params at scale {self._impulse_scale:.2f}")
        return params

    def _aimd_backoff(self):
        """Scale down impulse on guardrail FAIL."""
        self._impulse_scale *= 0.5
        if self._impulse_base_params is not None:
            upper_bounds = self._collect_upper_bounds(self.config.get('settings', {}))
            upper_bounds.sort(key=lambda x: x.get('range', 0), reverse=True)
            params = dict(self._impulse_base_params)
            for ub in upper_bounds[:3]:
                name = ub['name']
                value = ub['upper'] * self._impulse_scale
                if ub.get('is_int'):
                    value = int(value)
                if name in params:
                    if isinstance(params[name], list):
                        params[name] = [value] * len(params[name])
                    else:
                        params[name] = value
            self._impulse_params = params
            logger.warning(f"Impulse AIMD backoff to scale {self._impulse_scale:.2f}")
        else:
            self._impulse_params = None

    def decide_trial(self, trial, study) -> Dict[str, Any]:
        """Main entry point. Called once per trial.
        Returns dict with:
            'mode': 'optimize' | 'hold' | 'impulse'
            'params': dict or None (if None, breeder samples normally)
            'impulse_phase': 'ping' | 'listen' | None
        """
        # First-time initialization
        if not self._initialized:
            self._cleanup_stale_rounds()
            self._initialized = True

        if self.state == self.WARMUP:
            complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE) \
                if study and study.trials else 0
            if complete >= self.warmup_target:
                self._refresh_baseline(study)
                if self._try_start_round():
                    self.state = self.SENDER_PING
                    self._ping_count = 0
                    logger.info("Warmup complete — becoming SENDER")
                elif self._any_active_round():
                    self.state = self.RECEIVER_HOLD
                    logger.info("Warmup complete — becoming RECEIVER (other sender active)")
                else:
                    # No active rounds, couldn't start (was recent sender?) — stay warmup
                    # This shouldn't happen after cleanup, but handle gracefully
                    pass
            return {'mode': 'optimize', 'params': None}

        if self.state == self.SENDER_PING:
            params = self._get_impulse_params()
            if not params:
                logger.warning("No impulse params — falling back to optimize")
                return {'mode': 'optimize', 'params': None}
            self._ping_count += 1
            self.state = self.SENDER_LISTEN
            return {'mode': 'impulse', 'params': params, 'impulse_phase': 'ping'}

        if self.state == self.SENDER_LISTEN:
            params = dict(self._baseline_params) if self._baseline_params else None
            if self._ping_count >= self.impulses_per_round:
                self.state = self.SENDER_DONE
            else:
                self.state = self.SENDER_PING
            return {'mode': 'impulse', 'params': params, 'impulse_phase': 'listen'}

        if self.state == self.SENDER_DONE:
            self._complete_my_round()
            self._ping_count = 0
            self._impulse_params = None  # Force regeneration next round
            self._recover_count = 0
            self.state = self.RECOVER
            logger.info("Sender round complete — entering RECOVER")
            return {'mode': 'optimize', 'params': None}

        if self.state == self.RECEIVER_HOLD:
            # Check if sender still active
            if not self._any_active_round():
                # Sender finished — recover then try to become sender
                self._recover_count = 0
                self.state = self.RECOVER
                logger.info("Sender finished — entering RECOVER")
            else:
                params = dict(self._baseline_params) if self._baseline_params else None
                return {'mode': 'hold', 'params': params}
            # Fall through to RECOVER on next trial
            return {'mode': 'optimize', 'params': None}

        if self.state == self.RECOVER:
            self._recover_count += 1
            if self._recover_count >= self.recover_trials:
                self._refresh_baseline(study)
                if self._try_start_round():
                    self.state = self.SENDER_PING
                    self._ping_count = 0
                    logger.info("Recover complete — becoming SENDER")
                elif self._any_active_round():
                    self.state = self.RECEIVER_HOLD
                    logger.info("Recover complete — becoming RECEIVER")
                else:
                    # Nobody is sender and we can't start — retry next trial
                    self._recover_count = 0
            return {'mode': 'optimize', 'params': None}

        # Unknown state — safe fallback
        logger.warning(f"Unknown detection state: {self.state} — resetting to WARMUP")
        self.state = self.WARMUP
        return {'mode': 'optimize', 'params': None}

    def on_guardrail_fail(self, params: Dict[str, Any]):
        """Called when a trial fails guardrails. Scales down impulse if in ping phase."""
        if self.state == self.SENDER_PING:
            # Don't count this ping — it failed
            self._ping_count = max(0, self._ping_count - 1)
            self._aimd_backoff()
            # Stay in SENDER_PING state — will retry next trial with lower scale
            self.state = self.SENDER_PING
            logger.info(f"Guardrail FAIL during ping — AIMD backoff, retrying")

    def get_state(self) -> str:
        """Return current state for debugging/observability."""
        return self.state
