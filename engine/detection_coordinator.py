
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
Detection Coordinator — Phase-based state machine for coordinated interference detection.

Three primitive behaviors compose all detection phases:
  OPTIMIZE — sampler picks params, normal operation
  HOLD     — fixed neutral params, passive measurement
  PROBE    — extreme params, active stimulus

States:
  OPTIMIZE       — Normal optimization
  HOLD_CALIB     — Sender: hold neutral params, wait for partner readiness barrier
  IMPULSE_CALIB  — Sender: probe at decreasing scale until safe amplitude found
  PUSH           — Sender: probe at locked scale for N trials
  PAUSE          — Sender: hold neutral params for N trials
  DONE           — Sender: release lease, cleanup
  COOLDOWN       — Wait N trials before re-acquiring (fairness for turn-taking)
  HOLD           — Receiver: hold neutral params while sender is active

Sender flow:  OPTIMIZE → HOLD_CALIB → IMPULSE_CALIB → PUSH → PAUSE → DONE → COOLDOWN → OPTIMIZE
Receiver flow: OPTIMIZE → HOLD → OPTIMIZE

The receiver has one behavior: hold neutral params while any sender holds the lease.
The observer splits receiver hold trials by sender timestamps to detect coupling edges.

Lease design (count-based budget):
  The sender_lease table uses push_remaining / pause_remaining counters instead of a
  time-based expiry. A sender's turn lasts exactly as long as it has budget remaining.
  This avoids premature lease expiry when trials take longer than expected (DB retries,
  slow effectuation). Crash recovery is handled via last_heartbeat staleness: if the
  heartbeat is older than STALE_SENDER_MULTIPLIER × WORST_CASE_TRIAL_SECONDS, another
  breeder may force-acquire the lease.
"""

import json
import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class DetectionCoordinator:
    """Manages the coordinated detection lifecycle for a single breeder worker.

    The coordinator is called once per trial via decide_trial() and returns
    what mode the breeder should operate in for that trial.

    Coordination between breeders happens through:
    1. sender_lease table — fencing token lease with phase communication and
       count-based budget (push_remaining / pause_remaining)
    2. detection_readiness table — barrier for coordinated calibration
    """

    # States
    OPTIMIZE = "optimize"
    HOLD_CALIB = "hold_calib"
    IMPULSE_CALIB = "impulse_calib"
    PUSH = "push"
    PAUSE = "pause"
    DONE = "done"
    COOLDOWN = "cooldown"
    HOLD = "hold"  # receiver

    # Safety limits
    WORST_CASE_TRIAL_SECONDS = 600  # estimated worst-case single-trial duration
    STALE_SENDER_MULTIPLIER = 5     # stale = last_heartbeat older than N × worst case
    MIN_IMPULSE_SCALE = 0.125
    MAX_HOLD_TRIALS = 100
    MAX_CALIB_WAIT = 30  # extra trials beyond hold_calib_trials to wait for partner
    MAX_HOLD_CALIB_SEARCH = 40  # max hold_calib trials before giving up on flat params

    # Hold calibration: how flat the signal must be
    MAX_CALIB_STD = 0.05    # objective std must be below this to accept params
    MIN_CALIB_SAMPLES = 5   # need at least this many samples to evaluate flatness
    CALIB_STEP_FACTOR = 0.5  # how much to reduce each param on each flatness fail

    def __init__(
        self,
        breeder_id: str,
        config: Dict[str, Any],
        shared_db_fn: Callable,
        collect_upper_bounds_fn: Callable,
        compute_neutral_params_fn: Optional[Callable] = None,
    ):
        self.breeder_id = breeder_id
        self.config = config
        self._db = shared_db_fn
        self._collect_upper_bounds = collect_upper_bounds_fn
        self._compute_neutral_params_fn = compute_neutral_params_fn
        self._breeder_db_name = f"breeder_{breeder_id.replace('-', '_')}"

        # Config
        det_cfg = config.get('interference_detection', config.get('detection', {}))
        self.min_optimize_trials = det_cfg.get('min_optimize_trials', 15)
        self.hold_calib_trials = det_cfg.get('hold_calib_trials', 5)
        self.push_block_size = det_cfg.get('push_block_size', 15)
        self.pause_block_size = det_cfg.get('pause_block_size', 15)
        self.cooldown_trials = det_cfg.get('cooldown_trials', 5)

        # State
        self.state = self.OPTIMIZE
        self._initialized = False

        # Counters (in-memory, per-worker)
        self._hold_calib_count = 0
        self._push_count = 0
        self._pause_count = 0
        self._cooldown_count = 0
        self._hold_count = 0
        self._hold_calib_receiver_count = 0

        # Impulse calibration tracking
        self._calib_scale = 1.0
        self._calib_sent = False
        self._last_calib_failed = False
        self._locked_scale = 1.0

        # Params (computed once, cached)
        self._neutral_params = None
        self._hold_params_from_config = False  # True when hold_params came from YAML

        # Hold calibration — objective values observed during hold_calib
        self._calib_values = []  # list of (objective_index, value) tuples
        self._calib_evaluated = False
        self._calib_params_locked = False

        # Lease
        self._lease_token = 0

        # Readiness barrier
        self._ready_signaled = False

    # ─── Database Setup ──────────────────────────────────────────────

    def _stale_interval(self) -> str:
        """Return the stale threshold as a plain seconds string for SQL INTERVAL."""
        return str(self.STALE_SENDER_MULTIPLIER * self.WORST_CASE_TRIAL_SECONDS)

    def _ensure_tables(self):
        """Create lease table and readiness table. Called once at init.

        The sender_lease table uses a count-based budget (push_remaining,
        pause_remaining) instead of a time-based expiry. The lease does NOT
        expire by time — it lasts as long as the sender has budget remaining.
        last_heartbeat is used solely for crash-recovery staleness detection.
        """
        def op(conn):
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sender_lease (
                    id INT PRIMARY KEY DEFAULT 1,
                    holder VARCHAR(255),
                    token INT DEFAULT 0,
                    phase VARCHAR(50),
                    push_remaining INT DEFAULT 0,
                    pause_remaining INT DEFAULT 0,
                    last_heartbeat TIMESTAMPTZ,
                    CHECK (id = 1)
                )
            """)
            cur.execute(
                "INSERT INTO sender_lease "
                "(id, holder, token, phase, push_remaining, pause_remaining, last_heartbeat) "
                "VALUES (1, NULL, 0, NULL, 0, 0, NULL) ON CONFLICT (id) DO NOTHING"
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detection_readiness (
                    breeder_id VARCHAR(255) PRIMARY KEY,
                    ready_for VARCHAR(50) NOT NULL,
                    ready_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.close()
        try:
            self._db(op, "ensure_tables")
        except Exception as e:
            logger.warning(f"Failed to ensure tables: {e}")

    def _cleanup_stale_state(self):
        """Clear stale lease and readiness rows from previous runs.

        A lease is stale if last_heartbeat is older than the stale threshold
        (STALE_SENDER_MULTIPLIER × WORST_CASE_TRIAL_SECONDS). This handles
        crash recovery for senders that died without releasing.
        """
        stale = self._stale_interval()
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease "
                "SET holder = NULL, phase = NULL, "
                "push_remaining = 0, pause_remaining = 0 "
                "WHERE id = 1 AND holder IS NOT NULL "
                "AND (last_heartbeat IS NULL "
                "OR last_heartbeat < NOW() - INTERVAL '" + stale + " seconds')"
            )
            cur.execute(
                "DELETE FROM detection_readiness "
                "WHERE ready_at < NOW() - INTERVAL '10 minutes'"
            )
            cur.close()
        try:
            self._db(op, "cleanup_stale_state")
        except Exception as e:
            logger.warning(f"Failed to cleanup stale state: {e}")

    # ─── Lease Management ────────────────────────────────────────────

    def _try_acquire_lease(self, phase: str) -> bool:
        """Try to acquire the sender lease and set initial phase.

        Atomic conditional UPDATE — no race condition possible.
        Lease is free if holder is NULL or the current holder is stale
        (last_heartbeat older than the stale threshold).

        On acquisition, budget counters are zeroed; they are set later when
        the sender enters PUSH / PAUSE.
        """
        stale = self._stale_interval()
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease "
                "SET holder = %s, token = token + 1, phase = %s, "
                "push_remaining = 0, pause_remaining = 0, "
                "last_heartbeat = NOW() "
                "WHERE id = 1 AND ("
                "holder IS NULL "
                "OR last_heartbeat IS NULL "
                "OR last_heartbeat < NOW() - INTERVAL '" + stale + " seconds'"
                ")",
                (self.breeder_id, phase)
            )
            updated = cur.rowcount
            if updated > 0:
                cur.execute("SELECT token FROM sender_lease WHERE id = 1")
                self._lease_token = cur.fetchone()[0]
            cur.close()
            return updated > 0
        try:
            return self._db(op, "acquire_lease")
        except Exception as e:
            logger.warning(f"Failed to acquire lease: {e}")
            return False

    def _set_lease_phase(self, phase: str, push_budget: Optional[int] = None,
                         pause_budget: Optional[int] = None):
        """Update the phase column on the lease.

        Optionally set the push or pause budget atomically in the same UPDATE.
        This is used when entering PUSH (push_budget = push_block_size) or
        PAUSE (pause_budget = pause_block_size).
        """
        def op(conn):
            cur = conn.cursor()
            sets: list = ["phase = %s"]
            params: list = [phase]
            if push_budget is not None:
                sets.append("push_remaining = %s")
                params.append(push_budget)
            if pause_budget is not None:
                sets.append("pause_remaining = %s")
                params.append(pause_budget)
            params.extend([self.breeder_id, self._lease_token])
            cur.execute(
                "UPDATE sender_lease SET " + ", ".join(sets) + " "
                "WHERE id = 1 AND holder = %s AND token = %s",
                params
            )
            cur.close()
        try:
            self._db(op, "set_lease_phase")
        except Exception as e:
            logger.warning(f"Failed to set lease phase: {e}")

    def _heartbeat(self) -> bool:
        """Update last_heartbeat for the current lease.

        Returns False if we lost the lease (another breeder took over via
        stale recovery, so our token no longer matches). This provides
        fencing — a stolen lease is detected on the next heartbeat.
        """
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease SET last_heartbeat = NOW() "
                "WHERE id = 1 AND holder = %s AND token = %s",
                (self.breeder_id, self._lease_token)
            )
            result = cur.rowcount > 0
            cur.close()
            return result
        try:
            return self._db(op, "heartbeat")
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False

    def _decrement_budget(self, phase: str):
        """Decrement the budget counter for the given phase by 1.

        Called after each sender trial in PUSH or PAUSE completes.
        Uses GREATEST(col - 1, 0) to prevent negative values.
        Also updates the phase label in the same atomic UPDATE.
        """
        def op(conn):
            cur = conn.cursor()
            if phase == self.PUSH:
                cur.execute(
                    "UPDATE sender_lease "
                    "SET phase = %s, "
                    "push_remaining = GREATEST(push_remaining - 1, 0) "
                    "WHERE id = 1 AND holder = %s AND token = %s",
                    (phase, self.breeder_id, self._lease_token)
                )
            elif phase == self.PAUSE:
                cur.execute(
                    "UPDATE sender_lease "
                    "SET phase = %s, "
                    "pause_remaining = GREATEST(pause_remaining - 1, 0) "
                    "WHERE id = 1 AND holder = %s AND token = %s",
                    (phase, self.breeder_id, self._lease_token)
                )
            cur.close()
        try:
            self._db(op, "decrement_budget")
        except Exception as e:
            logger.warning(f"Failed to decrement budget: {e}")

    def _release_lease(self):
        """Release the lease so partner can become sender.

        Zeros the budget counters so _has_active_sender() immediately
        returns False for this holder.
        """
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease "
                "SET holder = NULL, phase = NULL, "
                "push_remaining = 0, pause_remaining = 0 "
                "WHERE id = 1 AND holder = %s AND token = %s",
                (self.breeder_id, self._lease_token)
            )
            cur.close()
        try:
            self._db(op, "release_lease")
            logger.info(f"Released sender lease (token={self._lease_token})")
        except Exception as e:
            logger.warning(f"Failed to release lease: {e}")

    def _has_active_sender(self) -> bool:
        """Check if any breeder holds a valid (active) lease.

        A sender is active if holder IS NOT NULL AND:
        - push_remaining > 0 OR pause_remaining > 0  (budget-based: PUSH/PAUSE)
        - OR last_heartbeat is fresh (not stale)      (calibration phases)

        The budget check prevents premature HOLD exit during long PUSH/PAUSE
        blocks. The heartbeat check covers HOLD_CALIB / IMPULSE_CALIB phases
        where no budget has been set yet, and also serves as a staleness guard
        so a crashed sender's stale lease is not considered active.
        """
        stale = self._stale_interval()
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT count(*) FROM sender_lease "
                "WHERE id = 1 AND holder IS NOT NULL "
                "AND (push_remaining > 0 OR pause_remaining > 0 "
                "OR last_heartbeat > NOW() - INTERVAL '" + stale + " seconds')"
            )
            result = cur.fetchone()[0] > 0
            cur.close()
            return result
        try:
            return self._db(op, "has_active_sender")
        except Exception:
            return False

    def _get_lease_phase(self) -> Optional[str]:
        """Read the current phase from the active lease.

        Uses the same active-sender conditions as _has_active_sender().
        Returns None if no sender is active.
        """
        stale = self._stale_interval()
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT phase FROM sender_lease "
                "WHERE id = 1 AND holder IS NOT NULL "
                "AND (push_remaining > 0 OR pause_remaining > 0 "
                "OR last_heartbeat > NOW() - INTERVAL '" + stale + " seconds')"
            )
            row = cur.fetchone()
            cur.close()
            return row[0] if row else None
        try:
            return self._db(op, "get_lease_phase")
        except Exception:
            return None

    def _is_stale(self) -> bool:
        """Check if the current lease holder's heartbeat is stale.

        Used for diagnostics / logging. The actual stale-recovery logic
        lives in _try_acquire_lease and _cleanup_stale_state.
        """
        stale = self._stale_interval()
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT count(*) FROM sender_lease "
                "WHERE id = 1 AND holder IS NOT NULL "
                "AND (last_heartbeat IS NULL "
                "OR last_heartbeat < NOW() - INTERVAL '" + stale + " seconds')"
            )
            result = cur.fetchone()[0] > 0
            cur.close()
            return result
        try:
            return self._db(op, "is_stale")
        except Exception:
            return False

    # ─── Readiness Barrier ───────────────────────────────────────────

    def _signal_ready(self, phase: str):
        """Signal that this breeder is ready for the given phase."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO detection_readiness (breeder_id, ready_for, ready_at) "
                "VALUES (%s, %s, NOW()) "
                "ON CONFLICT (breeder_id) DO UPDATE "
                "SET ready_for = %s, ready_at = NOW()",
                (self.breeder_id, phase, phase)
            )
            cur.close()
        try:
            self._db(op, "signal_ready")
        except Exception as e:
            logger.warning(f"Failed to signal ready: {e}")

    def _clear_ready(self):
        """Clear our readiness signal."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM detection_readiness WHERE breeder_id = %s",
                (self.breeder_id,)
            )
            cur.close()
        try:
            self._db(op, "clear_ready")
        except Exception:
            pass

    def _check_all_ready(self, phase: str) -> bool:
        """Check if all active breeders have signaled readiness for this phase.

        A breeder is 'active' if seen in interference_active_breeders within 6 min.
        A readiness signal is valid for 3 minutes.
        Requires at least 2 active breeders.

        NOTE: This barrier is ONLY used when hold_params are NOT from config
        (i.e. during flatness-search calibration). When hold_params come from
        config, the readiness barrier is skipped entirely.
        """
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM interference_active_breeders "
                "WHERE last_seen > NOW() - INTERVAL '360 seconds'"
            )
            n_active = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM detection_readiness "
                "WHERE ready_for = %s AND ready_at > NOW() - INTERVAL '180 seconds'",
                (phase,)
            )
            n_ready = cur.fetchone()[0]
            cur.close()
            return n_ready >= n_active and n_active >= 2
        try:
            return self._db(op, "check_all_ready")
        except Exception as e:
            logger.warning(f"Failed to check readiness: {e}")
            return False

    def _count_active_breeders(self) -> int:
        """Count breeders active in the last 6 minutes."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM interference_active_breeders "
                "WHERE last_seen > NOW() - INTERVAL '360 seconds'"
            )
            count = cur.fetchone()[0]
            cur.close()
            return count
        try:
            return self._db(op, "count_active_breeders")
        except Exception:
            return 1

    # ─── Params ──────────────────────────────────────────────────────

    def _get_neutral_params(self) -> Optional[Dict[str, Any]]:
        """Get or compute neutral hold params.

        Priority:
        1. Config-specified hold_params (detection.hold_params) — if present, use directly
        2. Cached params from calibration search
        3. Callback (compute_neutral_params_fn) — strain-aware midpoints
        4. Fallback: flat midpoints from constraint ranges
        """
        if self._neutral_params is not None:
            return self._neutral_params

        # Check config override first
        det_cfg = self.config.get('interference_detection', self.config.get('detection', {}))
        hold_params = det_cfg.get('hold_params')
        if hold_params:
            self._neutral_params = hold_params
            self._hold_params_from_config = True
            logger.info("Using config-specified hold params (skipping calibration search)")
            return hold_params

        if self._compute_neutral_params_fn:
            params = self._compute_neutral_params_fn()
            if params:
                self._neutral_params = params
                logger.info(f"Computed neutral hold params via callback")
                return params

        # Fallback: compute flat midpoints from constraint ranges
        upper_bounds = self._collect_upper_bounds(self.config.get('settings', {}))
        if not upper_bounds:
            return None
        params = {}
        for ub in upper_bounds:
            midpoint = (ub['lower'] + ub['upper']) / 2.0
            if ub.get('is_int'):
                midpoint = int(midpoint)
            params[ub['name']] = midpoint
        if params:
            self._neutral_params = params
            logger.info(f"Computed neutral hold params from midpoints (fallback)")
        return params

    def _get_impulse_params(self, scale: float) -> Optional[Dict[str, Any]]:
        """Generate extreme params from neutral template at given scale.

        Takes the top-3 params by range and pushes them toward upper bound × scale.
        """
        neutral = self._get_neutral_params()
        if neutral is None:
            return None

        upper_bounds = self._collect_upper_bounds(self.config.get('settings', {}))
        upper_bounds.sort(key=lambda x: x.get('range', 0), reverse=True)

        params = dict(neutral)
        for ub in upper_bounds[:3]:
            name = ub['name']
            value = ub['upper'] * scale
            if ub.get('is_int'):
                value = int(value)
            if name in params:
                if isinstance(params[name], list):
                    params[name] = [value] * len(params[name])
                else:
                    params[name] = value
        return params

    # ─── Trial Counting ──────────────────────────────────────────────

    def _count_complete_trials_db(self) -> int:
        """Count COMPLETE trials from the breeder's DB.

        Used to determine if enough optimization has happened to start detection.
        study.trials is local cache per worker and misses trials from other workers.
        """
        import os
        try:
            import psycopg2
            user = os.environ.get('GODON_ARCHIVE_DB_USER', 'yugabyte')
            pw = os.environ.get('GODON_ARCHIVE_DB_PASSWORD', 'yugabyte')
            host = os.environ.get('GODON_ARCHIVE_DB_SERVICE_HOST', 'localhost')
            port = os.environ.get('GODON_ARCHIVE_DB_SERVICE_PORT', '5433')
            conn = psycopg2.connect(
                f"host={host} port={port} user={user} password={pw} "
                f"dbname={self._breeder_db_name}"
            )
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM trials WHERE state = 'COMPLETE'")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except Exception as e:
            logger.warning(f"DB trial count failed: {e}")
            return -1

    # ─── Main State Machine ──────────────────────────────────────────

    def decide_trial(self, trial) -> Dict[str, Any]:
        """Main entry point. Called once per trial.

        Returns dict with:
            'mode': 'optimize' | 'hold' | 'impulse'
            'params': dict or None (None = breeder samples normally)
            'impulse_phase': str or None (tag for observer windowing)
            'detection_trial': bool (whether to flag for optimizer)
        """
        # First-time initialization
        if not self._initialized:
            self._ensure_tables()
            self._cleanup_stale_state()
            self._initialized = True

        # ── Heartbeat for sender states ──────────────────────────────
        # Updates last_heartbeat (for crash recovery) and verifies we still
        # hold the lease via the fencing token. If another breeder stole the
        # lease (via stale recovery), our token won't match and we bail out.
        if self.state in (self.HOLD_CALIB, self.IMPULSE_CALIB,
                          self.PUSH, self.PAUSE):
            if not self._heartbeat():
                logger.warning("Lost sender lease — returning to OPTIMIZE")
                self.state = self.OPTIMIZE
                self._clear_ready()
                return {'mode': 'optimize', 'params': None,
                        'detection_trial': False}

        # ── Dispatch by state ────────────────────────────────────────
        if self.state == self.OPTIMIZE:
            return self._handle_optimize(trial)
        if self.state == self.HOLD_CALIB:
            return self._handle_hold_calib(trial)
        if self.state == self.IMPULSE_CALIB:
            return self._handle_impulse_calib(trial)
        if self.state == self.PUSH:
            return self._handle_push(trial)
        if self.state == self.PAUSE:
            return self._handle_pause(trial)
        if self.state == self.DONE:
            return self._handle_done(trial)
        if self.state == self.COOLDOWN:
            return self._handle_cooldown(trial)
        if self.state == self.HOLD:
            return self._handle_hold(trial)

        # Unknown state — reset
        logger.warning(f"Unknown state {self.state} — resetting to OPTIMIZE")
        self.state = self.OPTIMIZE
        return {'mode': 'optimize', 'params': None, 'detection_trial': False}

    # ── OPTIMIZE ─────────────────────────────────────────────────────

    def _handle_optimize(self, trial) -> Dict[str, Any]:
        """Normal optimization. Check if ready to enter detection."""
        # Need enough trials for effectuation_params templates
        complete = self._count_complete_trials_db()
        if complete < 0:
            return self._optimize_result()
        if complete < self.min_optimize_trials:
            return self._optimize_result()

        # Need at least 2 active breeders for detection
        if self._count_active_breeders() < 2:
            return self._optimize_result()

        # Try to become sender
        if self._try_acquire_lease(self.HOLD_CALIB):
            logger.info("Acquired lease — entering HOLD_CALIB")
            self.state = self.HOLD_CALIB
            self._hold_calib_count = 0
            self._ready_signaled = False
            self._neutral_params = None  # force recompute with fresh data
            return self._handle_hold_calib(trial)

        # Someone else is sender — become receiver
        if self._has_active_sender():
            self.state = self.HOLD
            self._hold_count = 0
            self._hold_calib_receiver_count = 0
            self._ready_signaled = False
            return self._handle_hold(trial)

        return self._optimize_result()

    def _optimize_result(self) -> Dict[str, Any]:
        return {'mode': 'optimize', 'params': None, 'detection_trial': False}

    # ── HOLD_CALIB (sender) ──────────────────────────────────────────

    def _evaluate_hold_flatness(self) -> bool:
        """Evaluate whether current hold params produce a flat enough signal.

        Two criteria:
        1. System must be alive (obj0 mean > MIN_CALIB_MEAN)
        2. Signal must be quiet (obj0 absolute std < MAX_CALIB_STD)
        """
        if len(self._calib_values) < self.MIN_CALIB_SAMPLES:
            return False

        import statistics
        obj0_vals = [v for idx, v in self._calib_values if idx == 0]
        if len(obj0_vals) < self.MIN_CALIB_SAMPLES:
            return False

        std = statistics.stdev(obj0_vals) if len(obj0_vals) >= 2 else 999
        mean = statistics.mean(obj0_vals)
        logger.info(
            f"HOLD_CALIB flatness check: obj0 mean={mean:.4f} std={std:.4f} "
            f"(need mean > 0.1 AND std < {self.MAX_CALIB_STD})"
        )
        # Must be alive AND quiet
        if mean < 0.1:
            logger.info(f"HOLD_CALIB: system dead (mean={mean:.4f} < 0.1) — rejecting")
            return False
        return std <= self.MAX_CALIB_STD

    def _adjust_hold_params(self):
        """Current hold params are too noisy or system is dead — adjust.

        If system is dead (mean ~0), shift back toward midpoints.
        If system is noisy, shift toward lower bounds.
        """
        if self._neutral_params is None:
            return

        import statistics
        obj0_vals = [v for idx, v in self._calib_values if idx == 0]
        mean = statistics.mean(obj0_vals) if obj0_vals else 0.0

        upper_bounds = self._collect_upper_bounds(self.config.get('settings', {}))
        ub_map = {ub['name']: ub for ub in upper_bounds}

        # If dead, reset to midpoints
        if mean < 0.1:
            for name in list(self._neutral_params.keys()):
                if name in ub_map:
                    ub = ub_map[name]
                    midpoint = (ub['lower'] + ub['upper']) / 2.0
                    if ub.get('is_int'):
                        midpoint = int(midpoint)
                    if isinstance(self._neutral_params[name], list):
                        self._neutral_params[name] = [midpoint] * len(self._neutral_params[name])
                    else:
                        self._neutral_params[name] = midpoint
            self._calib_values = []
            logger.info("HOLD_CALIB: system dead — resetting to midpoints")
            return

        # If noisy, shift toward lower bounds
        changed = False
        for name, value in list(self._neutral_params.items()):
            if name not in ub_map:
                continue
            ub = ub_map[name]
            lower = ub['lower']
            upper = ub['upper']
            if isinstance(value, list):
                new_vals = [lower + (v - lower) * (1.0 - self.CALIB_STEP_FACTOR) for v in value]
                if ub.get('is_int'):
                    new_vals = [int(v) for v in new_vals]
                self._neutral_params[name] = new_vals
            else:
                new_value = lower + (value - lower) * (1.0 - self.CALIB_STEP_FACTOR)
                if ub.get('is_int'):
                    new_value = int(new_value)
                self._neutral_params[name] = new_value
            changed = True

        if changed:
            self._calib_values = []
            logger.info("HOLD_CALIB: params too noisy — shifted toward lower bounds")

    def _handle_hold_calib(self, trial) -> Dict[str, Any]:
        """Sender: hold neutral params, wait for partner readiness.

        When hold_params come from config (user-specified), skip the
        flatness search AND the readiness barrier entirely — just lock
        the params and proceed directly to IMPULSE_CALIB.

        When hold_params are computed (midpoints/callback), run the
        flatness search to find stable params, then wait for the readiness
        barrier (both agents must be quiet simultaneously).
        """
        params = self._get_neutral_params()
        if not params:
            logger.warning("HOLD_CALIB: no neutral params — aborting")
            self.state = self.DONE
            return self._handle_done(trial)

        self._hold_calib_count += 1

        # ── Config params: skip flatness search AND readiness barrier ──
        # The barrier is only needed for the flatness-search case where both
        # agents must be quiet simultaneously. With config params, the sender
        # just needs its own params locked, then proceeds directly.
        if self._hold_params_from_config:
            if not self._calib_params_locked:
                self._calib_params_locked = True
                logger.info(
                    "HOLD_CALIB: params from config — "
                    "skipping flatness search and readiness barrier"
                )
            self._clear_ready()
            self.state = self.IMPULSE_CALIB
            self._calib_scale = 1.0
            self._calib_sent = False
            self._last_calib_failed = False
            return self._handle_impulse_calib(trial)

        # ── Computed params: flatness search + readiness barrier ──────
        self._set_lease_phase(self.HOLD_CALIB)

        if (self._hold_calib_count >= self.MIN_CALIB_SAMPLES
                and not self._calib_params_locked):
            if self._evaluate_hold_flatness():
                self._calib_params_locked = True
                logger.info("HOLD_CALIB: params locked — signal is flat enough")
            else:
                # Too noisy — adjust and continue holding
                self._adjust_hold_params()
                params = self._get_neutral_params()

                # Safety: don't search forever
                if self._hold_calib_count > self.MAX_HOLD_CALIB_SEARCH:
                    logger.warning(
                        "HOLD_CALIB: could not find flat params after "
                        f"{self._hold_calib_count} trials — accepting best effort"
                    )
                    self._calib_params_locked = True

        # Signal readiness once params are locked
        if self._calib_params_locked and not self._ready_signaled:
            self._signal_ready(self.HOLD_CALIB)
            self._ready_signaled = True
            logger.info("HOLD_CALIB: signaled readiness (params locked)")

        # Check barrier
        if self._ready_signaled and self._check_all_ready(self.HOLD_CALIB):
            logger.info("HOLD_CALIB: all breeders ready — entering IMPULSE_CALIB")
            self._clear_ready()
            self.state = self.IMPULSE_CALIB
            self._calib_scale = 1.0
            self._calib_sent = False
            self._last_calib_failed = False
            return self._handle_impulse_calib(trial)

        # Safety: don't wait forever for partner
        if self._hold_calib_count > self.hold_calib_trials + self.MAX_CALIB_WAIT:
            logger.warning("HOLD_CALIB: partner never became ready — aborting")
            self.state = self.DONE
            return self._handle_done(trial)

        return {
            'mode': 'hold', 'params': dict(params),
            'impulse_phase': 'hold_calib',
            'detection_trial': True,
        }

    def record_calib_observation(self, trial_values: list):
        """Called by the worker after each hold_calib trial completes.

        Stores objective values for the flatness evaluation.
        """
        if not self._calib_params_locked:
            for idx, val in enumerate(trial_values):
                if val is not None:
                    self._calib_values.append((idx, float(val)))
            logger.debug(f"HOLD_CALIB: recorded {len(trial_values)} objective values "
                        f"(total samples: {len(self._calib_values)})")

    # ── IMPULSE_CALIB (sender) ───────────────────────────────────────

    def _handle_impulse_calib(self, trial) -> Dict[str, Any]:
        """Sender: probe at decreasing scale until safe amplitude found.

        Exponential backoff: 1.0 → 0.5 → 0.25 → 0.125.
        First scale that passes guardrails is locked for the entire push block.
        If nothing passes down to MIN_IMPULSE_SCALE, abort the round.
        """
        self._set_lease_phase(self.IMPULSE_CALIB)

        if not self._calib_sent:
            # Send a probe at current scale
            params = self._get_impulse_params(self._calib_scale)
            if not params:
                self.state = self.DONE
                return self._handle_done(trial)
            self._calib_sent = True
            logger.info(f"IMPULSE_CALIB: probing at scale {self._calib_scale:.3f}")
            return {
                'mode': 'impulse', 'params': params,
                'impulse_phase': 'impulse_calib',
                'impulse_scale': self._calib_scale,
                'detection_trial': True,
            }

        # Previous probe result is in (via on_guardrail_fail flag)
        if self._last_calib_failed:
            self._last_calib_failed = False
            self._calib_sent = False
            self._calib_scale *= 0.5

            if self._calib_scale < self.MIN_IMPULSE_SCALE:
                logger.warning(
                    f"IMPULSE_CALIB: scale {self._calib_scale:.3f} below minimum "
                    f"— aborting round"
                )
                self.state = self.DONE
                return self._handle_done(trial)

            # Send next probe at lower scale
            params = self._get_impulse_params(self._calib_scale)
            if not params:
                self.state = self.DONE
                return self._handle_done(trial)
            self._calib_sent = True
            logger.info(
                f"IMPULSE_CALIB: backing off to scale {self._calib_scale:.3f}"
            )
            return {
                'mode': 'impulse', 'params': params,
                'impulse_phase': 'impulse_calib',
                'impulse_scale': self._calib_scale,
                'detection_trial': True,
            }

        # Probe passed — lock scale and start pushing immediately
        self._locked_scale = self._calib_scale
        logger.info(
            f"IMPULSE_CALIB: safe scale {self._locked_scale:.3f} — entering PUSH"
        )
        self.state = self.PUSH
        self._push_count = 0
        return self._handle_push(trial)

    # ── PUSH (sender) ────────────────────────────────────────────────

    def _handle_push(self, trial) -> Dict[str, Any]:
        """Sender: probe at locked scale for push_block_size trials.

        On entry (first trial), sets push_remaining = push_block_size in the
        lease so the receiver knows the sender is budgeted for N push trials.
        Each subsequent trial decrements the budget.

        Guardrail FAILs during push are logged but do NOT change the scale.
        The effectuation happened — the coupling already propagated.
        AIMD moves to between rounds: adjust next round's starting scale
        based on this round's FAIL rate.
        """
        # Budget management: set on entry, decrement on subsequent trials
        if self._push_count == 0:
            # Entering PUSH — initialize the push budget
            logger.info(
                f"PUSH: entering push phase — "
                f"scale={self._locked_scale:.3f}, budget={self.push_block_size} trials"
            )
            self._set_lease_phase(self.PUSH, push_budget=self.push_block_size)
        else:
            # Continuing — previous PUSH trial completed, decrement budget
            self._decrement_budget(self.PUSH)

        params = self._get_impulse_params(self._locked_scale)
        if not params:
            self.state = self.DONE
            return self._handle_done(trial)

        self._push_count += 1
        if self._push_count >= self.push_block_size:
            logger.info(
                f"PUSH: {self._push_count}/{self.push_block_size} done → PAUSE"
            )
            self.state = self.PAUSE
            self._pause_count = 0

        return {
            'mode': 'impulse', 'params': params,
            'impulse_phase': 'push',
            'impulse_scale': self._locked_scale,
            'detection_trial': True,
        }

    # ── PAUSE (sender) ───────────────────────────────────────────────

    def _handle_pause(self, trial) -> Dict[str, Any]:
        """Sender: hold neutral params for pause_block_size trials.

        On entry (first trial), sets pause_remaining = pause_block_size in the
        lease. Each subsequent trial decrements the budget.

        This is the ABA recovery — the sender returns to the same neutral params
        the receiver is holding at. The coupling signal should recover.
        The observer uses push vs pause to compute the falling edge.
        """
        # Budget management: set on entry, decrement on subsequent trials
        if self._pause_count == 0:
            # Entering PAUSE — initialize the pause budget
            logger.info(
                f"PAUSE: entering pause phase — budget={self.pause_block_size} trials"
            )
            self._set_lease_phase(self.PAUSE, pause_budget=self.pause_block_size)
        else:
            # Continuing — previous PAUSE trial completed, decrement budget
            self._decrement_budget(self.PAUSE)

        params = self._get_neutral_params()
        if not params:
            self.state = self.DONE
            return self._handle_done(trial)

        self._pause_count += 1
        if self._pause_count >= self.pause_block_size:
            logger.info(
                f"PAUSE: {self._pause_count}/{self.pause_block_size} done → DONE"
            )
            self.state = self.DONE
            return self._handle_done(trial)

        return {
            'mode': 'hold', 'params': dict(params),
            'impulse_phase': 'pause',
            'detection_trial': True,
        }

    # ── DONE (sender) ────────────────────────────────────────────────

    def _handle_done(self, trial) -> Dict[str, Any]:
        """Release the lease and enter cooldown for turn-taking fairness."""
        self._release_lease()
        self._clear_ready()
        self.state = self.COOLDOWN
        self._cooldown_count = 0
        logger.info("DONE: released lease, entering cooldown")
        # This trial is an optimize trial — the transition trial
        return {'mode': 'optimize', 'params': None, 'detection_trial': False}

    # ── COOLDOWN ─────────────────────────────────────────────────────

    def _handle_cooldown(self, trial) -> Dict[str, Any]:
        """Wait cooldown_trials before re-acquiring. Gives partner a chance to send."""
        self._cooldown_count += 1
        if self._cooldown_count >= self.cooldown_trials:
            logger.info("COOLDOWN: done — back to OPTIMIZE")
            self.state = self.OPTIMIZE
        return {'mode': 'optimize', 'params': None, 'detection_trial': False}

    # ── HOLD (receiver) ──────────────────────────────────────────────

    def _handle_hold(self, trial) -> Dict[str, Any]:
        """Receiver: hold neutral params while sender is active.

        The receiver's entire world is this state. It enters when someone else
        holds the lease and exits when the lease is released or becomes stale.
        If the sender is in HOLD_CALIB (non-config path), participate in the
        readiness barrier.
        """
        if not self._has_active_sender():
            logger.info("HOLD: sender finished — back to OPTIMIZE")
            self.state = self.OPTIMIZE
            self._ready_signaled = False
            return {'mode': 'optimize', 'params': None,
                    'detection_trial': False}

        # Safety: don't hold forever if sender crashed without releasing
        self._hold_count += 1
        if self._hold_count > self.MAX_HOLD_TRIALS:
            logger.warning(
                f"HOLD: hit MAX_HOLD_TRIALS ({self.MAX_HOLD_TRIALS}) — returning to OPTIMIZE"
            )
            self.state = self.OPTIMIZE
            self._ready_signaled = False
            return {'mode': 'optimize', 'params': None,
                    'detection_trial': False}

        params = self._get_neutral_params()
        if not params:
            logger.warning("HOLD: no neutral params — returning to OPTIMIZE")
            self.state = self.OPTIMIZE
            self._ready_signaled = False
            return {'mode': 'optimize', 'params': None,
                    'detection_trial': False}

        # Participate in readiness barrier if sender is in hold_calib
        # (only happens for the non-config flatness-search path)
        phase = self._get_lease_phase()
        if phase == self.HOLD_CALIB:
            self._hold_calib_receiver_count += 1
            if (self._hold_calib_receiver_count >= self.hold_calib_trials
                    and not self._ready_signaled):
                self._signal_ready(self.HOLD_CALIB)
                self._ready_signaled = True
                logger.info("HOLD: signaled readiness for hold_calib")
        else:
            self._hold_calib_receiver_count = 0

        # Log receiver state periodically — every 5 trials or on phase change
        if self._hold_count % 5 == 1 or phase != getattr(self, '_last_observed_phase', None):
            logger.info(
                f"HOLD: receiver trial {self._hold_count}, "
                f"sender phase={phase}"
            )
            self._last_observed_phase = phase

        # Receiver hold — tag with the lease phase we observed so we can trace
        # whether the receiver correctly saw push/pause/hold_calib from the sender.
        return {
            'mode': 'hold', 'params': dict(params),
            'impulse_phase': None,
            'lease_phase': phase,
            'detection_trial': True,
        }

    # ─── Guardrail Callback ──────────────────────────────────────────

    def on_guardrail_fail(self, params: Dict[str, Any]):
        """Called when a trial fails guardrails.

        During IMPULSE_CALIB: triggers exponential backoff.
        During PUSH: logged only — scale does NOT change mid-block.
        """
        if self.state == self.IMPULSE_CALIB:
            self._last_calib_failed = True
            logger.info(
                f"Impulse calib: guardrail fail at scale {self._calib_scale:.3f}"
            )
        elif self.state == self.PUSH:
            logger.warning(
                f"Push trial guardrail fail — keeping scale {self._locked_scale:.3f}"
            )

    # ─── Observability ───────────────────────────────────────────────

    def get_state(self) -> str:
        """Return current state for debugging/observability."""
        return self.state
