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
    RECEIVER_BASELINE = "receiver_baseline"  # Hold before sender starts — clean baseline
    SENDER_PUSH = "sender_push"
    SENDER_PAUSE = "sender_pause"
    SENDER_DONE = "sender_done"
    RECEIVER_HOLD = "receiver_hold"  # Hold during sender's push+pause — signal window
    RECEIVER_POST = "receiver_post"  # Hold after sender finishes — post baseline
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
        self._breeder_db_name = f"breeder_{breeder_id.replace('-', '_')}"

        # Config
        det_cfg = config.get('detection', {})
        self.warmup_target = det_cfg.get('warmup_trials', 15)
        self.push_block_size = det_cfg.get('push_block_size', 5)
        self.pause_block_size = det_cfg.get('pause_block_size', 5)
        self.receiver_baseline_trials = det_cfg.get('receiver_baseline_trials', 3)
        self.receiver_post_trials = det_cfg.get('receiver_post_trials', 3)
        self.recover_trials = det_cfg.get('recover_trials', 3)

        # State
        self.state = self.WARMUP
        self._initialized = False

        # Counters
        self._push_count = 0        # Push trials in current block
        self._pause_count = 0       # Pause trials in current block
        self._recover_count = 0     # Optimize trials in current RECOVER phase
        self._receiver_baseline_count = 0  # Pre-impulse baseline hold trials
        self._receiver_post_count = 0      # Post-impulse baseline hold trials

        # Params
        self._baseline_params = None    # Current safe operating point (best trial)
        self._impulse_params = None     # Extreme params (cached after first generation)
        self._impulse_scale = 1.0       # AIMD scale factor
        self._impulse_base_params = None  # Original baseline for AIMD re-scaling

        # Fencing token lease — tracks our current lease token for stale detection
        self._lease_token = 0

    def _cleanup_stale_rounds(self):
        """Reset coordination state.

        First init: complete all active rounds NOT owned by this breeder.
        This clears stale rounds from previous runs without killing our own
        workers' active rounds. Subsequent calls: only clean rounds >10min old."""
        def op(conn):
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detection_rounds (
                    round_id SERIAL PRIMARY KEY,
                    sender_id VARCHAR(255) NOT NULL,
                    status VARCHAR(50) DEFAULT 'active',
                    receiver_violated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
            """)
            if self._initialized:
                # Subsequent init (multi-worker): only clean old rounds
                cur.execute(
                    "UPDATE detection_rounds SET status = 'completed', completed_at = NOW() "
                    "WHERE status = 'active' AND created_at < NOW() - INTERVAL '10 minutes'"
                )
            else:
                # First init: complete ALL active rounds.
                # This is a fresh start — stale rounds from previous bench runs
                # persist in YugaByte across restacks. Multi-worker race is
                # acceptable here: if worker A already started a round, worker B's
                # cleanup completes it, A's SENDER_PING fails, A re-enters WARMUP,
                # and starts a new round on the next trial. The cost is 1 lost
                # impulse trial, not an infinite deadlock.
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

    # Safety limits — no state runs forever
    MAX_PUSH_ATTEMPTS = 20       # Hard cap on push trials (including FAILs)
    MAX_PAUSE_TRIALS = 20        # Hard cap on pause trials
    MAX_HOLD_TRIALS = 100        # Hard cap on receiver hold before giving up
    MAX_RECOVER_TRIALS = 15      # Hard cap on recovery phase
    MAX_RECEIVER_BASELINE = 15   # Hard cap on pre-impulse baseline
    MAX_RECEIVER_POST = 15       # Hard cap on post-impulse baseline

    # Lease duration: sender must heartbeat every trial (~30s). Lease expires
    # if sender crashes or hangs. 90s = 3 missed heartbeats before takeover.
    LEASE_DURATION_SECONDS = 90

    def _ensure_lease_table(self):
        """Create the sender_lease table if it doesn't exist. Called once at init."""
        def op(conn):
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sender_lease (
                    id INT PRIMARY KEY DEFAULT 1,
                    holder VARCHAR(255),
                    token INT DEFAULT 0,
                    expires_at TIMESTAMPTZ,
                    CHECK (id = 1)
                )
            """)
            # Insert singleton row if not exists
            cur.execute(
                "INSERT INTO sender_lease (id, holder, token, expires_at) "
                "VALUES (1, NULL, 0, NULL) ON CONFLICT (id) DO NOTHING"
            )
            cur.close()
        try:
            self._db(op, "ensure_lease_table")
        except Exception as e:
            logger.warning(f"Failed to create lease table: {e}")

    def _try_acquire_lease(self) -> bool:
        """Try to become the sender by acquiring the lease.
        
        Atomic conditional UPDATE — no race condition possible.
        Returns True if we got the lease, False if someone else holds it.
        """
        def op(conn):
            cur = conn.cursor()
            # Acquire: lease is free if holder is NULL or lease expired
            cur.execute(
                "UPDATE sender_lease "
                "SET holder = %s, token = token + 1, expires_at = NOW() + INTERVAL '%d seconds' "
                "WHERE id = 1 AND (holder IS NULL OR expires_at < NOW())",
                (self.breeder_id, self.LEASE_DURATION_SECONDS)
            )
            updated = cur.rowcount
            if updated > 0:
                # Read back our token
                cur.execute("SELECT token FROM sender_lease WHERE id = 1")
                self._lease_token = cur.fetchone()[0]
            cur.close()
            return updated > 0
        try:
            return self._db(op, "acquire_lease")
        except Exception as e:
            logger.warning(f"Failed to acquire lease: {e}")
            return False

    def _heartbeat_lease(self) -> bool:
        """Renew the lease. Called every trial while sender.
        
        Returns True if we still hold the lease, False if we lost it
        (another breeder took over). Uses fencing token to detect staleness.
        """
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease "
                "SET expires_at = NOW() + INTERVAL '%d seconds' "
                "WHERE id = 1 AND holder = %s AND token = %s",
                (self.LEASE_DURATION_SECONDS, self.breeder_id, self._lease_token)
            )
            result = cur.rowcount > 0
            cur.close()
            return result
        try:
            return self._db(op, "heartbeat_lease")
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False

    def _release_lease(self):
        """Release the lease so the other breeder can become sender immediately."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease SET holder = NULL "
                "WHERE id = 1 AND holder = %s AND token = %s",
                (self.breeder_id, self._lease_token)
            )
            cur.close()
        try:
            self._db(op, "release_lease")
            logger.info(f"Released sender lease (token={self._lease_token})")
        except Exception as e:
            logger.warning(f"Failed to release lease: {e}")

    def _is_sender(self) -> bool:
        """Check if we are currently the lease holder (without acquiring)."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT holder, token FROM sender_lease WHERE id = 1"
            )
            row = cur.fetchone()
            cur.close()
            if not row:
                return False
            holder, token = row
            return holder == self.breeder_id and token == self._lease_token
        try:
            return self._db(op, "is_sender")
        except Exception:
            return False

    def _has_active_sender(self) -> bool:
        """Check if any breeder currently holds a valid (non-expired) lease."""
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT count(*) FROM sender_lease WHERE id = 1 AND holder IS NOT NULL AND expires_at > NOW()"
            )
            result = cur.fetchone()[0] > 0
            cur.close()
            return result
        try:
            return self._db(op, "has_active_sender")
        except Exception:
            return False

    def _try_start_round(self) -> bool:
        """Try to become the sender by acquiring the lease."""
        if not self._try_acquire_lease():
            return False
        logger.info(f"Acquired sender lease (token={self._lease_token})")
        return True

    def _complete_my_round(self):
        """Release the lease so the other breeder can become sender."""
        self._release_lease()

    def _refresh_baseline_db(self):
        """Query DB for best trial with effectuation_params.
        Handles parallel workers — study.trials is local cache only."""
        import os
        try:
            import psycopg2
            user = os.environ.get('GODON_ARCHIVE_DB_USER', 'yugabyte')
            pw = os.environ.get('GODON_ARCHIVE_DB_PASSWORD', 'yugabyte')
            host = os.environ.get('GODON_ARCHIVE_DB_SERVICE_HOST', 'localhost')
            port = os.environ.get('GODON_ARCHIVE_DB_SERVICE_PORT', '5433')
            conn = psycopg2.connect(
                f"host={host} port={port} user={user} password={pw} dbname={self._breeder_db_name}"
            )
            cur = conn.cursor()
            # Get best trial with effectuation_params (lowest objective value for minimize)
            cur.execute("""
                SELECT value_json FROM trial_user_attributes
                WHERE key = 'effectuation_params'
                ORDER BY trial_id DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                import json as _json
                params_str = row[0]
                # YugaByte stores as double-encoded JSON: "{\"shading\": ...}"
                if isinstance(params_str, str):
                    params_str = params_str.strip()
                    # Strip outer quotes if present
                    if params_str.startswith('"') and params_str.endswith('"'):
                        params_str = params_str[1:-1]
                    # Unescape inner quotes
                    params_str = params_str.replace('\\"', '"')
                params = _json.loads(params_str) if isinstance(params_str, str) else params_str
                self._baseline_params = params if isinstance(params, dict) else None
                self._impulse_params = None
                self._impulse_base_params = None
                if self._baseline_params:
                    logger.info(f"Refreshed baseline params from DB")
        except Exception as e:
            import traceback
            logger.warning(f"DB baseline refresh failed: {e}\n{traceback.format_exc()}")

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
            self._impulse_params = None
            self._impulse_base_params = None
            logger.info(f"Refreshed baseline params (best value: {best_value:.4f})")
        else:
            # Fallback: use raw trial params from best trial if no stashed params
            best_raw = None
            best_raw_value = None
            for trial in study.trials:
                if trial.state != TrialState.COMPLETE:
                    continue
                if not trial.values:
                    continue
                v = trial.values[0]
                if best_raw_value is None or (minimize and v < best_raw_value) or (not minimize and v > best_raw_value):
                    best_raw_value = v
                    best_raw = trial
            if best_raw and best_raw.params:
                self._baseline_params = dict(best_raw.params)
                self._impulse_params = None
                self._impulse_base_params = None
                logger.info(f"Refreshed baseline from raw trial params (best value: {best_raw_value:.4f}, no stashed params found)")

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

    def _count_phase_trials_db(self, phase: str) -> int:
        """Count trials with a specific impulse_phase from the breeder's DB.
        Handles parallel workers — each worker has separate in-memory state."""
        import os
        try:
            import psycopg2
            user = os.environ.get('GODON_ARCHIVE_DB_USER', 'yugabyte')
            pw = os.environ.get('GODON_ARCHIVE_DB_PASSWORD', 'yugabyte')
            host = os.environ.get('GODON_ARCHIVE_DB_SERVICE_HOST', 'localhost')
            port = os.environ.get('GODON_ARCHIVE_DB_SERVICE_PORT', '5433')
            conn = psycopg2.connect(
                f"host={host} port={port} user={user} password={pw} dbname={self._breeder_db_name}"
            )
            cur = conn.cursor()
            cur.execute(
                "SELECT count(*) FROM trial_user_attributes "
                "WHERE key = 'impulse_phase' AND value_json = %s",
                (f'"{phase}"',)
            )
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except Exception as e:
            logger.warning(f"DB phase count failed: {e}")
            return -1

    def _count_complete_trials_db(self) -> int:
        """Count COMPLETE trials from the breeder's DB.

        study.trials is cached locally per worker and doesn't see trials
        from other parallel workers. This queries the actual DB for the
        true count."""
        import os
        try:
            import psycopg2
            user = os.environ.get('GODON_ARCHIVE_DB_USER', 'yugabyte')
            pw = os.environ.get('GODON_ARCHIVE_DB_PASSWORD', 'yugabyte')
            host = os.environ.get('GODON_ARCHIVE_DB_SERVICE_HOST', 'localhost')
            port = os.environ.get('GODON_ARCHIVE_DB_SERVICE_PORT', '5433')
            conn = psycopg2.connect(
                f"host={host} port={port} user={user} password={pw} dbname={self._breeder_db_name}"
            )
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM trials WHERE state = 'COMPLETE'")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except Exception as e:
            logger.warning(f"DB trial count failed: {e}, falling back to study.trials")
            return -1

    def decide_trial(self, trial, study) -> Dict[str, Any]:
        """Main entry point. Called once per trial.
        Returns dict with:
            'mode': 'optimize' | 'hold' | 'impulse'
            'params': dict or None (if None, breeder samples normally)
            'impulse_phase': 'ping' | 'listen' | None
        """
        import os
        debug = os.environ.get('GODON_DETECTION_DEBUG', '0') == '1'

        def _log(msg):
            if debug:
                logger.info(f"[DEBUG] breeder={self.breeder_id[:8]} state={self.state} {msg}")
                trial.set_user_attr('coord_state', self.state)
                trial.set_user_attr('coord_debug', msg[:500])

        # First-time initialization
        if not self._initialized:
            _log(f"first init — cleaning stale rounds, creating lease table")
            self._cleanup_stale_rounds()
            self._ensure_lease_table()
            self._initialized = True

        # Lease heartbeat: if we're the sender, renew the lease every trial.
        # If the heartbeat fails (another breeder took over), transition to RECOVER.
        if self.state in (self.SENDER_PUSH, self.SENDER_PAUSE):
            if not self._heartbeat_lease():
                logger.warning("Lost sender lease — another breeder took over")
                self.state = self.RECOVER
                self._recover_count = 0
                return {'mode': 'optimize', 'params': None}

        if self.state == self.WARMUP:
            # Warmup MUST complete before any detection activity.
            # Do NOT check for active rounds during warmup — stale rounds
            # from previous runs can trigger premature receiver mode.
            
            # Count COMPLETE trials from DB — study.trials local cache is unreliable
            # with YugaByte (doesn't reflect all committed trials)
            complete = self._count_complete_trials_db()
            if complete < 0:
                complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE) \
                    if study and study.trials else 0
            if complete >= self.warmup_target:
                self._refresh_baseline_db()
                if self._baseline_params is None:
                    _log(f"warmup: {complete} complete but no effectuation_params stashed — staying WARMUP")
                elif self._try_start_round():
                    self.state = self.SENDER_PUSH
                    self._push_count = 0
                    self._pause_count = 0
                    _log(f"warmup done ({complete} trials) — became SENDER")
                elif self._has_active_sender():
                    self.state = self.RECEIVER_BASELINE
                    self._receiver_baseline_count = 0
                    _log(f"warmup done ({complete} trials) — became RECEIVER_BASELINE")
                else:
                    _log(f"warmup: {complete} trials, no active round and couldn't start one — staying WARMUP")
            else:
                _log(f"warmup: {complete}/{self.warmup_target} complete trials")
            return {'mode': 'optimize', 'params': None}

        if self.state == self.SENDER_PUSH:
            params = self._get_impulse_params()
            if not params:
                _log("SENDER_PUSH: no impulse params — optimize fallback")
                return {'mode': 'optimize', 'params': None}
            self._push_count += 1
            if self._push_count >= self.push_block_size:
                self.state = self.SENDER_PAUSE
                self._pause_count = 0
                _log(f"SENDER_PUSH: push {self._push_count}/{self.push_block_size} — block complete, entering PAUSE")
            elif self._push_count >= self.MAX_PUSH_ATTEMPTS:
                # Safety: too many push attempts (all FAILing) — give up and complete round
                logger.warning(f"SENDER_PUSH: hit MAX_PUSH_ATTEMPTS ({self.MAX_PUSH_ATTEMPTS}) — completing round early")
                self.state = self.SENDER_DONE
                _log(f"SENDER_PUSH: forced completion after {self._push_count} attempts")
            else:
                _log(f"SENDER_PUSH: push {self._push_count}/{self.push_block_size}")
            return {'mode': 'impulse', 'params': params, 'impulse_phase': 'push'}

        if self.state == self.SENDER_PAUSE:
            if self._baseline_params is None:
                self._refresh_baseline_db()
            if self._baseline_params is None:
                _log("SENDER_PAUSE: no baseline — optimize fallback")
                return {'mode': 'optimize', 'params': None}
            self._pause_count += 1
            if self._pause_count >= self.pause_block_size:
                self.state = self.SENDER_DONE
                _log(f"SENDER_PAUSE: pause {self._pause_count}/{self.pause_block_size} — block complete, DONE")
            else:
                _log(f"SENDER_PAUSE: pause {self._pause_count}/{self.pause_block_size}")
            return {'mode': 'impulse', 'params': dict(self._baseline_params), 'impulse_phase': 'pause'}

        if self.state == self.SENDER_DONE:
            self._complete_my_round()
            self._push_count = 0
            self._pause_count = 0
            self._impulse_params = None
            self._recover_count = 0
            self.state = self.RECOVER
            _log("SENDER_DONE: round completed, entering RECOVER")
            return {'mode': 'optimize', 'params': None}

        if self.state == self.RECEIVER_BASELINE:
            # Pre-impulse baseline hold — receiver settles before sender pushes
            self._receiver_baseline_count += 1
            if self._receiver_baseline_count >= self.receiver_baseline_trials or \
               self._receiver_baseline_count >= self.MAX_RECEIVER_BASELINE:
                self.state = self.RECEIVER_HOLD
                self._hold_trial_count = 0
                _log(f"RECEIVER_BASELINE: {self._receiver_baseline_count} trials done — entering RECEIVER_HOLD")
            else:
                _log(f"RECEIVER_BASELINE: trial {self._receiver_baseline_count}/{self.receiver_baseline_trials}")
            if self._baseline_params:
                return {'mode': 'hold', 'params': dict(self._baseline_params), 'hold_phase': 'baseline'}
            return {'mode': 'optimize', 'params': None}

        if self.state == self.RECEIVER_HOLD:
            if not self._has_active_sender():
                # Sender finished — enter POST baseline hold
                self._receiver_post_count = 0
                self.state = self.RECEIVER_POST
                _log("RECEIVER_HOLD: sender finished — entering RECEIVER_POST")
                if self._baseline_params:
                    return {'mode': 'hold', 'params': dict(self._baseline_params), 'hold_phase': 'signal'}
                return {'mode': 'optimize', 'params': None}
            # Safety: don't hold forever if sender crashed without releasing lock
            self._hold_trial_count = getattr(self, '_hold_trial_count', 0) + 1
            if self._hold_trial_count > self.MAX_HOLD_TRIALS:
                logger.warning(f"RECEIVER_HOLD: hit MAX_HOLD_TRIALS ({self.MAX_HOLD_TRIALS}) — giving up")
                self._recover_count = 0
                self.state = self.RECOVER
                _log("RECEIVER_HOLD: forced recovery after too many hold trials")
                return {'mode': 'optimize', 'params': None}
            if self._baseline_params is None:
                self._refresh_baseline_db()
            if self._baseline_params is None:
                _log("RECEIVER_HOLD: no baseline params — optimizing")
                return {'mode': 'optimize', 'params': None}
            params = dict(self._baseline_params)
            _log("RECEIVER_HOLD: holding with baseline params (signal window)")
            return {'mode': 'hold', 'params': params, 'hold_phase': 'signal'}

        if self.state == self.RECEIVER_POST:
            self._receiver_post_count += 1
            if self._receiver_post_count >= self.receiver_post_trials or \
               self._receiver_post_count >= self.MAX_RECEIVER_POST:
                self._recover_count = 0
                self.state = self.RECOVER
                _log(f"RECEIVER_POST: {self._receiver_post_count} trials done — entering RECOVER")
                return {'mode': 'optimize', 'params': None}
            _log(f"RECEIVER_POST: trial {self._receiver_post_count}/{self.receiver_post_trials}")
            if self._baseline_params:
                return {'mode': 'hold', 'params': dict(self._baseline_params), 'hold_phase': 'post'}
            return {'mode': 'optimize', 'params': None}

        if self.state == self.RECOVER:
            self._recover_count += 1
            if self._recover_count >= self.recover_trials or \
               self._recover_count >= self.MAX_RECOVER_TRIALS:
                self._refresh_baseline_db()
                if self._baseline_params is None:
                    _log(f"RECOVER: no baseline after {self._recover_count} trials — staying RECOVER")
                    self._recover_count = 0
                elif self._try_start_round():
                    self.state = self.SENDER_PUSH
                    self._push_count = 0
                    self._pause_count = 0
                    _log(f"RECOVER done ({self._recover_count} trials) — became SENDER")
                elif self._has_active_sender():
                    self.state = self.RECEIVER_BASELINE
                    self._receiver_baseline_count = 0
                    _log(f"RECOVER done ({self._recover_count} trials) — became RECEIVER_BASELINE")
                else:
                    _log(f"RECOVER: {self._recover_count} trials, no active round and couldn't start — staying RECOVER")
                    self._recover_count = 0
            else:
                _log(f"RECOVER: trial {self._recover_count}/{self.recover_trials}")
            return {'mode': 'optimize', 'params': None}

        _log(f"UNKNOWN state {self.state} — resetting to WARMUP")
        self.state = self.WARMUP
        return {'mode': 'optimize', 'params': None}

    def on_guardrail_fail(self, params: Dict[str, Any]):
        """Called when a trial fails guardrails. Scales down impulse if in push phase.
        
        The push counter is NOT decremented — the trial happened, it just failed.
        AIMD handles parameter adjustment separately. This ensures the push block
        always completes regardless of FAIL rate.
        """
        if self.state == self.SENDER_PUSH:
            self._aimd_backoff()
            logger.info(f"Guardrail FAIL during push (trial {self._push_count}/{self.push_block_size}) — AIMD backoff to scale {self._impulse_scale:.2f}")

    def get_state(self) -> str:
        """Return current state for debugging/observability."""
        return self.state
