#
# Copyright (c) 2019 Matthias Tafelmeier.
#
# AGPL-3.0 — see godon-breeders/LICENSE.
#
"""
Detection/Characterization Coordinator — the one loop.

Implements the characterization strategy (Aug 9 design):
  probe → measure → detect → interpolate → check convergence → halt or probe next

No separate detection phase. No extremistic top-3 push. Detection and
characterization are the same process at different completeness.

PROBE STRATEGY (what gets pushed):
  ONE Optuna study with all params as dimensions. Each trial samples
  a categorical (which param to push) and a stepped float (at what
  level). TPE picks both — no coverage guard, no round-robin. The
  sampler's startup randomness IS the coverage pass.

  Direction is maximize — high delta means the interpolated surface
  is still moving (informative region). TPE concentrates sampling
  where delta is high and ignores dead params.

  Disposable studies for refinement: can't change step mid-study, so
  close and create new at halved step. Curves persist in causal's
  CurveRegistry across studies.

  INFINITY catching: causal returns f64::MAX/2 for the first point
  on each curve (no prior to compare). Coordinator catches this and
  replaces with 1.0 before telling the study.

  One trial, dual feed: the physical measurement feeds both the
  optimization study (objective values, immediate tell) and the
  characterization study (delta, delayed tell after pause + causal).

FIRST level producing receiver response above CFAR threshold = detection.
ACCUMULATED levels across a param = characterization (response curve).
Convergence delta (interpolation change) = halting + drift detection.

States:
  OPTIMIZE    — Normal optimization, accumulate baseline
  PROBE_PUSH  — Sender: push at scheduled level for N trials
  PROBE_PAUSE — Sender: return to neutral for N trials (reversibility)
  DONE        — Release lease
  COOLDOWN    — Wait before re-acquiring (turn-taking fairness)
  HOLD        — Receiver: hold neutral while sender probes

Coordination: group-scoped fencing-token lease in shared DB.
One sender at a time per group. Crash recovery via heartbeat staleness.
"""

import logging
import os
import statistics
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


class ProbeCoordinator:
    """Unified detection + characterization coordinator.

    Called once per trial via decide_trial(). Returns what mode the breeder
    should operate in for that trial.

    The probe schedule is built from config constraints by default. When
    probe_override is present in config, the schedule comes from the operator.
    """

    # States
    OPTIMIZE = "optimize"
    PROBE_PUSH = "probe_push"
    PROBE_PAUSE = "probe_pause"
    DONE = "done"
    COOLDOWN = "cooldown"
    HOLD = "hold"

    # Safety limits
    WORST_CASE_TRIAL_SECONDS = 600
    STALE_SENDER_MULTIPLIER = 5
    MAX_HOLD_TRIALS = 200
    ACTIVE_BREEDER_WINDOW_SECONDS = 360

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

        det_cfg = config.get('interference_detection', config.get('detection', {}))
        self.group_id = det_cfg.get('group', config.get('group', 'default'))

        self.min_optimize_trials = det_cfg.get('min_optimize_trials', 15)
        self.push_block_size = det_cfg.get('push_block_size', 10)
        self.pause_block_size = det_cfg.get('pause_block_size', 10)
        self.cooldown_trials = det_cfg.get('cooldown_trials', 5)
        self.active_breeder_window = det_cfg.get(
            'active_breeder_window', self.ACTIVE_BREEDER_WINDOW_SECONDS)

        # Characterization precision — the one customer-facing knob.
        # Controls how fine to measure coupling response curves.
        # System derives probe resolution from this; customer never sets
        # step sizes for characterization.
        self.convergence_threshold = det_cfg.get('convergence_threshold', 0.02)

        # Max recursive refinement insertions between two measured points.
        # The coarse pass is always bounded. The refinement (substeps
        # where the curve moves most) is what can recurse. This caps it.
        self.refinement_depth = det_cfg.get('refinement_depth', 3)

        # Convergence state — set by causal responses
        self._converged_params: set = set()

        # Round timestamp tracking — for causal probe_result calls
        self._round_push_start: Optional[datetime] = None
        self._round_pause_end: Optional[datetime] = None

        # Causal client — adaptive timeout deskewed to system rhythm.
        # Budget = "waste at most 2 trials waiting for causal."
        # Tightened by observed RTT: if causal is consistently fast,
        # don't wait the full budget.
        self._causal_rtt_history: List[float] = []  # seconds, rolling 10
        self._trial_duration_history: List[float] = []  # seconds, rolling 10
        self._causal_url = det_cfg.get(
            'causal_url',
            os.environ.get('GODON_CAUSAL_URL', 'http://godon-godon-causal:9091'))

        # State
        self.state = self.OPTIMIZE
        self._initialized = False

        # Counters
        self._optimize_count = 0
        self._push_count = 0
        self._pause_count = 0
        self._cooldown_count = 0
        self._hold_count = 0

        # Characterization — ONE study with all params as dimensions.
        # TPE picks both which param to push and at what level, guided
        # by delta from causal. Disposable per refinement level: close
        # on convergence/exhaustion, create new at halved step.
        # Curves persist in causal's CurveRegistry across studies.
        self._char_study: Any = None  # single optuna.Study
        self._char_trial: Any = None  # in-flight trial
        self._char_step: float = 0.0
        self._refinement_level = 0
        self._param_names: List[str] = []
        self._param_bounds: Dict[str, Dict] = {}  # name → {lower, upper, is_int, idx}
        self._current_probe: Optional[Dict[str, Any]] = None

        # Neutral params (cached)
        self._neutral_params = None
        self._hold_params_from_config = False

        # Lease
        self._lease_token = 0

    # ─── DB Coordination ─────────────────────────────────────────────

    def _stale_interval(self) -> str:
        return str(self.STALE_SENDER_MULTIPLIER * self.WORST_CASE_TRIAL_SECONDS)

    def _ensure_tables(self):
        def op(conn):
            cur = conn.cursor()

            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'sender_lease' AND column_name = 'id'"
            )
            if cur.fetchone():
                logger.info("Migrating sender_lease from global singleton to group-scoped schema")
                cur.execute("DROP TABLE sender_lease")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS sender_lease (
                    group_id VARCHAR(255) PRIMARY KEY,
                    holder VARCHAR(255),
                    token INT DEFAULT 0,
                    phase VARCHAR(50),
                    push_remaining INT DEFAULT 0,
                    pause_remaining INT DEFAULT 0,
                    last_heartbeat TIMESTAMPTZ
                )
            """)
            cur.execute(
                "INSERT INTO sender_lease "
                "(group_id, holder, token, phase, push_remaining, pause_remaining, last_heartbeat) "
                "VALUES (%s, NULL, 0, NULL, 0, 0, NULL) ON CONFLICT (group_id) DO NOTHING",
                (self.group_id,)
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detection_readiness (
                    breeder_id VARCHAR(255) PRIMARY KEY,
                    group_id VARCHAR(255) NOT NULL DEFAULT 'default',
                    ready_for VARCHAR(50) NOT NULL,
                    ready_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'detection_readiness' AND column_name = 'group_id'"
            )
            if not cur.fetchone():
                cur.execute(
                    "ALTER TABLE detection_readiness "
                    "ADD COLUMN IF NOT EXISTS group_id VARCHAR(255) NOT NULL DEFAULT 'default'"
                )

            # Receiver observations — written by receiver during HOLD,
            # read by sender to compute coupling shift after each probe round.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS receiver_observations (
                    id SERIAL PRIMARY KEY,
                    group_id VARCHAR(255) NOT NULL,
                    receiver_id VARCHAR(255) NOT NULL,
                    trial_num INT,
                    objective_values JSONB,
                    lease_phase VARCHAR(50),
                    written_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.close()
        try:
            self._db(op, "ensure_tables")
        except Exception as e:
            logger.warning(f"Failed to ensure tables: {e}")

    def _cleanup_stale_state(self):
        stale = self._stale_interval()
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease "
                "SET holder = NULL, phase = NULL, "
                "push_remaining = 0, pause_remaining = 0 "
                "WHERE group_id = %s AND holder IS NOT NULL "
                "AND (last_heartbeat IS NULL "
                "OR last_heartbeat < NOW() - INTERVAL '" + stale + " seconds')",
                (self.group_id,)
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

    def _try_acquire_lease(self, phase: str) -> bool:
        stale = self._stale_interval()
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease "
                "SET holder = %s, token = token + 1, phase = %s, "
                "push_remaining = 0, pause_remaining = 0, "
                "last_heartbeat = NOW() "
                "WHERE group_id = %s AND ("
                "holder IS NULL "
                "OR last_heartbeat IS NULL "
                "OR last_heartbeat < NOW() - INTERVAL '" + stale + " seconds'"
                ")",
                (self.breeder_id, phase, self.group_id)
            )
            updated = cur.rowcount
            if updated > 0:
                cur.execute("SELECT token FROM sender_lease WHERE group_id = %s",
                            (self.group_id,))
                self._lease_token = cur.fetchone()[0]
            cur.close()
            return updated > 0
        try:
            return self._db(op, "acquire_lease")
        except Exception as e:
            logger.warning(f"Failed to acquire lease: {e}")
            return False

    def _heartbeat(self) -> bool:
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease SET last_heartbeat = NOW() "
                "WHERE group_id = %s AND holder = %s AND token = %s",
                (self.group_id, self.breeder_id, self._lease_token)
            )
            result = cur.rowcount > 0
            cur.close()
            return result
        try:
            return self._db(op, "heartbeat")
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False

    def _release_lease(self):
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE sender_lease "
                "SET holder = NULL, phase = NULL, "
                "push_remaining = 0, pause_remaining = 0 "
                "WHERE group_id = %s AND holder = %s AND token = %s",
                (self.group_id, self.breeder_id, self._lease_token)
            )
            cur.close()
        try:
            self._db(op, "release_lease")
            logger.info(f"Released sender lease (token={self._lease_token})")
        except Exception as e:
            logger.warning(f"Failed to release lease: {e}")

    def _set_lease_phase(self, phase: str, push_budget: Optional[int] = None,
                         pause_budget: Optional[int] = None):
        def op(conn):
            cur = conn.cursor()
            sets = ["phase = %s"]
            params = [phase]
            if push_budget is not None:
                sets.append("push_remaining = %s")
                params.append(push_budget)
            if pause_budget is not None:
                sets.append("pause_remaining = %s")
                params.append(pause_budget)
            params.extend([self.group_id, self.breeder_id, self._lease_token])
            cur.execute(
                "UPDATE sender_lease SET " + ", ".join(sets) + " "
                "WHERE group_id = %s AND holder = %s AND token = %s",
                params
            )
            cur.close()
        try:
            self._db(op, "set_lease_phase")
        except Exception as e:
            logger.warning(f"Failed to set lease phase: {e}")

    def _decrement_budget(self, phase: str, is_push: bool):
        col = "push_remaining" if is_push else "pause_remaining"
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                f"UPDATE sender_lease SET phase = %s, "
                f"{col} = GREATEST({col} - 1, 0) "
                "WHERE group_id = %s AND holder = %s AND token = %s",
                (phase, self.group_id, self.breeder_id, self._lease_token)
            )
            cur.close()
        try:
            self._db(op, "decrement_budget")
        except Exception as e:
            logger.warning(f"Failed to decrement budget: {e}")

    def _has_active_sender(self) -> bool:
        stale = self._stale_interval()
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT count(*) FROM sender_lease "
                "WHERE group_id = %s AND holder IS NOT NULL "
                "AND last_heartbeat IS NOT NULL "
                "AND last_heartbeat > NOW() - INTERVAL '" + stale + " seconds'",
                (self.group_id,)
            )
            result = cur.fetchone()[0] > 0
            cur.close()
            return result
        try:
            return self._db(op, "has_active_sender")
        except Exception:
            return False

    def _get_lease_phase(self) -> Optional[str]:
        stale = self._stale_interval()
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT phase FROM sender_lease "
                "WHERE group_id = %s AND holder IS NOT NULL "
                "AND last_heartbeat IS NOT NULL "
                "AND last_heartbeat > NOW() - INTERVAL '" + stale + " seconds'",
                (self.group_id,)
            )
            row = cur.fetchone()
            cur.close()
            return row[0] if row else None
        try:
            return self._db(op, "get_lease_phase")
        except Exception:
            return None

    def _count_active_breeders(self) -> int:
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM interference_active_breeders "
                "WHERE group_id = %s AND last_seen > NOW() - INTERVAL '"
                + str(self.active_breeder_window) + " seconds'",
                (self.group_id,)
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
        if self._neutral_params is not None:
            return self._neutral_params

        det_cfg = self.config.get('interference_detection', self.config.get('detection', {}))
        hold_params = det_cfg.get('hold_params')
        if hold_params:
            self._neutral_params = hold_params
            self._hold_params_from_config = True
            logger.info("Using config-specified hold params")
            return hold_params

        if self._compute_neutral_params_fn:
            params = self._compute_neutral_params_fn()
            if params:
                self._neutral_params = params
                logger.info("Computed neutral hold params via callback")
                return params

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
            logger.info("Computed neutral hold params from midpoints (fallback)")
        return params

    # ─── Adaptive Timeout ────────────────────────────────────────────

    def _record_trial_duration(self, duration_secs: float):
        """Called by worker after each trial. Tracks system rhythm."""
        self._trial_duration_history.append(duration_secs)
        if len(self._trial_duration_history) > 10:
            self._trial_duration_history.pop(0)

    def _causal_timeout(self) -> float:
        """Deskewed timeout for causal calls.

        Budget = 2 × median trial duration ("waste at most 2 trials").
        Tightened = 10 × median observed causal RTT (if we know it's fast).
        Effective = min(budget, tightened), floored at 2s.

        Scales with the system:
        - Fast bench (1s trials): budget=2s, causal RTT 2ms → timeout 2s
        - Real system (45s trials): budget=90s, causal RTT 5ms → timeout 90s
        - Dead causal: timeout fires after budget, not a flat constant
        """
        import statistics as stat

        floor = 2.0

        budget = floor
        if self._trial_duration_history:
            med_trial = stat.median(self._trial_duration_history)
            budget = max(floor, med_trial * 2.0)

        if self._causal_rtt_history:
            med_rtt = stat.median(self._causal_rtt_history)
            tightened = max(floor, med_rtt * 10.0)
            return min(budget, tightened)

        # No RTT history yet — use budget as-is (generous on first call)
        return budget

    # ─── Causal Client ───────────────────────────────────────────────

    def _query_causal_probe_result(self, probe: dict) -> Optional[dict]:
        """Call causal's real-time endpoint after a probe round completes.

        Causal computes shift (CFAR-aware), updates its ResponseCurve,
        returns {shift, delta, converged}. Coordinator is the dumb
        instrument — causal owns all curves.

        Falls back to None on timeout or error. Coordinator continues
        to next scheduled probe blind.
        """
        import urllib.request
        import json as _json

        if self._round_push_start is None or self._round_pause_end is None:
            return None

        payload = {
            'group_id': self.group_id,
            'sender_id': self.breeder_id,
            'probe_param': probe['param_name'],
            'probe_level': probe['level'],
            'push_start': self._round_push_start.isoformat(),
            'pause_end': self._round_pause_end.isoformat(),
            'convergence_threshold': self.convergence_threshold,
        }

        timeout = self._causal_timeout()
        url = f"{self._causal_url}/characterize"

        import time
        t0 = time.monotonic()

        try:
            req = urllib.request.Request(
                url,
                data=_json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                rtt = time.monotonic() - t0
                self._causal_rtt_history.append(rtt)
                if len(self._causal_rtt_history) > 10:
                    self._causal_rtt_history.pop(0)

                result = _json.loads(resp.read())
                logger.info(
                    f"CAUSAL probe_result: param={probe['param_name']} "
                    f"level={probe['level']} rtt={rtt*1000:.0f}ms "
                    f"shift={result.get('shift', '?')} "
                    f"delta={result.get('delta', '?')} "
                    f"converged={result.get('converged', '?')}"
                )
                return result

        except Exception as e:
            rtt = time.monotonic() - t0
            logger.warning(
                f"CAUSAL probe_result FAILED after {rtt:.1f}s "
                f"(timeout was {timeout:.1f}s): {e} — continuing blind"
            )
            return None

    # ─── Receiver Observation Sharing ────────────────────────────────

    def record_receiver_observation(self, trial_num: int,
                                     objective_values: Dict[str, float],
                                     lease_phase: Optional[str] = None):
        """Called by the worker after each HOLD trial completes.

        Writes the receiver's objective values to the shared table so
        the sender can compute coupling shift after each probe round.
        """
        import json
        def op(conn):
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO receiver_observations "
                "(group_id, receiver_id, trial_num, objective_values, lease_phase) "
                "VALUES (%s, %s, %s, %s, %s)",
                (self.group_id, self.breeder_id, trial_num,
                 json.dumps(objective_values), lease_phase)
            )
            cur.close()
        try:
            self._db(op, "record_receiver_obs")
        except Exception as e:
            logger.warning(f"Failed to record receiver observation: {e}")

    def _compute_probe_shift(self, probe: dict) -> Optional[float]:
        """Compute receiver's coupling shift for the probe round just completed.

        Reads receiver_observations written during this push/pause window.
        Shift = median(receiver objective_0 during push) - median(during pause).

        Returns None if insufficient receiver data.
        """
        if self._round_push_start is None or self._round_pause_end is None:
            return None

        push_start = self._round_push_start
        pause_end = self._round_pause_end

        def op(conn):
            import json
            cur = conn.cursor()
            # Receiver observations during push window (lease_phase = probe_push)
            cur.execute(
                "SELECT objective_values FROM receiver_observations "
                "WHERE group_id = %s AND receiver_id != %s "
                "AND lease_phase = 'probe_push' "
                "AND written_at >= %s AND written_at <= %s",
                (self.group_id, self.breeder_id, push_start, pause_end)
            )
            push_vals = []
            for (obj_json,) in cur.fetchall():
                try:
                    obj = json.loads(obj_json) if isinstance(obj_json, str) else obj_json
                    v = obj.get('objective_0')
                    if v is not None:
                        push_vals.append(float(v))
                except (json.JSONDecodeError, TypeError):
                    pass

            # Receiver observations during pause window (lease_phase = probe_pause)
            cur.execute(
                "SELECT objective_values FROM receiver_observations "
                "WHERE group_id = %s AND receiver_id != %s "
                "AND lease_phase = 'probe_pause' "
                "AND written_at >= %s AND written_at <= %s",
                (self.group_id, self.breeder_id, push_start, pause_end)
            )
            pause_vals = []
            for (obj_json,) in cur.fetchall():
                try:
                    obj = json.loads(obj_json) if isinstance(obj_json, str) else obj_json
                    v = obj.get('objective_0')
                    if v is not None:
                        pause_vals.append(float(v))
                except (json.JSONDecodeError, TypeError):
                    pass

            cur.close()
            return push_vals, pause_vals

        try:
            push_vals, pause_vals = self._db(op, "compute_probe_shift")
        except Exception as e:
            logger.warning(f"Failed to compute probe shift: {e}")
            return None

        if len(push_vals) < 2 or len(pause_vals) < 2:
            logger.info(
                f"SHIFT: insufficient receiver data "
                f"(push={len(push_vals)}, pause={len(pause_vals)})"
            )
            return None

        push_med = statistics.median(push_vals)
        pause_med = statistics.median(pause_vals)
        shift = push_med - pause_med

        logger.info(
            f"SHIFT: param={probe['param_name']} level={probe['level']} "
            f"push_med={push_med:.4f} pause_med={pause_med:.4f} "
            f"shift={shift:+.4f} (push_n={len(push_vals)}, pause_n={len(pause_vals)})"
        )
        return shift

    def _process_probe_result(self, probe: dict) -> Optional[float]:
        """Called when a probe round (push + pause) completes.

        Calls causal to compute shift, update ResponseCurve, check
        convergence. Returns delta for the characterization study.

        Falls back gracefully: if causal is unreachable, returns None
        (coordinator continues blind, trial told as FAIL to char study).
        """
        result = self._query_causal_probe_result(probe)
        if result is None:
            logger.info(
                f"PROBE_RESULT: causal unavailable — continuing blind "
                f"for {probe['param_name']}"
            )
            return None

        delta = result.get('delta')
        converged = result.get('converged', False)
        param_name = probe['param_name']

        if converged:
            logger.info(f"CONVERGED: {param_name}")
            self._converged_params.add(param_name)

        return delta

    # ─── Characterization Study ─────────────────────────────────────

    def _init_characterization(self):
        """Create ONE characterization study with all params as dimensions.

        Each trial samples a categorical (which param to push) and a
        float (at what level). TPE learns across both: which params
        produce high delta (informative) and which levels within those
        params are worth probing. No coverage guard — the sampler's
        startup randomness IS the coverage pass.
        """
        import optuna

        neutral = self._get_neutral_params()
        if not neutral:
            return

        upper_bounds = self._collect_upper_bounds(self.config.get('settings', {}))
        if not upper_bounds:
            return

        self._param_names = []
        global_low = float('inf')
        global_high = float('-inf')
        for param_idx, ub in enumerate(upper_bounds):
            name = ub['name']
            lower = ub.get('lower', 0.0)
            upper = ub.get('upper', 100.0)
            is_int = ub.get('is_int', False)
            self._param_bounds[name] = {
                'lower': lower, 'upper': upper,
                'is_int': is_int, 'idx': param_idx,
            }
            self._param_names.append(name)
            global_low = min(global_low, lower)
            global_high = max(global_high, upper)

        # Single step for all params — derived from the widest range.
        step = self._derive_initial_step(global_low, global_high)
        self._char_step = step

        # n_startup_trials = number of params × levels per param.
        # This ensures random exploration covers all params before
        # TPE takes over. TPE's startup IS the coverage pass.
        n_params = len(self._param_names)
        n_startup = max(5, n_params * 3)

        self._char_study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup),
            storage=optuna.storages.InMemoryStorage(),
            study_name=f'char_{self.breeder_id}_r0',
        )

        logger.info(
            f"CHAR INIT: {n_params} params, step={step:.1f}, "
            f"startup={n_startup}, threshold={self.convergence_threshold}"
        )

    def _ask_next_probe(self) -> Optional[Dict[str, Any]]:
        """Ask the study for the next param + level to push.

        One trial samples which param (categorical) and what level
        (stepped float). The coordinator builds a push config from
        that — one param at the sampled level, rest at neutral.
        """
        if self._char_study is None:
            return None

        trial = self._char_study.ask()
        param_name = trial.suggest_categorical('push_param', self._param_names)
        bounds = self._param_bounds[param_name]

        level = trial.suggest_float(
            'push_level', bounds['lower'], bounds['upper'],
            step=self._char_step,
        )
        if bounds.get('is_int'):
            level = float(int(round(level)))

        self._char_trial = trial

        neutral = self._get_neutral_params()
        config = dict(neutral)
        if isinstance(config.get(param_name), list):
            config[param_name] = [level] * len(config[param_name])
        else:
            config[param_name] = level

        return {
            'param_name': param_name,
            'param_idx': bounds['idx'],
            'level': level,
            'config': config,
        }

    def _tell_char_study(self, param_name: str, delta: Optional[float]):
        """Tell the study the convergence delta for the last probe.

        Catches INFINITY (first point on a curve, no prior to compare)
        and replaces it with 1.0 — "maximally informative but not
        astronomically so." Prevents TPE from being poisoned by
        8.99e+307 values.

        After telling, checks if all discrete combinations are exhausted
        without convergence — if so, refine (new study at halved step).
        """
        from optuna.trial import TrialState

        if self._char_trial is None:
            return

        trial = self._char_trial
        self._char_trial = None

        if delta is None:
            self._char_study.tell(trial.number, state=TrialState.FAIL)
            logger.info(f"CHAR TELL: trial={trial.number} FAIL (causal unavailable)")
            return

        # Catch INFINITY-replacement from causal (f64::MAX/2 ≈ 8.99e+307)
        # and absurd values. Replace with 1.0 — informative enough for
        # TPE to want to revisit, but not so large it dominates.
        if delta > 1e10:
            logger.info(
                f"CHAR TELL: catching INFINITY delta={delta:.2e} "
                f"→ replacing with 1.0 (first point, no prior)"
            )
            delta = 1.0

        self._char_study.tell(trial.number, float(delta))
        logger.info(
            f"CHAR TELL: {param_name} level trial={trial.number} "
            f"delta={delta:.6f}"
        )

        # Check exhaustion: all (param × level) combinations visited?
        if len(self._converged_params) < len(self._param_names):
            n_complete = sum(
                1 for t in self._char_study.trials
                if t.state == TrialState.COMPLETE
            )
            # Total discrete space = sum of levels across all params
            total_space = sum(
                self._count_param_levels(name)
                for name in self._param_names
            )
            logger.info(
                f"CHAR PROGRESS: {n_complete}/{total_space} "
                f"combinations explored, "
                f"{len(self._converged_params)}/{len(self._param_names)} "
                f"params converged"
            )
            if n_complete >= total_space:
                logger.info(
                    f"CHAR EXHAUSTED: {n_complete}/{total_space} → refining"
                )
                self._refine_study()

    def _count_param_levels(self, param_name: str) -> int:
        """Number of discrete levels for a param at current step."""
        bounds = self._param_bounds.get(param_name)
        if not bounds:
            return 0
        if self._char_step <= 0:
            return 0
        return int(round((bounds['upper'] - bounds['lower']) / self._char_step)) + 1

    def _refine_study(self):
        """Close current study, create new one at halved step.

        Optuna drops params with inconsistent distributions when step
        changes mid-study. Disposable studies are the workaround.
        Curves persist in causal's CurveRegistry.
        """
        import optuna

        if self._refinement_level >= self.refinement_depth:
            logger.info(
                f"REFINEMENT: exhausted (depth={self.refinement_depth}) "
                f"— accepting best effort"
            )
            self._converged_params.update(self._param_names)
            return

        old_step = self._char_step
        new_step = old_step / 2.0
        self._char_step = new_step
        self._refinement_level += 1

        n_params = len(self._param_names)
        n_startup = max(5, n_params * 3)

        self._char_study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup),
            storage=optuna.storages.InMemoryStorage(),
            study_name=f'char_{self.breeder_id}_r{self._refinement_level}',
        )

        logger.info(
            f"REFINEMENT: study r{self._refinement_level - 1}→"
            f"r{self._refinement_level} step={old_step:.2f}→{new_step:.2f}"
        )

    # ─── Step Derivation ─────────────────────────────────────────────

    def _derive_initial_step(self, lower: float, upper: float,
                             is_int: bool = False) -> float:
        """Derive initial step size for stepped trial sampling.

        Starts coarse (range / 4) to keep the initial probe count small.
        The metaheuristic sampler within the characterization study
        picks which levels to visit; refinement halves the step where
        the curve is still moving.

        For integer params: snap to an integer step that divides the
        range into even segments and reaches the upper bound. We try
        divisors of the range starting from the raw /4 estimate down
        to 1, picking the largest that divides cleanly.
        """
        param_range = upper - lower
        if param_range <= 0:
            return param_range

        # Start coarse: 4 segments across the range.
        step = param_range / 4.0

        if is_int:
            # Snap to integer step that divides range evenly.
            # Search outward from the ideal /4 estimate to find
            # the closest divisor that reaches the upper bound.
            int_range = int(round(param_range))
            raw = max(1, int(round(step)))
            best = None
            for delta in range(0, max(raw, int_range)):
                for candidate in (raw - delta, raw + delta):
                    if candidate < 1:
                        continue
                    if int_range % candidate == 0:
                        if best is None or abs(candidate - raw) < abs(best - raw):
                            best = candidate
                if best is not None and delta > 0:
                    break  # found closest, no need to search further
            step = float(best) if best else 1.0

        logger.info(
            f"DERIVED STEP: range=[{lower}, {upper}] "
            f"threshold={self.convergence_threshold} "
            f"initial_step={step:.1f} is_int={is_int}"
        )
        return step

    # ─── Main State Machine ──────────────────────────────────────────

    def decide_trial(self, trial) -> Dict[str, Any]:
        if not self._initialized:
            self._ensure_tables()
            self._cleanup_stale_state()
            self._initialized = True

        logger.info(f"COORD trial={trial.number} state={self.state} breeder={self.breeder_id[:8]}")

        # Heartbeat for sender states
        if self.state in (self.PROBE_PUSH, self.PROBE_PAUSE):
            if not self._heartbeat():
                logger.warning("Lost sender lease — returning to OPTIMIZE")
                self.state = self.OPTIMIZE
                return {'mode': 'optimize', 'params': None, 'detection_trial': False}

        if self.state == self.OPTIMIZE:
            return self._handle_optimize(trial)
        if self.state == self.PROBE_PUSH:
            return self._handle_probe_push(trial)
        if self.state == self.PROBE_PAUSE:
            return self._handle_probe_pause(trial)
        if self.state == self.DONE:
            return self._handle_done(trial)
        if self.state == self.COOLDOWN:
            return self._handle_cooldown(trial)
        if self.state == self.HOLD:
            return self._handle_hold(trial)

        logger.warning(f"Unknown state {self.state} — resetting to OPTIMIZE")
        self.state = self.OPTIMIZE
        return {'mode': 'optimize', 'params': None, 'detection_trial': False}

    # ── OPTIMIZE ─────────────────────────────────────────────────────

    def _handle_optimize(self, trial) -> Dict[str, Any]:
        self._optimize_count += 1

        if self._optimize_count < self.min_optimize_trials:
            return self._optimize_result()

        if self._count_active_breeders() < 2:
            return self._optimize_result()

        # Initialize characterization study on first entry
        if self._char_study is None:
            self._init_characterization()
            if self._char_study is None:
                return self._optimize_result()

        # Try to become sender
        if self._try_acquire_lease(self.PROBE_PUSH):
            logger.info("Acquired lease — starting characterization")
            self.state = self.PROBE_PUSH
            self._push_count = 0
            self._neutral_params = None  # force recompute
            return self._handle_probe_push(trial)

        # Someone else is sender — become receiver
        if self._has_active_sender():
            self.state = self.HOLD
            self._hold_count = 0
            return self._handle_hold(trial)

        return self._optimize_result()

    def _optimize_result(self) -> Dict[str, Any]:
        return {'mode': 'optimize', 'params': None, 'detection_trial': False}

    # ── PROBE_PUSH (sender) ──────────────────────────────────────────

    def _handle_probe_push(self, trial) -> Dict[str, Any]:
        # First trial of a new push block: ask char study for next level
        if self._push_count == 0:
            probe = self._ask_next_probe()
            if probe is None:
                self.state = self.DONE
                return self._handle_done(trial)
            self._current_probe = probe

            self._set_lease_phase(self.PROBE_PUSH, push_budget=self.push_block_size)
            self._round_push_start = datetime.now()
            logger.info(
                f"PROBE_PUSH: param={probe['param_name']} "
                f"level={probe['level']}"
            )
        else:
            self._decrement_budget(self.PROBE_PUSH, is_push=True)

        self._push_count += 1
        if self._push_count >= self.push_block_size:
            self.state = self.PROBE_PAUSE
            self._pause_count = 0

        probe = self._current_probe
        return {
            'mode': 'impulse',
            'params': dict(probe['config']),
            'impulse_phase': 'probe_push',
            'probe_param': probe['param_name'],
            'probe_param_idx': probe.get('param_idx'),
            'probe_level': probe['level'],
            'detection_trial': True,
        }

    # ── PROBE_PAUSE (sender) ─────────────────────────────────────────

    def _handle_probe_pause(self, trial) -> Dict[str, Any]:
        probe = self._current_probe

        if self._pause_count == 0:
            self._set_lease_phase(self.PROBE_PAUSE, pause_budget=self.pause_block_size)
        else:
            self._decrement_budget(self.PROBE_PAUSE, is_push=False)

        params = self._get_neutral_params()
        if not params:
            self.state = self.DONE
            return self._handle_done(trial)

        self._pause_count += 1
        if self._pause_count >= self.pause_block_size:
            self._round_pause_end = datetime.now()
            logger.info(
                f"PROBE_PAUSE: done for param={probe['param_name']} "
                f"level={probe['level']}"
            )

            # Close the loop: causal computes delta, tell char study.
            delta = self._process_probe_result(probe)
            self._tell_char_study(probe['param_name'], delta)

            self._push_count = 0
            self._pause_count = 0
            self.state = self.PROBE_PUSH

        return {
            'mode': 'hold',
            'params': dict(params),
            'impulse_phase': 'probe_pause',
            'probe_param': probe['param_name'],
            'probe_param_idx': probe.get('param_idx'),
            'probe_level': probe['level'],
            'detection_trial': True,
        }

    # ── DONE (sender) ────────────────────────────────────────────────

    def _handle_done(self, trial) -> Dict[str, Any]:
        self._release_lease()
        self.state = self.COOLDOWN
        self._cooldown_count = 0

        n_converged = len(self._converged_params)
        n_total = len(self._param_names)
        logger.info(
            f"DONE: released lease. "
            f"Converged {n_converged}/{n_total} params."
        )
        return {'mode': 'optimize', 'params': None, 'detection_trial': False}

    # ── COOLDOWN ─────────────────────────────────────────────────────

    def _handle_cooldown(self, trial) -> Dict[str, Any]:
        self._cooldown_count += 1
        if self._cooldown_count >= self.cooldown_trials:
            non_converged = [
                p for p in self._param_names
                if p not in self._converged_params
            ]
            if non_converged:
                logger.info("COOLDOWN: done — re-acquiring for more probes")
                if self._try_acquire_lease(self.PROBE_PUSH):
                    self.state = self.PROBE_PUSH
                    self._push_count = 0
                    return self._handle_probe_push(trial)
            logger.info("COOLDOWN: done — back to OPTIMIZE")
            self.state = self.OPTIMIZE
        return {'mode': 'optimize', 'params': None, 'detection_trial': False}

    # ── HOLD (receiver) ──────────────────────────────────────────────

    def _handle_hold(self, trial) -> Dict[str, Any]:
        if not self._has_active_sender():
            logger.info("HOLD: sender finished — back to OPTIMIZE")
            self.state = self.OPTIMIZE
            return {'mode': 'optimize', 'params': None, 'detection_trial': False}

        self._hold_count += 1
        if self._hold_count > self.MAX_HOLD_TRIALS:
            logger.warning(
                f"HOLD: hit MAX_HOLD_TRIALS ({self.MAX_HOLD_TRIALS}) — "
                f"returning to OPTIMIZE"
            )
            self.state = self.OPTIMIZE
            return {'mode': 'optimize', 'params': None, 'detection_trial': False}

        params = self._get_neutral_params()
        if not params:
            self.state = self.OPTIMIZE
            return {'mode': 'optimize', 'params': None, 'detection_trial': False}

        phase = self._get_lease_phase()
        if self._hold_count % 5 == 1 or phase != getattr(self, '_last_observed_phase', None):
            logger.info(
                f"HOLD: receiver trial {self._hold_count}, "
                f"sender phase={phase}"
            )
            self._last_observed_phase = phase

        return {
            'mode': 'hold',
            'params': dict(params),
            'impulse_phase': None,
            'lease_phase': phase,
            'detection_trial': True,
        }

    # ─── Observability ───────────────────────────────────────────────

    def get_state(self) -> str:
        return self.state

    def get_char_status(self) -> Dict[str, Any]:
        """Structured characterization status for trial attrs / dashboards.

        Exposes study internals: which params converged, refinement
        levels, trials completed per study, current step sizes.
        """
        from optuna.trial import TrialState

        status = {}
        if self._char_study is not None:
            for name in self._param_names:
                n_levels = self._count_param_levels(name)
                status[name] = {
                    'converged': name in self._converged_params,
                    'step': self._char_step,
                    'levels_total': n_levels,
                }

            n_complete = sum(
                1 for t in self._char_study.trials
                if t.state == TrialState.COMPLETE
            )
            total_space = sum(
                self._count_param_levels(n) for n in self._param_names
            )
        else:
            n_complete = 0
            total_space = 0

        return {
            'state': self.state,
            'refinement_level': self._refinement_level,
            'step': self._char_step,
            'params': status,
            'combinations_explored': n_complete,
            'combinations_total': total_space,
            'converged_count': len(self._converged_params),
            'params_total': len(self._param_names),
        }
