#!/usr/bin/env python3
"""
Post-hoc characterization analysis.

Reads trial data from the breeder DB, reconstructs response curves from
tagged probe trials, and validates characterization quality.

Usage:
  python3 analyze_characterization.py <sender_db> <receiver_db>

Both args are breeder DB names (e.g. breeder_<uuid>).

Output:
  - Per-param response curves (level → receiver shift)
  - Convergence deltas
  - Shape classification (linear / threshold / saturation / flat)
  - Detection: which param(s) carry coupling
"""

import sys
import os
import json
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from f.breeder.engine.characterization import ResponseCurve


def fetch_trials(db_name):
    """Fetch all COMPLETE trials with user_attrs from the breeder DB."""
    import psycopg2
    user = os.environ.get('GODON_ARCHIVE_DB_USER', 'yugabyte')
    pw = os.environ.get('GODON_ARCHIVE_DB_PASSWORD', 'yugabyte')
    host = os.environ.get('GODON_ARCHIVE_DB_SERVICE_HOST', 'localhost')
    port = os.environ.get('GODON_ARCHIVE_DB_SERVICE_PORT', '5433')

    conn = psycopg2.connect(
        f"host={host} port={port} user={user} password={pw} dbname={db_name}"
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT t.number, t.state,
               ta.key, ta.value_json
        FROM trials t
        LEFT JOIN trial_user_attributes ta ON t.number = ta.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY t.number, ta.key
    """)

    trials = defaultdict(lambda: {'attrs': {}, 'values': []})
    for trial_num, state, key, value_json in cur.fetchall():
        trials[trial_num]['state'] = state
        if key:
            try:
                trials[trial_num]['attrs'][key] = json.loads(value_json)
            except (json.JSONDecodeError, TypeError):
                trials[trial_num]['attrs'][key] = value_json

    # Also fetch trial values (objectives)
    cur.execute("""
        SELECT tv.trial_id, tv.objective_id, tv.value
        FROM trial_values tv
        ORDER BY tv.trial_id, tv.objective_id
    """)
    for trial_id, obj_id, value in cur.fetchall():
        trials[trial_id]['values'].append(value)

    cur.close()
    conn.close()

    return dict(trials)


def analyze(sender_db, receiver_db):
    """Main analysis: build response curves from tagged trials."""

    print(f"=== Characterization Analysis ===")
    print(f"Sender:   {sender_db}")
    print(f"Receiver: {receiver_db}")
    print()

    sender_trials = fetch_trials(sender_db)
    receiver_trials = fetch_trials(receiver_db)

    # Find probe push/pause pairs from sender
    # Group receiver trials by trial number to align timestamps
    # For each sender probe (push block + pause block), measure receiver shift

    # Collect sender probe phases
    sender_probes = []
    for trial_num in sorted(sender_trials.keys()):
        t = sender_trials[trial_num]
        attrs = t.get('attrs', {})
        phase = attrs.get('impulse_phase', '')

        if phase in ('characterize_push', 'characterize_pause'):
            sender_probes.append({
                'trial_num': trial_num,
                'phase': phase,
                'probe_param': attrs.get('probe_param'),
                'probe_param_idx': attrs.get('probe_param_idx'),
                'probe_level': attrs.get('probe_level'),
                'sender_obj': t['values'][0] if t['values'] else None,
            })

    # Collect receiver HOLD trials (aligned by trial number — not exact,
    # but receiver holds during sender's push/pause)
    receiver_hold_trials = []
    for trial_num in sorted(receiver_trials.keys()):
        t = receiver_trials[trial_num]
        attrs = t.get('attrs', {})
        if attrs.get('detection_mode') == 'hold':
            receiver_hold_trials.append({
                'trial_num': trial_num,
                'receiver_obj0': t['values'][0] if t['values'] else None,
                'receiver_obj1': t['values'][1] if len(t['values']) > 1 else None,
                'lease_phase': attrs.get('lease_phase'),
            })

    print(f"Sender probe trials: {len(sender_probes)}")
    print(f"Receiver hold trials: {len(receiver_hold_trials)}")
    print()

    # Group sender probes by (param, level)
    probes_by_param_level = defaultdict(lambda: {'push_trials': [], 'pause_trials': []})
    for p in sender_probes:
        key = (p['probe_param'], p['probe_level'])
        if p['phase'] == 'characterize_push':
            probes_by_param_level[key]['push_trials'].append(p)
        elif p['phase'] == 'characterize_pause':
            probes_by_param_level[key]['pause_trials'].append(p)

    # For each (param, level), find the receiver trials that overlap
    # and compute median receiver objective during push vs pause
    # This is simplified — proper alignment needs timestamp matching.
    # For the bench (single shared simulator), trial numbers roughly align.

    curves_by_param = defaultdict(list)

    for (param, level), groups in sorted(probes_by_param_level.items()):
        push_trials = groups['push_trials']
        pause_trials = groups['pause_trials']

        if not push_trials or not pause_trials:
            continue

        # Find receiver trials near the sender push/pause trial numbers
        push_nums = [t['trial_num'] for t in push_trials]
        pause_nums = [t['trial_num'] for t in pause_trials]

        # Receiver trials in the push window
        push_receiver = [
            r['receiver_obj0'] for r in receiver_hold_trials
            if r['receiver_obj0'] is not None
            and min(push_nums) <= r['trial_num'] <= max(push_nums)
        ]
        pause_receiver = [
            r['receiver_obj0'] for r in receiver_hold_trials
            if r['receiver_obj0'] is not None
            and min(pause_nums) <= r['trial_num'] <= max(pause_nums)
        ]

        if not push_receiver or not pause_receiver:
            continue

        push_median = statistics.median(push_receiver)
        pause_median = statistics.median(pause_receiver)
        shift = push_median - pause_median

        curves_by_param[param].append((level, shift))

        print(f"  {param} @ {level:>6.1f}: push={push_median:.4f} "
              f"pause={pause_median:.4f} shift={shift:+.4f} "
              f"(push_n={len(push_receiver)}, pause_n={len(pause_receiver)})")

    # Build ResponseCurves per param
    print(f"\n{'='*60}")
    print("RESPONSE CURVES")
    print(f"{'='*60}")

    for param in sorted(curves_by_param.keys()):
        points = sorted(curves_by_param[param], key=lambda p: p[0])
        print(f"\n--- {param} ({len(points)} points) ---")

        curve = ResponseCurve(convergence_threshold=0.01, min_points=2)
        for level, shift in points:
            curve.add_point(level, shift)

        summary = curve.summary()
        print(f"  Converged: {summary.get('converged', False)}")
        print(f"  Last delta: {summary.get('last_delta', 'N/A'):.6f}")
        print(f"  Max slope: {summary.get('max_slope', 0):.6f}")
        print(f"  Response range: {summary.get('response_range', [0, 0])}")

        # Shape classification
        if len(points) < 2:
            print(f"  Shape: insufficient data")
        elif all(abs(p[1]) < 0.02 for p in points):
            print(f"  Shape: FLAT (no coupling on this param)")
        elif summary.get('max_slope', 0) > 0.05:
            print(f"  Shape: THRESHOLD/JUMP (steep segment)")
        else:
            print(f"  Shape: GRADUAL (linear or saturation)")

    # Detection summary
    print(f"\n{'='*60}")
    print("DETECTION (which params carry coupling)")
    print(f"{'='*60}")
    for param in sorted(curves_by_param.keys()):
        points = curves_by_param[param]
        max_shift = max(abs(p[1]) for p in points) if points else 0
        carrier = "CARRIER" if max_shift > 0.05 else "dead"
        print(f"  {param}: max_shift={max_shift:.4f} → {carrier}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <sender_db> <receiver_db>")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2])
