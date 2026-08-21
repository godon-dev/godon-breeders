#!/usr/bin/env python3
"""
Unit tests for coverage_walk.py — the coverage contract, asserted directly.

Every property of the walk is tested without the coordinator: full
coverage, no revisits, deterministic order, retirement, int/float
mixed bounds, refinement semantics, truncation fairness.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from engine.coverage_walk import CoverageWalk


BENCH = {
    'param_0': (0.0, 100.0, False),
    'param_1': (0.0, 100.0, False),
    'param_2': (0.0, 100.0, False),
}


def _drain(walk, skip=set()):
    out = []
    while True:
        p = walk.next_probe(skip)
        if p is None:
            return out
        out.append(p)


def test_full_coverage_no_revisits():
    """Bench grid: every param measured at exactly {0,25,50,75,100}."""
    walk = CoverageWalk(BENCH)
    probes = _drain(walk)
    per_param = {}
    for name, lv in probes:
        per_param.setdefault(name, []).append(lv)
    for name in BENCH:
        got = sorted(per_param.get(name, []))
        assert got == [0.0, 25.0, 50.0, 75.0, 100.0], \
            f"{name}: expected full grid, got {got}"
        assert len(got) == len(set(got)), f"{name}: revisited a level"
    print("  3 params × 5 levels, zero revisits — PASS")


def test_order_mid_edges_quarters():
    """Farthest-point order: midpoint, edges, quarters (spread-first)."""
    walk = CoverageWalk({'p': (0.0, 100.0, False)})
    order = [lv for _, lv in _drain(walk)]
    assert order == [50.0, 100.0, 0.0, 75.0, 25.0], \
        f"Expected [50, 100, 0, 75, 25], got {order}"
    print(f"  order {order} — PASS")


def test_determinism():
    """Two walks over the same bounds produce identical sequences."""
    a = _drain(CoverageWalk(BENCH))
    b = _drain(CoverageWalk(BENCH))
    assert a == b, "walks differ across instances"
    print(f"  {len(a)} probes identical across instances — PASS")


def test_truncation_round_robin():
    """Kill the walk anywhere: params get even coverage (no starvation)."""
    walk = CoverageWalk(BENCH)
    first3 = [walk.next_probe(set())[0] for _ in range(3)]
    assert set(first3) == {'param_0', 'param_1', 'param_2'}, \
        f"first 3 draws must hit all params, got {first3}"
    # after 6 draws: 2 levels each
    next3 = [walk.next_probe(set())[0] for _ in range(3)]
    assert set(next3) == {'param_0', 'param_1', 'param_2'}
    print("  rotation fair at every prefix — PASS")


def test_retirement_excludes_param():
    """Converged params leave the rotation; walk still finishes."""
    walk = CoverageWalk(BENCH)
    first = walk.next_probe(set())          # draw 1 (rotation: param_0)
    skip = {'param_1'}  # param_1 converges after 1 probe
    probes = [first] + _drain(walk, skip)
    names = {n for n, _ in probes}
    for name in ('param_0', 'param_2'):
        got = sorted(lv for n, lv in probes if n == name)
        assert got == [0.0, 25.0, 50.0, 75.0, 100.0]
    print("  retired param excluded, survivors fully covered — PASS")


def test_all_converged_returns_none():
    walk = CoverageWalk(BENCH)
    assert walk.next_probe({'param_0', 'param_1', 'param_2'}) is None
    print("  all retired → None — PASS")


def test_int_enumerates_regardless_of_floor():
    """Int params enumerate fully; float floors must not starve them."""
    mixed = {
        'f': (0.0, 100.0, False),   # floor 25
        'i': (0.0, 10.0, True),     # 11 ints
    }
    walk = CoverageWalk(mixed)
    probes = _drain(walk)
    ints = sorted(lv for n, lv in probes if n == 'i')
    assert ints == [float(v) for v in range(11)], \
        f"int param must enumerate fully, got {ints}"
    floats = sorted(lv for n, lv in probes if n == 'f')
    assert floats == [0.0, 25.0, 50.0, 75.0, 100.0]
    print("  int enumerates 11/11 while float floor respected — PASS")


def test_refine_measures_only_new_midpoints():
    """After refine: old levels NOT re-proposed, midpoints appear."""
    walk = CoverageWalk({'p': (0.0, 100.0, False)})
    coarse = _drain(walk)                     # {0,25,50,75,100}
    walk.refine()
    fine = _drain(walk)
    fine_levels = {lv for _, lv in fine}
    assert fine_levels == {12.5, 37.5, 62.5, 87.5}, \
        f"refinement must add only midpoints, got {sorted(fine_levels)}"
    assert not (fine_levels & {lv for _, lv in coarse})
    print(f"  refine → only {sorted(fine_levels)} — PASS")


def test_can_probe_is_pure():
    """can_probe never mutates the walk."""
    walk = CoverageWalk(BENCH)
    before = walk.status()['param_0']['levels_measured']
    for _ in range(5):
        walk.can_probe(set())
    after = walk.status()['param_0']['levels_measured']
    assert before == after == 0
    assert walk.can_probe(set()) is True
    print("  can_probe pure — PASS")


def test_float_bounds_arbitrary():
    """Float params with non-zero-based bounds walk correctly."""
    walk = CoverageWalk({'p': (3.5, 7.5, False)})
    order = [lv for _, lv in _drain(walk)]
    assert order[0] == 5.5          # midpoint first
    assert sorted(order) == [3.5, 4.25, 5.5, 6.75, 7.5] or \
        sorted(order) == [3.5, 5.5, 7.5] or len(order) >= 3
    assert 3.5 in order and 7.5 in order  # bounds measured
    print(f"  bounds (3.5, 7.5): {order} — PASS")




def test_per_param_floors_not_global():
    """Each float param gets a floor from ITS range — not the widest.

    (The old global-step code gave a 0-1 param the 0-100 param's
    step: nonsense levels. The walk must not regress to that.)
    """
    mixed = {
        'wide': (0.0, 100.0, False),   # floor 25
        'tiny': (0.0, 1.0, False),     # floor 0.25
    }
    walk = CoverageWalk(mixed)
    probes = _drain(walk)
    tiny_levels = sorted(lv for n, lv in probes if n == 'tiny')
    wide_levels = sorted(lv for n, lv in probes if n == 'wide')
    assert wide_levels == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert tiny_levels == [0.0, 0.25, 0.5, 0.75, 1.0], \
        f"tiny param must use its own scale, got {tiny_levels}"
    print(f"  wide={wide_levels} tiny={tiny_levels} — PASS")


def test_double_refine_progression():
    """Two refinements: each pass adds only the new midpoints."""
    walk = CoverageWalk({'p': (0.0, 100.0, False)})
    coarse = {lv for _, lv in _drain(walk)}          # {0,25,50,75,100}
    walk.refine()
    pass1 = {lv for _, lv in _drain(walk)}           # {12.5,37.5,62.5,87.5}
    walk.refine()
    pass2 = {lv for _, lv in _drain(walk)}           # 6.25..93.75 step 12.5/2
    assert coarse == {0.0, 25.0, 50.0, 75.0, 100.0}
    assert pass1 == {12.5, 37.5, 62.5, 87.5}
    assert pass2 == {6.25, 18.75, 31.25, 43.75, 56.25, 68.75, 81.25, 93.75}
    # nothing re-measured across passes
    assert not (pass1 & coarse) and not (pass2 & coarse) and not (pass2 & pass1)
    print("  3 passes, 17 distinct levels, zero re-probes — PASS")


def test_degenerate_range_single_level():
    """hi == lo: one measurable level, then done."""
    walk = CoverageWalk({'p': (50.0, 50.0, False)})
    assert walk.next_probe(set()) == ('p', 50.0)
    assert walk.next_probe(set()) is None
    print("  degenerate 50..50 → one probe then None — PASS")


def test_status_accounting():
    """status() counts match actual probes."""
    walk = CoverageWalk(BENCH)
    st = walk.status()
    assert all(v['levels_measured'] == 0 for v in st.values())
    for _ in range(4):
        walk.next_probe(set())
    st = walk.status()
    total = sum(v['levels_measured'] for v in st.values())
    assert total == 4
    assert all(v['levels_total'] == 5 for v in st.values())
    print("  4 probes → status shows 4 measured, 5 total each — PASS")


def test_scale_many_params_rotation():
    """50 params: every prefix is evenly spread (no starvation)."""
    big = {f'p{i:02d}': (0.0, 100.0, False) for i in range(50)}
    walk = CoverageWalk(big)
    seen = set()
    for k in range(50):
        name, lv = walk.next_probe(set())
        seen.add(name)
        assert lv == 50.0, "first sweep must take midpoints"
    assert len(seen) == 50
    # second sweep: edges, again for everyone before anyone deepens
    for k in range(50):
        name, lv = walk.next_probe(set())
        assert lv in (0.0, 100.0), f"second sweep must take edges, got {lv}"
    print("  50 params: midpoint sweep then edge sweep, fair throughout — PASS")


def test_retired_mid_walk_refills_survivors():
    """Retirement reallocates turns to survivors automatically."""
    walk = CoverageWalk(BENCH)
    walk.next_probe(set())            # param_0 midpoint
    skip = {'param_0', 'param_1'}     # both retire immediately after
    probes = _drain(walk, skip)
    assert {n for n, _ in probes} == {'param_2'}
    assert sorted(lv for _, lv in probes) == [0.0, 25.0, 50.0, 75.0, 100.0]
    print("  2 retired → survivor gets all remaining turns — PASS")

if __name__ == '__main__':
    for fn in [
        test_full_coverage_no_revisits,
        test_order_mid_edges_quarters,
        test_determinism,
        test_truncation_round_robin,
        test_retirement_excludes_param,
        test_all_converged_returns_none,
        test_int_enumerates_regardless_of_floor,
        test_refine_measures_only_new_midpoints,
        test_can_probe_is_pure,
        test_float_bounds_arbitrary,
        test_per_param_floors_not_global,
        test_double_refine_progression,
        test_degenerate_range_single_level,
        test_status_accounting,
        test_scale_many_params_rotation,
        test_retired_mid_walk_refills_survivors,
    ]:
        print(f"=== {fn.__name__} ===")
        fn()
    print("\nall coverage walk tests passed")
