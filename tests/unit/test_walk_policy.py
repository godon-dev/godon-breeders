"""WalkPolicy unit tests — the stateless compass over causal's notebook.

Every test drives the policy through a fake transport serving explicit
notebook views; no HTTP, no causal process. The point: the walk is a pure
function of the view, so invocation boundaries cannot lose it.
"""

import pytest

from engine.walk_policy import WalkPolicy


SCEN = "scenario-junction-gate"  # naming aid only; no scenario IO here


def make_view(param="param_0", refinement_level=0, curves=None):
    return {
        "sender_id": "B4",
        "param": param,
        "refinement_level": refinement_level,
        "curves": curves or [],
    }


def curve(recv, ch="objective_0", levels=(), gaps=(), converged=False):
    return {
        "receiver_id": recv,
        "channel": ch,
        "converged": converged,
        "levels": list(levels),
        "gaps": list(gaps),
    }


def gap(from_level, to_level, jump, bars_sum, unresolved, ignorance):
    return {
        "from_level": from_level,
        "to_level": to_level,
        "jump": jump,
        "bars_sum": bars_sum,
        "unresolved": unresolved,
        "ignorance": ignorance,
    }


class FakeTransport:
    """Serves a mutable view; records refine calls."""

    def __init__(self, views_by_param=None):
        self.views = views_by_param or {}
        self.refines = []

    def __call__(self, method, url, payload=None):
        if method == "GET":
            for param, view in self.views.items():
                if f"param={param}" in url:
                    return view
            return make_view()
        if method == "POST":
            self.refines.append(payload)
            return {"refinement_level": len(self.refines)}
        raise AssertionError(f"unexpected method {method}")


def make_policy(transport, bounds=None, depth=3):
    bounds = bounds or {"param_0": (0.0, 100.0, False),
                        "param_1": (0.0, 100.0, False)}
    return WalkPolicy(
        causal_url="http://causal:8091",
        group_id="bench-characterization",
        breeder_id="B4",
        refinement_depth=depth,
        param_bounds=bounds,
        transport=transport,
    )


class TestPrefixContract:
    def test_empty_notebook_opens_at_center_then_upper_then_lower(self):
        t = FakeTransport({"param_0": make_view("param_0")})
        p = make_policy(t, bounds={"param_0": (0.0, 100.0, False)})
        assert p.next_probe(set()) == ("param_0", 50.0)
        t.views["param_0"] = make_view("param_0", curves=[
            curve("R1", levels=[50.0])])
        assert p.next_probe(set()) == ("param_0", 100.0)
        t.views["param_0"] = make_view("param_0", curves=[
            curve("R1", levels=[50.0, 100.0])])
        assert p.next_probe(set()) == ("param_0", 0.0)

    def test_prefix_derived_from_banked_levels_not_ram(self):
        # The notebook already carries the anchors: no prefix re-walk.
        t = FakeTransport({"param_0": make_view("param_0", curves=[
            curve("R1", levels=[0.0, 50.0, 100.0]),
            curve("SELF", levels=[0.0, 50.0, 100.0]),
        ])})
        p = make_policy(t, bounds={"param_0": (0.0, 100.0, False)})
        # With anchors banked and NO unresolved gaps, the param is done.
        assert p.next_probe(set()) is None


class TestCompass:
    def test_steers_to_fattest_unresolved_bracket(self):
        t = FakeTransport({"param_0": make_view("param_0", curves=[
            curve("R1", levels=[0.0, 50.0, 100.0], gaps=[
                gap(0.0, 50.0, 0.1, 0.04, True, 0.05),
                gap(50.0, 100.0, 1.0, 0.04, True, 0.25),
            ]),
            curve("SELF", levels=[0.0, 50.0, 100.0], gaps=[
                gap(0.0, 50.0, 0.006, 0.037, False, 0.003),
                gap(50.0, 100.0, 1.004, 0.041, True, 0.502),
            ]),
        ])})
        p = make_policy(t, bounds={"param_0": (0.0, 100.0, False)})
        # The fattest price lives on the SELF curve's (50,100) bracket —
        # the seed-48 defect: steering must see it.
        assert p.next_probe(set()) == ("param_0", 75.0)

    def test_ties_wider_then_center_then_higher(self):
        # Equal ignorance, equal width -> closer to center wins.
        t = FakeTransport({"param_0": make_view("param_0", curves=[
            curve("R1", levels=[0.0, 25.0, 50.0, 75.0, 100.0], gaps=[
                gap(0.0, 25.0, 0.5, 0.04, True, 0.125),
                gap(25.0, 50.0, 0.5, 0.04, True, 0.125),
                gap(50.0, 75.0, 0.5, 0.04, True, 0.125),
                gap(75.0, 100.0, 0.5, 0.04, True, 0.125),
            ]),
        ])})
        p = make_policy(t, bounds={"param_0": (0.0, 100.0, False)})
        # Same price, same width. |mid-50|: 37.5, 12.5, 12.5, 37.5 -> the
        # center-tie is between 37.5 and 62.5; final tie-break: higher -> 62.5.
        assert p.next_probe(set()) == ("param_0", 62.5)

    def test_priced_out_returns_none(self):
        t = FakeTransport({"param_0": make_view("param_0", curves=[
            curve("R1", levels=[0.0, 50.0, 100.0], gaps=[
                gap(0.0, 50.0, 0.006, 0.037, False, 0.003),
                gap(50.0, 100.0, 0.001, 0.044, False, 0.0006),
            ]),
        ])})
        p = make_policy(t, bounds={"param_0": (0.0, 100.0, False)})
        assert p.next_probe(set()) is None

    def test_resolved_brackets_cannot_steer(self):
        # Phantom-progress guard: re-blends tightened the bars; the only
        # unresolved brackets are priced out. Nothing to steer to.
        t = FakeTransport({"param_0": make_view("param_0", curves=[
            curve("R1", levels=[0.0, 25.0, 50.0, 75.0, 100.0], gaps=[
                gap(0.0, 25.0, 0.3, 0.9, False, 0.075),
                gap(25.0, 50.0, 0.2, 0.8, False, 0.05),
            ]),
        ])})
        p = make_policy(t, bounds={"param_0": (0.0, 100.0, False)})
        assert p.next_probe(set()) is None


class TestRota:
    def test_skip_and_round_robin(self):
        t = FakeTransport({
            "param_0": make_view("param_0", curves=[
                curve("R1", levels=[0.0, 50.0, 100.0], gaps=[
                    gap(0.0, 50.0, 0.1, 0.04, True, 0.05)])]),
            "param_1": make_view("param_1", curves=[
                curve("R1", levels=[0.0, 50.0, 100.0], gaps=[
                    gap(50.0, 100.0, 0.1, 0.04, True, 0.05)])]),
        })
        p = make_policy(t)
        first = p.next_probe(set())
        second = p.next_probe(set())
        assert first[0] == "param_0" and second[0] == "param_1"

    def test_skipped_param_returns_none_only_when_all_skipped(self):
        t = FakeTransport({"param_0": make_view("param_0"), "param_1": make_view("param_1")})
        p = make_policy(t)
        assert p.next_probe({"param_0", "param_1"}) is None


class TestNotebook:
    def test_refine_posts_and_status_reflects_depth(self):
        t = FakeTransport({"param_0": make_view("param_0", refinement_level=1)})
        p = make_policy(t, bounds={"param_0": (0.0, 100.0, False)})
        p.refine("param_0")
        assert t.refines == [{"group_id": "bench-characterization",
                              "sender_id": "B4", "probe_param": "param_0"}]
        st = p.status()["param_0"]
        assert st["step"] == pytest.approx(100.0 / 2 ** (1 + 2))

    def test_resume_exactness_matches_uninterrupted_choice(self):
        # The core property: a walker that died and restarted from the
        # notebook picks the SAME next level as one that never stopped.
        levels_after_two = [50.0, 100.0, 0.0, 75.0]
        t_restarted = FakeTransport({"param_0": make_view("param_0", curves=[
            curve("R1", levels=levels_after_two, gaps=[
                gap(0.0, 50.0, 0.9, 0.04, True, 0.45),
                gap(50.0, 75.0, 0.02, 0.05, False, 0.005),
                gap(75.0, 100.0, 0.01, 0.06, False, 0.0025),
            ])])})
        p = make_policy(t_restarted, bounds={"param_0": (0.0, 100.0, False)})
        assert p.next_probe(set()) == ("param_0", 25.0)
