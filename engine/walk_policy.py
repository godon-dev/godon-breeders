"""Stateless walk policy over causal's notebook page.

The walk is a pure function of the view causal reports — no RAM plan,
nothing to lose at invocation boundaries (the seed-48 amnesia defect).
Each step: fetch the notebook page for the param; honor the anchor
prefix; otherwise steer to the midpoint of the highest-priced unresolved
bracket across ALL curves of the param (listeners + self), each price
judged against its own bars (jump x width vs the gap's own endpoint
fuzz). Nothing priced above its bar -> the param is done at this floor.

Deterministic tie-breaks for equal prices: wider bracket, then closer to
the range center, then higher level. All-equal prices degrade to a
geometric sweep — the safe fallback.
"""


import logging

logger = logging.getLogger(__name__)


class WalkPolicy:
    def __init__(self, causal_url, group_id, breeder_id, refinement_depth,
                 param_bounds, transport=None):
        self._url = causal_url.rstrip("/")
        self._group = group_id
        self._breeder = breeder_id
        self._depth = refinement_depth
        self._bounds = dict(param_bounds)
        self._transport = transport or self._urllib_transport
        self._rr = 0

    # ── notebook IO ──────────────────────────────────────────────────

    def view(self, param):
        from urllib.parse import urlencode
        qs = urlencode({"param": param, "group": self._group})
        return self._transport("GET", f"{self._url}/walk-view/{self._breeder}?{qs}")

    @staticmethod
    def _urllib_transport(method, url, payload=None):
        import json as _json
        import urllib.request
        # 2 s: causal RTT is ~ms; this guards a hung socket so a dead
        # causal degrades to the blind ladder within one init, not minutes.
        timeout = 2.0
        req = urllib.request.Request(
            url,
            data=_json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"} if payload is not None else {},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read())

    # ── the policy ───────────────────────────────────────────────────

    @staticmethod
    def _fattest_bracket_midpoint(view, center):
        best_key = None
        best_mid = None
        for curve in view.get("curves", []):
            for g in curve.get("gaps", []):
                if not g.get("unresolved"):
                    continue
                if g.get("ignorance", float("inf")) <= g.get("bars_sum", 0.0):
                    continue  # priced out: its own bars say nothing hides here
                mid = (g["from_level"] + g["to_level"]) / 2.0
                width = g["to_level"] - g["from_level"]
                key = (g["ignorance"], width, -abs(mid - center), mid)
                if best_key is None or key > best_key:
                    best_key, best_mid = key, mid
        return best_mid

    def _next_level(self, name):
        lo, hi, is_int = self._bounds[name]
        try:
            view = self.view(name)
        except Exception as e:
            logger.warning(
                "walk-view unreachable for %s (%s) — param reports done; "
                "the notebook is authoritative, blind-guessing is not",
                name, e)
            return None
        measured = set()
        for curve in view.get("curves", []):
            for lv in curve.get("levels", []):
                measured.add(round(lv, 9))

        def unbanked(level):
            return round(level, 9) not in measured

        # Prefix contract: anchors for pricing, derived from the notebook
        # (not from RAM) — a restarted walker never re-walks them.
        center = (lo + hi) / 2.0
        for anchor in ((lo + hi) / 2.0, hi, lo):
            if unbanked(anchor):
                return anchor

        # Compass: midpoint of the fattest unresolved bracket anywhere on
        # this param's curves (listeners + self).
        mid = self._fattest_bracket_midpoint(view, center)
        if mid is None:
            return None
        return int(round(mid)) if is_int else mid

    def next_probe(self, skip):
        """Next (param, level), or None when every eligible param is done.

        `skip` holds retired param names (the two-key verdict's output).
        """
        eligible = [n for n in self._bounds if n not in skip]
        if not eligible:
            return None
        for _ in range(len(eligible)):
            name = eligible[self._rr % len(eligible)]
            self._rr += 1
            level = self._next_level(name)
            if level is not None:
                return name, level
        return None

    def can_probe(self, skip):
        return self.next_probe(set(skip)) is not None

    def refine(self, param=None):
        """Descend a floor: one param, or every bounded param when None."""
        params = [param] if param is not None else list(self._bounds)
        for p in params:
            try:
                self._transport(
                    "POST",
                    f"{self._url}/walk-view/refine",
                    payload={"group_id": self._group, "sender_id": self._breeder,
                             "probe_param": p},
                )
            except Exception as e:
                logger.warning("walk-view refine failed for %s (%s)", p, e)

    def status(self):
        """Per-param walk state for the CHAR PROGRESS trail."""
        out = {}
        for name, (lo, hi, _is_int) in self._bounds.items():
            try:
                view = self.view(name)
            except Exception:
                continue
            levels = set()
            for curve in view.get("curves", []):
                for lv in curve.get("levels", []):
                    levels.add(round(lv, 9))
            level = view.get("refinement_level", 0)
            out[name] = {
                "step": (hi - lo) / float(2 ** (level + 2)),
                "levels_total": None,
                "levels_measured": len(levels),
                "levels": sorted(levels),
            }
        return out
