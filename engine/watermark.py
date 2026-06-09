import random
from typing import Dict, Any, Optional, List
from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


class Watermark:
    def __init__(self, params_config: Dict[str, Any]):
        self.params_config = params_config
        self._trial_count = 0

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError

    def trial_count(self) -> int:
        return self._trial_count

    def is_active_phase(self) -> bool:
        return True

    def _clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def _coerce(self, pname: str, value, is_int: bool):
        """Round float watermarks back to int for integer-range params."""
        if is_int and isinstance(value, float):
            return int(round(value))
        return value


class Impulse(Watermark):
    """Impulse probing for interference detection.

    On a sparse duty cycle, pushes all watermarked parameters to their
    extreme values (upper or lower bound).  Between impulses the sampler
    runs freely — no modulation at all.

    Detection is temporal, not spectral: the observer checks whether the
    receiver's objectives shift in a post-impulse window vs baseline
    (Mann-Whitney U).  This works on any channel — linear, nonlinear,
    non-stationary — because it only needs presence, not fidelity.
    """

    def __init__(
        self,
        params_config: Dict[str, Any],
        param_configs: List[Dict[str, Any]],
        duty_cycle: float = 0.02,
        direction: str = "random",
    ):
        """
        Args:
            params_config: full breeder params config (for range extraction)
            param_configs: list of {name, lower, upper} dicts — which params
                           to impulse
            duty_cycle: fraction of trials that are impulse trials (default 2%)
            direction: "upper", "lower", or "random" — which bound to push to.
                       "random" alternates between upper and lower.
        """
        super().__init__(params_config)
        self._param_ranges = self._extract_ranges(params_config)
        self.duty_cycle = duty_cycle
        self.direction = direction

        # Track which params are int-typed
        self._int_params: set = set()

        # The params we impulse — with their bounds
        self._impulse_params = []
        for pc in param_configs:
            name = pc["name"]
            lo, hi = pc["lower"], pc["upper"]
            is_int = isinstance(lo, int) and isinstance(hi, int)
            if is_int:
                self._int_params.add(name)
            self._impulse_params.append({
                "name": name,
                "lower": lo,
                "upper": hi,
                "is_int": is_int,
            })

        # Impulse schedule: deterministic from trial index + duty cycle
        # Use a simple threshold: trial_idx % period == 0
        self._period = max(1, round(1.0 / duty_cycle))

    def _extract_ranges(self, params_config) -> Dict[str, tuple]:
        ranges = {}
        settings = params_config.get(
            "greenhouse", params_config.get("microgrid", params_config)
        )
        for pname, pconfig in settings.items():
            if not isinstance(pconfig, dict) or "constraints" not in pconfig:
                continue
            for c in pconfig["constraints"]:
                if "lower" in c and "upper" in c:
                    ranges[pname] = (c["lower"], c["upper"])
        return ranges

    def _is_impulse_trial(self, trial_idx: int) -> bool:
        return trial_idx % self._period == 0

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        self._trial_count += 1

        if not self._is_impulse_trial(trial_idx):
            # Normal trial — return sampler params untouched
            return dict(base_params)

        # Impulse trial — push params to extremes
        result = dict(base_params)
        for ip in self._impulse_params:
            name = ip["name"]
            if name not in base_params:
                continue

            # Pick direction for this impulse
            if self.direction == "upper":
                target = ip["upper"]
            elif self.direction == "lower":
                target = ip["lower"]
            else:  # random — alternate based on trial index
                if (trial_idx // self._period) % 2 == 0:
                    target = ip["upper"]
                else:
                    target = ip["lower"]

            # Apply to scalar or list params
            if isinstance(base_params[name], list):
                result[name] = [
                    self._coerce(name, target, ip["is_int"])
                    for _ in base_params[name]
                ]
            else:
                result[name] = self._coerce(name, target, ip["is_int"])

        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "impulse",
            "duty_cycle": self.duty_cycle,
            "period": self._period,
            "direction": self.direction,
            "params": [
                {
                    "param_name": ip["name"],
                    "lower": ip["lower"],
                    "upper": ip["upper"],
                }
                for ip in self._impulse_params
            ],
        }

    def is_impulse_trial(self, trial_idx: int) -> bool:
        """Public accessor for the observer to tag impulse trials."""
        return self._is_impulse_trial(trial_idx)


def create_watermark(
    config: Dict[str, Any],
    params_config: Dict[str, Any],
    override_type: Optional[str] = None,
    breeder_uuid: Optional[str] = None,
) -> Optional[Watermark]:
    interference_config = config.get("interference_detection", {})
    if interference_config.get("mode", "inactive") != "active":
        return None

    # Find all params with ranges, sorted by range size (largest first)
    param_candidates = []
    settings = params_config.get(
        "greenhouse", params_config.get("microgrid", params_config)
    )
    for pname, pconfig in settings.items():
        if not isinstance(pconfig, dict) or "constraints" not in pconfig:
            continue
        for c in pconfig["constraints"]:
            if "lower" in c and "upper" in c:
                param_candidates.append(
                    {
                        "name": pname,
                        "lower": c["lower"],
                        "upper": c["upper"],
                        "range": c["upper"] - c["lower"],
                    }
                )
    param_candidates.sort(key=lambda x: x["range"], reverse=True)

    if not param_candidates:
        logger.warning("No params with ranges found — cannot create impulse watermark")
        return None

    # Take top 3 params (or fewer) — the ones with largest range
    selected = param_candidates[: min(3, len(param_candidates))]

    # Read impulse config or use defaults
    impulse_config = interference_config.get("impulse", {})
    duty_cycle = impulse_config.get("duty_cycle", 0.02)
    direction = impulse_config.get("direction", "random")

    wm = Impulse(
        params_config=params_config,
        param_configs=selected,
        duty_cycle=duty_cycle,
        direction=direction,
    )
    logger.info(
        "Created impulse watermark: %d params, duty_cycle=%.3f, period=%d, direction=%s"
        % (
            len(selected),
            duty_cycle,
            wm._period,
            direction,
        )
    )
    return wm
