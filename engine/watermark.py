import hashlib
import math
import random
from typing import Dict, Any, Optional, List
from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


class Watermark:
    def __init__(self, params_config: Dict[str, Any]):
        self.params_config = params_config
        self._trial_count = 0
        self._int_params: set = set()

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError

    def cycle_count(self) -> int:
        raise NotImplementedError

    def trial_count(self) -> int:
        return self._trial_count

    def is_active_phase(self) -> bool:
        return True

    def _clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def _coerce(self, pname: str, value):
        """Round float watermarks back to int for integer-range params.

        Sinusoidal offsets are always float.  Params whose config bounds are
        both int (e.g. sim_steps: 10..200) should stay int after watermarking.
        """
        if pname in self._int_params and isinstance(value, float):
            return int(round(value))
        return value


class Sinusoidal(Watermark):
    def __init__(self, params_config: Dict[str, Any], param_name: str,
                 amplitude: float = 0.1, period: int = 20, phase_offset: float = 0.0):
        super().__init__(params_config)
        self.param_name = param_name
        self.amplitude = amplitude
        self.period = period
        self.phase_offset = phase_offset
        self._param_ranges = self._extract_ranges(params_config)

    def _extract_ranges(self, params_config) -> Dict[str, tuple]:
        ranges = {}
        settings = params_config.get('greenhouse', params_config.get('microgrid', params_config))
        for pname, pconfig in settings.items():
            if not isinstance(pconfig, dict) or 'constraints' not in pconfig:
                continue
            for c in pconfig['constraints']:
                if 'lower' in c and 'upper' in c:
                    ranges[pname] = (c['lower'], c['upper'])
                    if isinstance(c['lower'], int) and isinstance(c['upper'], int):
                        self._int_params.add(pname)
        return ranges

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base_params)
        if self.param_name in base_params:
            base_val = base_params[self.param_name]
            offset = self.amplitude * math.sin(2 * math.pi * trial_idx / self.period + self.phase_offset)
            if isinstance(base_val, list):
                result[self.param_name] = [
                    self._coerce(self.param_name, self._clamp(v + offset, *self._param_ranges.get(self.param_name, (v * 0.5, v * 1.5))))
                    for v in base_val
                ]
            else:
                lo, hi = self._param_ranges.get(self.param_name, (base_val * 0.5, base_val * 1.5))
                result[self.param_name] = self._coerce(self.param_name, self._clamp(base_val + offset, lo, hi))
        self._trial_count += 1
        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            'type': 'sinusoidal',
            'param_name': self.param_name,
            'amplitude': self.amplitude,
            'period': self.period,
            'phase_offset': round(self.phase_offset, 4),
        }

    def cycle_count(self) -> int:
        return 1


class MultiFrequency(Watermark):
    """Superposes multiple sinusoidal watermarks at prime periods.

    Each breeder uses a unique subset of periods (fingerprint) selected by
    its UUID hash. Research shows coupling factor does not affect
    detectability — only SNR matters — so using multiple frequencies
    simultaneously provides statistical robustness without signal loss.
    The amplitude per component is divided equally across frequencies.
    """

    def __init__(self, params_config: Dict[str, Any], param_name: str,
                 total_amplitude: float, periods: List[int],
                 phase_offsets: Optional[List[float]] = None):
        super().__init__(params_config)
        self.param_name = param_name
        self.periods = periods
        self.total_amplitude = total_amplitude
        self._param_ranges = self._extract_ranges(params_config)
        # Divide amplitude equally across frequency components
        self._component_amp = total_amplitude / len(periods)
        if phase_offsets and len(phase_offsets) == len(periods):
            self._phase_offsets = phase_offsets
        else:
            self._phase_offsets = [random.uniform(0, 2 * math.pi) for _ in periods]

    def _extract_ranges(self, params_config) -> Dict[str, tuple]:
        ranges = {}
        settings = params_config.get('greenhouse', params_config.get('microgrid', params_config))
        for pname, pconfig in settings.items():
            if not isinstance(pconfig, dict) or 'constraints' not in pconfig:
                continue
            for c in pconfig['constraints']:
                if 'lower' in c and 'upper' in c:
                    ranges[pname] = (c['lower'], c['upper'])
                    if isinstance(c['lower'], int) and isinstance(c['upper'], int):
                        self._int_params.add(pname)
        return ranges

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base_params)
        if self.param_name in base_params:
            base_val = base_params[self.param_name]
            offset = sum(
                self._component_amp * math.sin(2 * math.pi * trial_idx / p + po)
                for p, po in zip(self.periods, self._phase_offsets)
            )
            if isinstance(base_val, list):
                result[self.param_name] = [
                    self._coerce(self.param_name, self._clamp(v + offset, *self._param_ranges.get(self.param_name, (v * 0.5, v * 1.5))))
                    for v in base_val
                ]
            else:
                lo, hi = self._param_ranges.get(self.param_name, (base_val * 0.5, base_val * 1.5))
                result[self.param_name] = self._coerce(self.param_name, self._clamp(base_val + offset, lo, hi))
        self._trial_count += 1
        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            'type': 'multi_frequency',
            'param_name': self.param_name,
            'total_amplitude': self.total_amplitude,
            'component_amplitude': round(self._component_amp, 4),
            'periods': self.periods,
            'phase_offsets': [round(po, 4) for po in self._phase_offsets],
        }

    def cycle_count(self) -> int:
        # LCM of all periods — full cycle repeats at this interval
        from math import gcd
        from functools import reduce
        def lcm(a, b):
            return a * b // gcd(a, b)
        return reduce(lcm, self.periods)


class MultiFrequencyMultiParam(Watermark):
    """Applies multi-frequency sinusoidal watermarks to multiple parameters.

    Each parameter gets its own set of periods (from the breeder's fingerprint),
    creating multiple independent signal paths from sender to receiver objectives.
    This gives MIMO-like processing gain: the coupling signal reaches the
    receiver through multiple parameter channels simultaneously.

    Amplitude per parameter is 50% of that parameter's mid-range.
    Each param gets a subset of the available periods, rotated so params
    don't share the same frequencies (reducing interference between channels).
    """

    def __init__(self, params_config: Dict[str, Any],
                 param_configs: List[Dict[str, Any]],
                 periods: List[int],
                 phase_offsets: Optional[List[float]] = None):
        super().__init__(params_config)
        self._param_ranges = self._extract_ranges(params_config)
        self.periods = periods

        # Each param gets its own amplitude (50% of mid-range)
        self._param_watermarks = []
        for i, pc in enumerate(param_configs):
            pname = pc['name']
            lo, hi = pc['lower'], pc['upper']
            param_range = hi - lo
            amplitude = 0.25 * param_range  # 25% of range — strong signal

            # Rotate periods across params so they don't share same frequencies
            # Each param gets 2 periods, offset by param index
            param_periods = []
            for j in range(2):
                idx = (i * 2 + j) % len(periods)
                param_periods.append(periods[idx])

            param_phases = [random.uniform(0, 2 * math.pi) for _ in param_periods]
            self._param_watermarks.append({
                'name': pname,
                'amplitude': amplitude,
                'periods': param_periods,
                'phase_offsets': param_phases,
            })

    def _extract_ranges(self, params_config) -> Dict[str, tuple]:
        ranges = {}
        settings = params_config.get('greenhouse', params_config.get('microgrid', params_config))
        for pname, pconfig in settings.items():
            if not isinstance(pconfig, dict) or 'constraints' not in pconfig:
                continue
            for c in pconfig['constraints']:
                if 'lower' in c and 'upper' in c:
                    ranges[pname] = (c['lower'], c['upper'])
                    if isinstance(c['lower'], int) and isinstance(c['upper'], int):
                        self._int_params.add(pname)
        return ranges

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base_params)
        for wm in self._param_watermarks:
            pname = wm['name']
            if pname not in base_params:
                continue
            # Use midpoint as base — ignore sampler's choice for clean watermark signal.
            # The optimizer still controls non-watermarked parameters freely.
            lo, hi = self._param_ranges.get(pname, (0, 1000))
            base_val = (lo + hi) / 2.0
            offset = sum(
                (wm['amplitude'] / len(wm['periods'])) *
                math.sin(2 * math.pi * trial_idx / p + po)
                for p, po in zip(wm['periods'], wm['phase_offsets'])
            )
            if isinstance(base_params[pname], list):
                result[pname] = [
                    self._coerce(pname, self._clamp(base_val + offset, lo, hi))
                    for _ in base_params[pname]
                ]
            else:
                result[pname] = self._coerce(pname, self._clamp(base_val + offset, lo, hi))
        self._trial_count += 1
        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            'type': 'multi_frequency_multi_param',
            'periods': self.periods,
            'params': [
                {
                    'param_name': wm['name'],
                    'amplitude': round(wm['amplitude'], 4),
                    'periods': wm['periods'],
                    'phase_offsets': [round(po, 4) for po in wm['phase_offsets']],
                }
                for wm in self._param_watermarks
            ],
        }

    def cycle_count(self) -> int:
        from math import gcd
        from functools import reduce
        def lcm(a, b):
            return a * b // gcd(a, b)
        # Collect all periods across all params
        all_periods = set()
        for wm in self._param_watermarks:
            all_periods.update(wm['periods'])
        if not all_periods:
            return 1
        return reduce(lcm, all_periods)


def create_watermark(config: Dict[str, Any], params_config: Dict[str, Any],
                     override_type: Optional[str] = None, breeder_uuid: Optional[str] = None) -> Optional[Watermark]:
    interference_config = config.get('interference_detection', {})
    if interference_config.get('mode', 'inactive') != 'active':
        return None

    # Find all params with ranges, sorted by range size (largest first)
    param_candidates = []
    settings = params_config.get('greenhouse', params_config.get('microgrid', params_config))
    for pname, pconfig in settings.items():
        if not isinstance(pconfig, dict) or 'constraints' not in pconfig:
            continue
        for c in pconfig['constraints']:
            if 'lower' in c and 'upper' in c:
                param_candidates.append({
                    'name': pname,
                    'lower': c['lower'],
                    'upper': c['upper'],
                    'range': c['upper'] - c['lower'],
                })
    param_candidates.sort(key=lambda x: x['range'], reverse=True)

    # Select breeder fingerprint periods — non-overlapping across breeders.
    # Each breeder gets a unique slice of the prime pool so no two breeders
    # share any frequency.  This eliminates cross-contamination where
    # self-subtraction residuals at shared periods cause false positives.
    period_candidates = [17, 23, 29, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    freqs_per_breeder = 2
    max_slots = len(period_candidates) // freqs_per_breeder

    # Slot assignment from controller (collision-free). Required.
    breeder_section = config.get('breeder', {})
    explicit_slot = breeder_section.get('watermark_slot')
    if explicit_slot is None:
        raise ValueError(
            "No watermark_slot assigned. The controller must assign a collision-free "
            "slot when creating the breeder. Cannot continue without one."
        )
    if not (0 <= explicit_slot < max_slots):
        raise ValueError(
            f"Invalid watermark_slot {explicit_slot}. Must be 0-{max_slots - 1}."
        )
    periods = period_candidates[explicit_slot * freqs_per_breeder : (explicit_slot + 1) * freqs_per_breeder]
    logger.info(f"Using assigned watermark slot {explicit_slot}, periods={periods}")

    # Use multi-param watermark if we have 2+ params, otherwise single-param
    if len(param_candidates) >= 2:
        # Take top 3 params (or fewer if not enough)
        selected = param_candidates[:min(3, len(param_candidates))]
        return MultiFrequencyMultiParam(
            params_config=params_config,
            param_configs=selected,
            periods=periods,
        )

    # Fallback: single-param with old behavior
    best = param_candidates[0] if param_candidates else {'name': 'light_intensity', 'lower': 0, 'upper': 1000}
    param_range = best['upper'] - best['lower']
    amplitude = 0.25 * param_range
    phase_offsets = [random.uniform(0, 2 * math.pi) for _ in periods]

    if len(periods) == 1:
        return Sinusoidal(
            params_config=params_config,
            param_name=best['name'],
            amplitude=amplitude,
            period=periods[0],
            phase_offset=phase_offsets[0],
        )
    return MultiFrequency(
        params_config=params_config,
        param_name=best['name'],
        total_amplitude=amplitude,
        periods=periods,
        phase_offsets=phase_offsets,
    )


def _pick_param_and_amplitude(params_config: Dict[str, Any]) -> tuple:
    best_name = ''
    best_range = 0.0
    best_mid = 1.0
    settings = params_config.get('greenhouse', params_config.get('microgrid', params_config))
    for pname, pconfig in settings.items():
        if not isinstance(pconfig, dict) or 'constraints' not in pconfig:
            continue
        for c in pconfig['constraints']:
            if 'lower' in c and 'upper' in c:
                r = c['upper'] - c['lower']
                if r > best_range:
                    best_range = r
                    best_name = pname
                    best_mid = (c['lower'] + c['upper']) / 2.0
    if not best_name:
        best_name = 'light_intensity'
        best_mid = 500.0
        best_range = 1000.0
    # Research shows SNR ~0.75 is the detection breaking point.
    # At 30% of mid-range, SNR stays above 0.75 even with aggressive
    # optimizer noise (std ~ mid-range/3). Previous 15% was too weak.
    amplitude = 0.30 * best_mid
    return best_name, amplitude
