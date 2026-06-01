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
        return ranges

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base_params)
        if self.param_name in base_params:
            base_val = base_params[self.param_name]
            offset = self.amplitude * math.sin(2 * math.pi * trial_idx / self.period + self.phase_offset)
            if isinstance(base_val, list):
                result[self.param_name] = [
                    self._clamp(v + offset, *self._param_ranges.get(self.param_name, (v * 0.5, v * 1.5)))
                    for v in base_val
                ]
            else:
                lo, hi = self._param_ranges.get(self.param_name, (base_val * 0.5, base_val * 1.5))
                result[self.param_name] = self._clamp(base_val + offset, lo, hi)
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
                    self._clamp(v + offset, *self._param_ranges.get(self.param_name, (v * 0.5, v * 1.5)))
                    for v in base_val
                ]
            else:
                lo, hi = self._param_ranges.get(self.param_name, (base_val * 0.5, base_val * 1.5))
                result[self.param_name] = self._clamp(base_val + offset, lo, hi)
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


def create_watermark(config: Dict[str, Any], params_config: Dict[str, Any],
                     override_type: Optional[str] = None, breeder_uuid: Optional[str] = None) -> Optional[Watermark]:
    interference_config = config.get('interference_detection', {})
    if interference_config.get('mode', 'inactive') != 'active':
        return None

    param_name, amplitude = _pick_param_and_amplitude(params_config)

    period_candidates = [17, 23, 29, 37]
    if breeder_uuid:
        # Use UUID hash to select 2-3 unique periods as breeder fingerprint
        h = int(hashlib.md5(breeder_uuid.encode()).hexdigest(), 16)
        n_freqs = 2 + (h % 2)  # 2 or 3 frequencies
        indices = [(h >> (i * 3)) % len(period_candidates) for i in range(n_freqs)]
        periods = list(dict.fromkeys(period_candidates[i] for i in indices))  # unique, ordered
        if len(periods) < 2:
            periods = period_candidates[:2]
    else:
        period = max(10, min(interference_config.get('phase_trials', 20), 40))
        periods = [period]

    phase_offsets = [random.uniform(0, 2 * math.pi) for _ in periods]

    if len(periods) == 1:
        return Sinusoidal(
            params_config=params_config,
            param_name=param_name,
            amplitude=amplitude,
            period=periods[0],
            phase_offset=phase_offsets[0],
        )
    return MultiFrequency(
        params_config=params_config,
        param_name=param_name,
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
