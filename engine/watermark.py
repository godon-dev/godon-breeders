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


class OnOff(Watermark):
    def __init__(self, params_config: Dict[str, Any], period: int = 10):
        super().__init__(params_config)
        self.period = period

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        phase = (trial_idx // self.period) % 2
        if phase == 0:
            self._trial_count += 1
            return dict(base_params)
        else:
            return {}

    def is_active_phase(self) -> bool:
        return True

    def metadata(self) -> Dict[str, Any]:
        return {
            'type': 'on_off',
            'period': self.period,
            'total_trials': self.period * 2,
            'cycles': 1,
        }

    def cycle_count(self) -> int:
        return 1


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
        gh_settings = params_config.get('greenhouse', params_config)
        for pname, pconfig in gh_settings.items():
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


class Step(Watermark):
    def __init__(self, params_config: Dict[str, Any], param_name: str,
                 step_fraction: float = 0.2, period: int = 10):
        super().__init__(params_config)
        self.param_name = param_name
        self.step_fraction = step_fraction
        self.period = period
        self._param_ranges = self._extract_ranges(params_config)

    def _extract_ranges(self, params_config) -> Dict[str, tuple]:
        ranges = {}
        gh_settings = params_config.get('greenhouse', params_config)
        for pname, pconfig in gh_settings.items():
            if not isinstance(pconfig, dict) or 'constraints' not in pconfig:
                continue
            for c in pconfig['constraints']:
                if 'lower' in c and 'upper' in c:
                    ranges[pname] = (c['lower'], c['upper'])
        return ranges

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base_params)
        phase = (trial_idx // self.period) % 2
        if self.param_name in base_params and phase == 0:
            base_val = base_params[self.param_name]
            offset = self.step_fraction * abs(base_val)
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
            'type': 'step',
            'param_name': self.param_name,
            'step_fraction': self.step_fraction,
            'period': self.period,
        }

    def cycle_count(self) -> int:
        return 1


class MultiFrequency(Watermark):
    def __init__(self, params_config: Dict[str, Any], param_names: List[str],
                 amplitude: float = 0.1, base_period: int = 20):
        super().__init__(params_config)
        self.param_names = param_names
        self.amplitude = amplitude
        self.base_period = base_period
        self._param_ranges = self._extract_ranges(params_config)

    def _extract_ranges(self, params_config) -> Dict[str, tuple]:
        ranges = {}
        gh_settings = params_config.get('greenhouse', params_config)
        for pname, pconfig in gh_settings.items():
            if not isinstance(pconfig, dict) or 'constraints' not in pconfig:
                continue
            for c in pconfig['constraints']:
                if 'lower' in c and 'upper' in c:
                    ranges[pname] = (c['lower'], c['upper'])
        return ranges

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base_params)
        for i, pname in enumerate(self.param_names):
            if pname not in base_params:
                continue
            freq = i + 1
            offset = self.amplitude * math.sin(2 * math.pi * freq * trial_idx / self.base_period)
            base_val = base_params[pname]
            if isinstance(base_val, list):
                result[pname] = [
                    self._clamp(v + offset * abs(v), *self._param_ranges.get(pname, (v * 0.5, v * 1.5)))
                    for v in base_val
                ]
            else:
                lo, hi = self._param_ranges.get(pname, (base_val * 0.5, base_val * 1.5))
                result[pname] = self._clamp(base_val + offset * abs(base_val), lo, hi)
        self._trial_count += 1
        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            'type': 'multi_frequency',
            'param_names': self.param_names,
            'amplitude': self.amplitude,
            'base_period': self.base_period,
            'frequencies': {pname: i + 1 for i, pname in enumerate(self.param_names)},
        }

    def cycle_count(self) -> int:
        return 1


class Composite(Watermark):
    def __init__(self, watermarks: List[Watermark], cycles: int = 1):
        super().__init__({})
        self.watermarks = watermarks
        self.cycles = cycles
        self._trial_idx = 0
        self._current_wm_idx = 0
        self._current_cycle = 0

    def generate(self, trial_idx: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        wm = self.watermarks[self._current_wm_idx]
        local_idx = self._trial_idx

        result = wm.generate(local_idx, base_params)

        self._trial_idx += 1
        if self._trial_idx >= wm.trial_count() if hasattr(wm, 'total_trials') else wm.metadata().get('total_trials', 20):
            self._trial_idx = 0
            self._current_wm_idx += 1
            if self._current_wm_idx >= len(self.watermarks):
                self._current_wm_idx = 0
                self._current_cycle += 1

        return result

    def is_active_phase(self) -> bool:
        wm = self.watermarks[self._current_wm_idx]
        return wm.is_active_phase()

    def metadata(self) -> Dict[str, Any]:
        return {
            'type': 'composite',
            'watermarks': [wm.metadata() for wm in self.watermarks],
            'cycles': self.cycles,
        }

    def cycle_count(self) -> int:
        return self.cycles

    def is_complete(self) -> bool:
        return self._current_cycle >= self.cycles


def create_watermark(config: Dict[str, Any], params_config: Dict[str, Any],
                     override_type: Optional[str] = None) -> Optional[Watermark]:
    interference_config = config.get('interference_detection', {})
    if interference_config.get('mode', 'inactive') != 'active':
        return None

    wm_type = override_type or 'sinusoidal'

    if wm_type == 'on_off':
        period = max(5, min(interference_config.get('phase_trials', 10), 20))
        return OnOff(params_config=params_config, period=period)
    elif wm_type == 'sinusoidal':
        param_name, amplitude = _pick_param_and_amplitude(params_config)
        period = max(10, min(interference_config.get('phase_trials', 20), 40))
        phase_offset = random.uniform(0, 2 * math.pi)
        return Sinusoidal(
            params_config=params_config,
            param_name=param_name,
            amplitude=amplitude,
            period=period,
            phase_offset=phase_offset,
        )
    elif wm_type == 'step':
        param_name, _ = _pick_param_and_amplitude(params_config)
        return Step(
            params_config=params_config,
            param_name=param_name,
            step_fraction=0.3,
            period=10,
        )
    elif wm_type == 'multi_frequency':
        param_names = _pick_multi_params(params_config, count=3)
        return MultiFrequency(
            params_config=params_config,
            param_names=param_names,
            amplitude=0.15,
            base_period=20,
        )
    elif wm_type == 'composite':
        on_off = OnOff(params_config=params_config, period=10)
        multi = MultiFrequency(params_config=params_config, param_names=_pick_multi_params(params_config, count=3), amplitude=0.15, base_period=20)
        return Composite(watermarks=[on_off, multi], cycles=1)
    else:
        logger.warning(f"Unknown watermark type: {wm_type}, falling back to sinusoidal")
        param_name, amplitude = _pick_param_and_amplitude(params_config)
        return Sinusoidal(params_config=params_config, param_name=param_name, amplitude=amplitude, period=20)


def _pick_param_and_amplitude(params_config: Dict[str, Any]) -> tuple:
    best_name = ''
    best_range = 0.0
    best_mid = 1.0
    gh_settings = params_config.get('greenhouse', params_config)
    for pname, pconfig in gh_settings.items():
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
    amplitude = 0.15 * best_mid
    return best_name, amplitude


def _pick_multi_params(params_config: Dict[str, Any], count: int = 3) -> List[str]:
    candidates = []
    gh_settings = params_config.get('greenhouse', params_config)
    for pname, pconfig in gh_settings.items():
        if not isinstance(pconfig, dict) or 'constraints' not in pconfig:
            continue
        for c in pconfig['constraints']:
            if 'lower' in c and 'upper' in c:
                candidates.append((c['upper'] - c['lower'], pname))
    candidates.sort(reverse=True)
    return [name for _, name in candidates[:count]]
