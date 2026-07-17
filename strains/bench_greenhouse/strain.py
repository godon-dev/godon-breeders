
#
# Copyright (c) 2019 Matthias Tafelmeier.
#
# This file is part of godon
#
# godon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# godon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this godon. If not, see <http://www.gnu.org/licenses/>.
#
from typing import Dict, Any, List

import optuna
from f.breeder.strains.bench_greenhouse.parameter_registry import PARAMETER_REGISTRY
from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)

PER_ZONE_PARAMS = {"heating_setpoints", "vent_openings"}


def suggest_params(trial, settings: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    gh_settings = settings.get('greenhouse', settings)
    zones = gh_settings.get('zones', 2)

    for param_name, param_config in gh_settings.items():
        if param_name == 'zones':
            continue

        if not isinstance(param_config, dict) or 'constraints' not in param_config:
            continue

        constraints = param_config['constraints']
        registry = PARAMETER_REGISTRY.get(param_name, {})
        scope = registry.get('scope', 'global')

        if param_name in PER_ZONE_PARAMS or scope == 'per_zone':
            zone_values = []
            for zone_idx in range(zones):
                optuna_name = f"{param_name}_{zone_idx}"
                value = _suggest_single(trial, optuna_name, constraints)
                zone_values.append(value)
                logger.debug(f"Suggested {param_name}[{zone_idx}] = {value}")
            params[param_name] = zone_values
        else:
            value = _suggest_single(trial, param_name, constraints)
            params[param_name] = value
            logger.debug(f"Suggested {param_name} = {value}")

    return params


def _suggest_single(trial, param_name: str, constraints_list: List[Dict[str, Any]]) -> Any:
    if not isinstance(constraints_list, list) or len(constraints_list) == 0:
        raise ValueError(f"{param_name}: constraints must be a non-empty list")

    first = constraints_list[0]

    if 'values' in first:
        return trial.suggest_categorical(param_name, first['values'])
    elif 'step' in first and 'lower' in first and 'upper' in first:
        lower, upper, step = first['lower'], first['upper'], first['step']
        if isinstance(step, int) and isinstance(lower, int) and isinstance(upper, int):
            return trial.suggest_int(param_name, lower, upper, step=step)
        else:
            return trial.suggest_float(param_name, lower, upper, step=step)
    else:
        raise ValueError(
            f"{param_name}: constraint must have either 'values' (categorical) "
            f"or 'step/lower/upper' (numeric range)"
        )


def validate_config(config):
    from f.breeder.strains.bench_greenhouse import preflight
    return preflight.main(config)
