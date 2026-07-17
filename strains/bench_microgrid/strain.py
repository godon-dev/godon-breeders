
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
from f.breeder.strains.bench_microgrid.parameter_registry import PARAMETER_REGISTRY
from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


def suggest_params(trial, settings: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    mg_settings = settings.get('microgrid', settings)

    for param_name, param_config in mg_settings.items():
        if not isinstance(param_config, dict) or 'constraints' not in param_config:
            continue

        constraints = param_config['constraints']
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
    from f.breeder.strains.bench_microgrid import preflight
    return preflight.main(config)
