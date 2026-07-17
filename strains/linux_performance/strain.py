
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
from typing import Dict, Any, List, Optional

import optuna
from f.breeder.strains.linux_performance.parameter_registry import PARAMETER_REGISTRY
from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)

CATEGORIES = ["sysctl", "sysfs", "cpufreq", "ethtool"]


def suggest_params(trial, settings: Dict[str, Any]) -> Dict[str, Any]:
    params = {}

    for category in CATEGORIES:
        if category not in settings:
            continue

        category_settings = settings[category]

        if category == 'ethtool':
            for interface_name, interface_config in category_settings.items():
                if not isinstance(interface_config, dict):
                    logger.warning(f"Invalid ethtool config for {interface_name}: not a dict")
                    continue

                for param_name, param_config in interface_config.items():
                    if 'constraints' not in param_config:
                        logger.warning(f"Missing constraints for ethtool.{interface_name}.{param_name}")
                        continue

                    value = _suggest_single_param(
                        trial, param_name, param_config['constraints'],
                        category=f"ethtool.{interface_name}"
                    )
                    params[f"{interface_name}_{param_name}"] = value
                    logger.debug(f"Suggested ethtool.{interface_name}.{param_name} = {value}")
        else:
            for param_name, param_config in category_settings.items():
                if 'constraints' not in param_config:
                    logger.warning(f"Missing constraints for {category}.{param_name}")
                    continue

                constraints = param_config['constraints']
                value = _suggest_single_param(trial, param_name, constraints, category=category)
                params[param_name] = value
                logger.debug(f"Suggested {category}.{param_name} = {value}")

    return params


def _suggest_single_param(trial: optuna.Trial, param_name: str,
                         constraints_list: List[Dict[str, Any]], category: str) -> Any:
    if not isinstance(constraints_list, list) or len(constraints_list) == 0:
        raise ValueError(f"{category}.{param_name}: constraints must be a non-empty list")

    first_constraint = constraints_list[0]

    if 'values' in first_constraint:
        values = first_constraint['values']
        return trial.suggest_categorical(param_name, values)

    elif 'step' in first_constraint and 'lower' in first_constraint and 'upper' in first_constraint:
        lower = first_constraint['lower']
        upper = first_constraint['upper']
        step = first_constraint['step']

        if isinstance(step, int) and isinstance(lower, int) and isinstance(upper, int):
            return trial.suggest_int(param_name, lower, upper, step=step)
        else:
            return trial.suggest_float(param_name, lower, upper, step=step)
    else:
        raise ValueError(
            f"{category}.{param_name}: constraint must have either 'values' (categorical) "
            f"or 'step/lower/upper' (numeric range)"
        )


def validate_config(config):
    from f.breeder.strains.linux_performance import preflight
    return preflight.main(config)
