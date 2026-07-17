
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
from f.breeder.strains.bench_greenhouse.parameter_registry import PARAMETER_REGISTRY

from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


def main(config=None, strict_mode=True):
    if not config:
        return {
            "result": "FAILURE",
            "error": "Missing config parameter"
        }

    errors = []
    warnings = []

    meta_section = config.get('meta', {})
    if 'strict_validation' in meta_section:
        strict_mode = meta_section['strict_validation']

    try:
        settings = config.get('settings', {})
        gh_settings = settings.get('greenhouse', settings)

        if not isinstance(gh_settings, dict):
            return {
                "result": "FAILURE",
                "error": "settings.greenhouse must be a dict"
            }

        zones = gh_settings.get('zones', 2)
        if not isinstance(zones, int) or zones < 1:
            errors.append(f"settings.greenhouse.zones must be a positive integer, got: {zones}")

        for param_name, param_config in gh_settings.items():
            if param_name == 'zones':
                continue

            if not isinstance(param_config, dict):
                errors.append(f"settings.greenhouse.{param_name}: must be a dict")
                continue

            if param_name not in PARAMETER_REGISTRY:
                if strict_mode:
                    errors.append(
                        f"settings.greenhouse.{param_name}: unsupported parameter. "
                        f"Supported: {', '.join(PARAMETER_REGISTRY.keys())}"
                    )
                    continue
                else:
                    warnings.append(
                        f"settings.greenhouse.{param_name}: unknown parameter, "
                        f"skipping registry validation"
                    )

            if 'constraints' not in param_config:
                errors.append(f"settings.greenhouse.{param_name}: missing 'constraints'")
                continue

            constraints = param_config['constraints']

            if isinstance(constraints, dict):
                if 'values' in constraints:
                    constraints = [constraints]
                else:
                    errors.append(f"settings.greenhouse.{param_name}: constraints dict must have 'values' key")
                    continue

            if not isinstance(constraints, list):
                errors.append(f"settings.greenhouse.{param_name}: constraints must be a list")
                continue

            if param_name in PARAMETER_REGISTRY:
                registry_type = PARAMETER_REGISTRY[param_name]['type']
                if registry_type == "categorical":
                    if not any('values' in c for c in constraints):
                        errors.append(
                            f"settings.greenhouse.{param_name}: "
                            f"categorical param needs 'values' in constraints"
                        )
                elif registry_type in ["int", "float"]:
                    if not any('step' in c and 'lower' in c and 'upper' in c for c in constraints):
                        errors.append(
                            f"settings.greenhouse.{param_name}: "
                            f"{registry_type} param needs step/lower/upper in constraints"
                        )

        if errors:
            error_msg = "Preflight validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            return {
                "result": "FAILURE",
                "error": error_msg
            }

        result = {
            "result": "SUCCESS",
            "data": {
                "message": "Preflight validation passed"
            }
        }

        if warnings:
            result["data"]["warnings"] = warnings

        return result

    except Exception as e:
        return {
            "result": "FAILURE",
            "error": f"Preflight validation error: {str(e)}"
        }
