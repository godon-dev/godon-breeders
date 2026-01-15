"""
Preflight validation for linux_performance breeder.

This script performs semantic validation before workers are launched:
- Validates all parameters are supported by this breeder
- Validates constraint types match parameter types
- Returns error if config is invalid, allowing controller to fail fast

Called synchronously by controller before starting workers async.
"""

# Import parameter registry from separate file
from linux_performance.parameter_registry import PARAMETER_REGISTRY, ETHTOOL_PARAMS

def main(config=None, strict_mode=True):
    """
    Validate breeder configuration for linux_performance breeder.

    Args:
        config: Breeder configuration dict
        strict_mode: If True (default), reject unknown parameters.
                   If False, allow unknown parameters with warnings.

    Returns:
        dict with result status and either success or error details
    """
    if not config:
        return {
            "result": "FAILURE",
            "error": "Missing config parameter"
        }

    errors = []
    warnings = []

    # Determine strict mode: parameter > meta config > default (true)
    meta_section = config.get('meta', {})
    if 'strict_validation' in meta_section:
        strict_mode = meta_section['strict_validation']
    # else: use the strict_mode parameter passed in (default: true)

    try:
        settings = config.get('settings', {})

        # Validate each settings category
        for category in ['sysctl', 'sysfs', 'cpufreq', 'ethtool']:
            if category not in settings:
                continue

            category_settings = settings[category]

            if not isinstance(category_settings, dict):
                errors.append(f"settings.{category}: must be a dict")
                continue

            for param_name, param_config in category_settings.items():
                if not isinstance(param_config, dict):
                    errors.append(f"settings.{category}.{param_name}: must be a dict")
                    continue

                # Special handling for ethtool (per-interface parameters)
                if category == 'ethtool':
                    # param_name is interface name (e.g., "eth0")
                    if not isinstance(param_config, dict):
                        errors.append(f"settings.ethtool.{param_name}: must be a dict with interface parameters")
                        continue

                    for ethtool_param, ethtool_config in param_config.items():
                        if ethtool_param not in ETHTOOL_PARAMS:
                            errors.append(
                                f"settings.ethtool.{param_name}.{ethtool_param}: "
                                f"unsupported ethtool parameter. "
                                f"Supported: {', '.join(ETHTOOL_PARAMS.keys())}"
                            )
                            continue

                        # Validate constraint type matching
                        registry_type = ETHTOOL_PARAMS[ethtool_param]['type']
                        if 'constraints' not in ethtool_config:
                            errors.append(f"settings.ethtool.{param_name}.{ethtool_param}: missing 'constraints'")
                            continue

                        constraints = ethtool_config['constraints']

                        # Pragmatic: Accept both dict and list formats
                        if isinstance(constraints, dict):
                            if 'values' in constraints:
                                # Normalize dict to list for categorical
                                constraints = [constraints]
                            else:
                                errors.append(f"settings.ethtool.{param_name}.{ethtool_param}: constraints dict must have 'values' key")
                                continue

                        if not isinstance(constraints, list):
                            errors.append(f"settings.ethtool.{param_name}.{ethtool_param}: constraints must be a list or dict with 'values'")
                            continue

                        # Check type matching
                        if registry_type == "categorical":
                            if not any('values' in c for c in constraints):
                                errors.append(
                                    f"settings.ethtool.{param_name}.{ethtool_param}: "
                                    f"parameter is categorical but constraints don't have 'values'"
                                )
                        elif registry_type in ["int", "float"]:
                            if not any('step' in c and 'lower' in c and 'upper' in c for c in constraints):
                                errors.append(
                                    f"settings.ethtool.{param_name}.{ethtool_param}: "
                                    f"parameter is {registry_type} but constraints don't have step/lower/upper"
                                )

                # Non-ethtool parameters
                else:
                    if param_name not in PARAMETER_REGISTRY:
                        if strict_mode:
                            # Strict mode: reject unknown parameters
                            errors.append(
                                f"settings.{category}.{param_name}: "
                                f"unsupported parameter. "
                                f"Supported {category} parameters: "
                                f"{', '.join([k for k, v in PARAMETER_REGISTRY.items() if v['category'] == category])}"
                            )
                            continue
                        else:
                            # Permissive mode: warn but allow
                            warnings.append(
                                f"settings.{category}.{param_name}: "
                                f"parameter not in registry. "
                                f"Metadata unavailable (reboot requirement, description, safety info). "
                                f"Proceeding with structural validation only."
                            )
                            # Continue to structural validation

                    # Validate constraint type matching
                    registry_type = PARAMETER_REGISTRY[param_name]['type']
                    if 'constraints' not in param_config:
                        errors.append(f"settings.{category}.{param_name}: missing 'constraints'")
                        continue

                    constraints = param_config['constraints']

                    # Pragmatic: Accept both dict and list formats
                    if isinstance(constraints, dict):
                        if 'values' in constraints:
                            # Normalize dict to list for categorical
                            constraints = [constraints]
                        else:
                            errors.append(f"settings.{category}.{param_name}: constraints dict must have 'values' key")
                            continue

                    if not isinstance(constraints, list):
                        errors.append(f"settings.{category}.{param_name}: constraints must be a list or dict with 'values'")
                        continue

                    # Check type matching
                    if registry_type == "categorical":
                        if not any('values' in c for c in constraints):
                            errors.append(
                                f"settings.{category}.{param_name}: "
                                f"parameter is categorical but constraints don't have 'values'"
                            )
                    elif registry_type in ["int", "float"]:
                        if not any('step' in c and 'lower' in c and 'upper' in c for c in constraints):
                            errors.append(
                                f"settings.{category}.{param_name}: "
                                f"parameter is {registry_type} but constraints don't have step/lower/upper"
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
