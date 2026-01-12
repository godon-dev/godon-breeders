"""
Preflight validation for linux_performance breeder.

This script performs semantic validation before workers are launched:
- Validates all parameters are supported by this breeder
- Validates constraint types match parameter types
- Returns error if config is invalid, allowing controller to fail fast

Called synchronously by controller before starting workers async.
"""

def main(config=None):
    """
    Validate breeder configuration for linux_performance breeder.

    Args:
        config: Breeder configuration dict

    Returns:
        dict with result status and either success or error details
    """
    if not config:
        return {
            "result": "FAILURE",
            "error": "Missing config parameter"
        }

    # Parameter registry for linux_performance breeder
    #
    # MAKESHIFT IMPLEMENTATION - WILL BE REPLACED WITH AUTO-DISCOVERY
    #
    # Current approach: Hardcoded registry of known parameters
    # Future approach: Auto-discovery script that runs:
    #   - sysctl -a (to discover all sysctl parameters)
    #   - scan /sys filesystem (for sysfs parameters)
    #   - ethtool queries (for network interface parameters)
    #   - parse kernel docs for metadata
    # Then export to YAML file that preflight loads.
    #
    # Why makeshift:
    # - Hardcoded list will become incomplete as Linux evolves
    # - Manual maintenance doesn't scale
    # - Auto-discovery is the sustainable long-term solution
    #
    # For now: This works for v0.3. Add parameters as needed during testing.
    # Unknown parameters will be logged as warnings (not errors) to allow
    # experimentation. Registry will grow organically until auto-discovery
    # is implemented.
    #
    # Defines all supported parameters, their types, and metadata
    PARAMETER_REGISTRY = {
        # sysctl parameters
        "net.ipv4.tcp_rmem": {
            "type": "int",
            "category": "sysctl",
            "requires_reboot": False,
            "description": "TCP read buffer sizes"
        },
        "net.ipv4.tcp_wmem": {
            "type": "int",
            "category": "sysctl",
            "requires_reboot": False,
            "description": "TCP write buffer sizes"
        },
        "net.core.netdev_budget": {
            "type": "int",
            "category": "sysctl",
            "requires_reboot": False,
            "description": "Network device budget"
        },
        "net.core.netdev_max_backlog": {
            "type": "int",
            "category": "sysctl",
            "requires_reboot": False,
            "description": "Maximum backlog queue length"
        },
        "net.core.dev_weight": {
            "type": "int",
            "category": "sysctl",
            "requires_reboot": False,
            "description": "CPU weight for network device processing"
        },
        "net.ipv4.tcp_congestion_control": {
            "type": "categorical",
            "category": "sysctl",
            "requires_reboot": False,
            "description": "TCP congestion control algorithm"
        },

        # sysfs parameters (aliases for filesystem paths)
        "cpu_governor": {
            "type": "categorical",
            "category": "sysfs",
            "requires_reboot": False,
            "path": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
            "description": "CPU frequency scaling governor"
        },
        "transparent_hugepage": {
            "type": "categorical",
            "category": "sysfs",
            "requires_reboot": False,
            "path": "/sys/kernel/mm/transparent_hugepage/enabled",
            "description": "Transparent huge pages setting"
        },
        "qdisc": {
            "type": "categorical",
            "category": "sysfs",
            "requires_reboot": False,
            "path": "/sys/class/net/eth0/queue/disc",
            "description": "Network interface queue discipline"
        },

        # cpufreq parameters
        "governor": {
            "type": "categorical",
            "category": "cpufreq",
            "requires_reboot": False,
            "description": "CPU frequency governor"
        },
        "min_freq_ghz": {
            "type": "float",
            "category": "cpufreq",
            "requires_reboot": False,
            "description": "Minimum CPU frequency in GHz"
        },
        "max_freq_ghz": {
            "type": "float",
            "category": "cpufreq",
            "requires_reboot": False,
            "description": "Maximum CPU frequency in GHz"
        },

        # ethtool parameters (per-interface)
        # Note: Interface names (eth0, etc.) are dynamic keys under ethtool category
    }

    # Supported ethtool parameters (apply to any network interface)
    ETHTOOL_PARAMS = {
        "tso": {
            "type": "categorical",
            "requires_reboot": False,
            "description": "TCP Segmentation Offload"
        },
        "gro": {
            "type": "categorical",
            "requires_reboot": False,
            "description": "Generic Receive Offload"
        },
        "rx_ring": {
            "type": "int",
            "requires_reboot": False,
            "description": "RX ring buffer size"
        },
        "tx_ring": {
            "type": "int",
            "requires_reboot": False,
            "description": "TX ring buffer size"
        },
    }

    errors = []

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
                        if not isinstance(constraints, list):
                            errors.append(f"settings.ethtool.{param_name}.{ethtool_param}: constraints must be a list")
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
                        errors.append(
                            f"settings.{category}.{param_name}: "
                            f"unsupported parameter. "
                            f"Supported {category} parameters: "
                            f"{', '.join([k for k, v in PARAMETER_REGISTRY.items() if v['category'] == category])}"
                        )
                        continue

                    # Validate constraint type matching
                    registry_type = PARAMETER_REGISTRY[param_name]['type']
                    if 'constraints' not in param_config:
                        errors.append(f"settings.{category}.{param_name}: missing 'constraints'")
                        continue

                    constraints = param_config['constraints']
                    if not isinstance(constraints, list):
                        errors.append(f"settings.{category}.{param_name}: constraints must be a list")
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

        return {
            "result": "SUCCESS",
            "data": {
                "message": "Preflight validation passed"
            }
        }

    except Exception as e:
        return {
            "result": "FAILURE",
            "error": f"Preflight validation error: {str(e)}"
        }
