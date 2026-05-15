PARAMETER_REGISTRY = {
    "power_draw": {
        "type": "float",
        "scope": "global",
        "causes_downtime": False,
        "description": "Power draw from grid (kW)",
        "typical_range": [0.0, 1000.0],
    },
    "storage_dispatch": {
        "type": "float",
        "scope": "global",
        "causes_downtime": False,
        "description": "Battery dispatch (-500=discharge, +500=charge kW)",
        "typical_range": [-500.0, 500.0],
    },
    "local_generation": {
        "type": "float",
        "scope": "global",
        "causes_downtime": False,
        "description": "On-site generation output (kW)",
        "typical_range": [0.0, 500.0],
    },
    "sim_steps": {
        "type": "int",
        "scope": "global",
        "causes_downtime": False,
        "description": "Simulation steps per trial",
        "typical_range": [10, 200],
    },
}
