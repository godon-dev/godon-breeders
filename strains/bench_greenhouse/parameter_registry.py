
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
PARAMETER_REGISTRY = {
    "heating_setpoints": {
        "type": "float",
        "scope": "per_zone",
        "causes_downtime": False,
        "description": "Target temperature per zone (Celsius)",
        "typical_range": [5.0, 40.0],
    },
    "vent_openings": {
        "type": "float",
        "scope": "per_zone",
        "causes_downtime": False,
        "description": "Ventilation opening fraction per zone (0.0-1.0)",
        "typical_range": [0.0, 1.0],
    },
    "shading": {
        "type": "float",
        "scope": "global",
        "causes_downtime": False,
        "description": "Shading coefficient (0.0-1.0)",
        "typical_range": [0.0, 1.0],
    },
    "co2_injection": {
        "type": "float",
        "scope": "global",
        "causes_downtime": False,
        "description": "CO2 injection rate",
        "typical_range": [0.0, 20.0],
    },
    "light_intensity": {
        "type": "float",
        "scope": "global",
        "causes_downtime": False,
        "description": "Supplemental light intensity",
        "typical_range": [0.0, 1000.0],
    },
    "irrigation": {
        "type": "float",
        "scope": "global",
        "causes_downtime": False,
        "description": "Irrigation rate (0.0-3.0)",
        "typical_range": [0.0, 3.0],
    },
    "sim_steps": {
        "type": "int",
        "scope": "global",
        "causes_downtime": False,
        "description": "Simulation steps per trial",
        "typical_range": [10, 200],
    },
}
