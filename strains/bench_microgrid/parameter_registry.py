
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
