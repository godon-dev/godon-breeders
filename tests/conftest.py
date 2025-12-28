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

"""
Pytest configuration for godon unit tests.

Mocks external dependencies to allow testing pure logic functions
without requiring full Windmill/Prometheus client installations.
"""

import sys
from unittest.mock import MagicMock

# Mock external dependencies that production code imports
# These modules are never actually called by the functions we test,
# they're just placeholders to satisfy Python's import system.
sys.modules['wmill'] = MagicMock()
sys.modules['prometheus_api_client'] = MagicMock()
sys.modules['prometheus_api_client.exceptions'] = MagicMock()