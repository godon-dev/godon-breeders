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

import sys
import os
from unittest.mock import MagicMock

sys.modules['wmill'] = MagicMock()
sys.modules['prometheus_api_client'] = MagicMock()
sys.modules['prometheus_api_client.exceptions'] = MagicMock()
sys.modules['prometheus_client'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

class FakeBreederModule:
    def __init__(self):
        self.__path__ = []
        self.__spec__ = None
        self.__name__ = 'f.breeder.linux_performance'

fake_f = MagicMock()
fake_breeder = MagicMock()
fake_linux_performance = FakeBreederModule()
fake_shared = MagicMock()
fake_f.breeder = fake_breeder
fake_breeder.linux_performance = fake_linux_performance
fake_f.shared = fake_shared
sys.modules['f'] = fake_f
sys.modules['f.breeder'] = fake_breeder
sys.modules['f.breeder.linux_performance'] = fake_linux_performance
sys.modules['f.shared'] = fake_shared

fake_otel = MagicMock()
fake_otel.get_logger = lambda name: MagicMock()
sys.modules['f.shared.otel_logging'] = fake_otel

# Pre-populate breeder modules BEFORE any imports
for module_name in ['breeder_worker', 'breeder_metrics_client', 'parameter_registry', 'preflight']:
    stub = FakeBreederModule()
    sys.modules[f'f.breeder.linux_performance.{module_name}'] = stub

# Now import the actual modules and populate the stubs with their contents
def populate_stub_module(stub_module, source_module):
    """Copy attributes from source_module to stub_module"""
    for attr_name in dir(source_module):
        if not attr_name.startswith('_'):
            setattr(stub_module, attr_name, getattr(source_module, attr_name))

# Import breeder modules and populate stubs (in dependency order)
import linux_performance.breeder_metrics_client as breeder_metrics_client
populate_stub_module(sys.modules['f.breeder.linux_performance.breeder_metrics_client'], breeder_metrics_client)

import linux_performance.parameter_registry as parameter_registry
populate_stub_module(sys.modules['f.breeder.linux_performance.parameter_registry'], parameter_registry)

import linux_performance.preflight as preflight
populate_stub_module(sys.modules['f.breeder.linux_performance.preflight'], preflight)

import linux_performance.breeder_worker as breeder_worker
populate_stub_module(sys.modules['f.breeder.linux_performance.breeder_worker'], breeder_worker)
