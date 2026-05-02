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
import types
from unittest.mock import MagicMock

sys.modules['wmill'] = MagicMock()
sys.modules['prometheus_api_client'] = MagicMock()
sys.modules['prometheus_api_client.exceptions'] = MagicMock()
sys.modules['prometheus_client'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


class FakeModule:
    def __init__(self, name='unknown'):
        self.__path__ = []
        self.__spec__ = None
        self.__name__ = name


fake_f = MagicMock()
fake_breeder = MagicMock()
fake_engine = FakeModule('f.breeder.engine')
fake_strains = MagicMock()
fake_strains_lnx_perf = FakeModule('f.breeder.strains.linux_performance')
fake_strains_bench_gh = FakeModule('f.breeder.strains.bench_greenhouse')
fake_breeder_shared = MagicMock()
fake_f.breeder = fake_breeder
fake_breeder.engine = fake_engine
fake_breeder.strains = fake_strains
fake_strains.linux_performance = fake_strains_lnx_perf
fake_strains.bench_greenhouse = fake_strains_bench_gh
fake_breeder.shared = fake_breeder_shared
sys.modules['f'] = fake_f
sys.modules['f.breeder'] = fake_breeder
sys.modules['f.breeder.engine'] = fake_engine
sys.modules['f.breeder.strains'] = fake_strains
sys.modules['f.breeder.strains.linux_performance'] = fake_strains_lnx_perf
sys.modules['f.breeder.strains.bench_greenhouse'] = fake_strains_bench_gh
sys.modules['f.breeder.shared'] = fake_breeder_shared

fake_otel = MagicMock()
fake_otel.get_logger = lambda name: MagicMock()
sys.modules['f.breeder.shared.otel_logging'] = fake_otel

# Pre-populate engine module stubs
for module_name in ['breeder_worker', 'breeder_metrics_client', 'communication', 'strain_loader']:
    stub = FakeModule()
    sys.modules[f'f.breeder.engine.{module_name}'] = stub

# Pre-populate strain module stubs
for module_name in ['parameter_registry', 'preflight', 'strain']:
    stub = FakeModule()
    sys.modules[f'f.breeder.strains.linux_performance.{module_name}'] = stub

for module_name in ['parameter_registry', 'preflight', 'strain']:
    stub = FakeModule()
    sys.modules[f'f.breeder.strains.bench_greenhouse.{module_name}'] = stub


def populate_stub_module(stub_module, source_module):
    for attr_name in dir(source_module):
        if not attr_name.startswith('_'):
            setattr(stub_module, attr_name, getattr(source_module, attr_name))


# Import engine modules and populate stubs (in dependency order)
import engine.breeder_metrics_client as breeder_metrics_client
populate_stub_module(sys.modules['f.breeder.engine.breeder_metrics_client'], breeder_metrics_client)

import engine.communication as communication
populate_stub_module(sys.modules['f.breeder.engine.communication'], communication)

import engine.strain_loader as strain_loader
populate_stub_module(sys.modules['f.breeder.engine.strain_loader'], strain_loader)

# Import strain modules and populate stubs
import strains.linux_performance.parameter_registry as parameter_registry
populate_stub_module(sys.modules['f.breeder.strains.linux_performance.parameter_registry'], parameter_registry)

import strains.linux_performance.preflight as preflight
populate_stub_module(sys.modules['f.breeder.strains.linux_performance.preflight'], preflight)

import strains.linux_performance.strain as strain
populate_stub_module(sys.modules['f.breeder.strains.linux_performance.strain'], strain)

import strains.bench_greenhouse.parameter_registry as gh_parameter_registry
populate_stub_module(sys.modules['f.breeder.strains.bench_greenhouse.parameter_registry'], gh_parameter_registry)

import strains.bench_greenhouse.preflight as gh_preflight
populate_stub_module(sys.modules['f.breeder.strains.bench_greenhouse.preflight'], gh_preflight)

import strains.bench_greenhouse.strain as gh_strain
populate_stub_module(sys.modules['f.breeder.strains.bench_greenhouse.strain'], gh_strain)

sys.modules['f.breeder.engine.watermark'] = types.ModuleType('f.breeder.engine.watermark')
sys.modules['f.breeder.engine.watermark'].create_watermark = lambda *a, **kw: None
sys.modules['f.breeder.engine.watermark'].Watermark = type('Watermark', (), {})

import engine.breeder_worker as breeder_worker
populate_stub_module(sys.modules['f.breeder.engine.breeder_worker'], breeder_worker)
