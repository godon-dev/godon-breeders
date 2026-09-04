
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
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from engine.strain_loader import load_strain, _validate_strain


class TestLoadStrain:
    def test_loads_linux_performance_strain(self):
        mock_module = MagicMock()
        mock_module.suggest_params = lambda t, s: {}
        mock_module.validate_config = lambda c: {}

        with patch('engine.strain_loader.importlib.import_module', return_value=mock_module):
            result = load_strain('linux_performance')
            assert result == mock_module

    def test_unknown_strain_type_raises(self):
        with pytest.raises(ValueError, match="Unknown strain type"):
            load_strain('nonexistent_strain')

    def test_import_error_propagates(self):
        with patch('engine.strain_loader.importlib.import_module', side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="No module"):
                load_strain('linux_performance')


class TestValidateStrain:
    def test_valid_module_passes(self):
        module = MagicMock()
        module.suggest_params = lambda: None
        module.validate_config = lambda: None
        _validate_strain(module, 'test_strain')

    def test_missing_suggest_params_raises(self):
        module = MagicMock(spec=[])
        with pytest.raises(ValueError, match="missing required attributes"):
            _validate_strain(module, 'broken_strain')

    def test_missing_validate_config_raises(self):
        module = MagicMock()
        del module.validate_config
        with pytest.raises(ValueError, match="missing required attributes"):
            _validate_strain(module, 'broken_strain')

    def test_missing_both_raises(self):
        module = MagicMock(spec=[])
        with pytest.raises(ValueError, match="suggest_params") as exc_info:
            _validate_strain(module, 'broken_strain')
        assert 'validate_config' in str(exc_info.value)
