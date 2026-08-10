"""
Test bootstrap — stubs the Windmill-runtime imports that don't exist
in the local test environment.
"""
import sys
import types
import logging

# Stub f.breeder.shared.otel_logging
f_mod = types.ModuleType('f')
breeder_mod = types.ModuleType('f.breeder')
shared_mod = types.ModuleType('f.breeder.shared')
otel_mod = types.ModuleType('f.breeder.shared.otel_logging')
engine_mod = types.ModuleType('f.breeder.engine')

def get_logger(name):
    return logging.getLogger(name)

otel_mod.get_logger = get_logger

f_mod.breeder = breeder_mod
breeder_mod.shared = shared_mod
shared_mod.otel_logging = otel_mod
breeder_mod.engine = engine_mod

sys.modules['f'] = f_mod
sys.modules['f.breeder'] = breeder_mod
sys.modules['f.breeder.shared'] = shared_mod
sys.modules['f.breeder.shared.otel_logging'] = otel_mod
sys.modules['f.breeder.engine'] = engine_mod
