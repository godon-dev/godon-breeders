#extra_requirements:
#opentelemetry-api
#opentelemetry-sdk
#opentelemetry-exporter-otlp

import importlib
from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)

STRAIN_MODULES = {
    "linux_performance": "f.breeder.strains.linux_performance.strain",
}


def load_strain(strain_type: str):
    logger.info(f"Loading strain: {strain_type}")

    module_path = STRAIN_MODULES.get(strain_type)
    if not module_path:
        raise ValueError(
            f"Unknown strain type: '{strain_type}'. "
            f"Available: {', '.join(STRAIN_MODULES.keys())}"
        )

    try:
        module = importlib.import_module(module_path)
        logger.info(f"Loaded strain module: {module_path}")
        return module
    except ImportError as e:
        logger.error(f"Failed to import strain '{strain_type}': {e}")
        raise
