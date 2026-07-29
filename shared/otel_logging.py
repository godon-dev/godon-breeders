
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
#requirements:
#opentelemetry-api
#opentelemetry-sdk
#opentelemetry-exporter-otlp

import os
import logging
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

OTEL_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://godon-observability-opentelemetry-collector.godon-observability.svc.cluster.local:4318"
)

_service_name = os.environ.get("OTEL_SERVICE_NAME", "godon-breeders")
_tracer = None
_logger_provider = None
_initialized = False


def init_telemetry(service_name: str = None):
    global _tracer, _logger_provider, _initialized, _service_name
    
    if _initialized:
        return _tracer, _logger_provider
    
    if service_name:
        _service_name = service_name
    
    try:
        resource = Resource.create({"service.name": _service_name})
        
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTEL_ENDPOINT}/v1/traces"))
        )
        try:
            trace.set_tracer_provider(tracer_provider)
        except Exception as e:
            logging.getLogger(__name__).warning(f"OTLP tracer provider already set: {e}")
        _tracer = trace.get_tracer(_service_name)
        
        _logger_provider = LoggerProvider(resource=resource)
        _logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{OTEL_ENDPOINT}/v1/logs"))
        )
        
        _initialized = True
    except Exception as e:
        logging.getLogger(__name__).error(f"OTLP telemetry init failed: {e}", exc_info=True)
    
    return _tracer, _logger_provider


def get_logger(name: str, service_name: str = None) -> logging.Logger:
    init_telemetry(service_name)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not any(isinstance(h, LoggingHandler) for h in logger.handlers):
        handler = LoggingHandler(logger_provider=_logger_provider)
        logger.addHandler(handler)
    
    return logger


def get_tracer(service_name: str = None):
    init_telemetry(service_name)
    return _tracer
