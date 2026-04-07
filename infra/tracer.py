import os

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# https://oneuptime.com/blog/post/2026-02-06-monitor-langchain-opentelemetry/view


OLTP_ENDPOINT= os.getenv("OLTP_ENDPOINT") or "http://127.0.0.1:4318"

service_name = os.getenv("SERVICE_NAME") or "zasst"
resource = Resource.create({
    "service.name": service_name,
    "service.version": os.getenv("SERVICE_VERSION") or "0.0.1",
    "deployment.environment": os.getenv("ENV") or "dev",
})

provider = TracerProvider(resource=resource)
provider.add_span_processor(
    BatchSpanProcessor(span_exporter=OTLPSpanExporter(endpoint=OLTP_ENDPOINT+"/v1/traces"))
)

trace.set_tracer_provider(provider)

metrics_reader = PeriodicExportingMetricReader(
    exporter=OTLPMetricExporter(endpoint=OLTP_ENDPOINT+"/v1/metrics"),
    export_interval_millis=30000,
)

metrics.set_meter_provider(
    meter_provider=MeterProvider(
        resource=resource,
        metric_readers=[metrics_reader],
    )
)

tracer = trace.get_tracer(instrumenting_module_name=service_name)
meter = metrics.get_meter(name=service_name)