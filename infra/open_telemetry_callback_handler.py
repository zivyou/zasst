"""open-telemetry observation tools"""
import time
from typing import Any, Dict
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from opentelemetry.trace import Span, StatusCode, set_span_in_context

from infra.tracer import tracer


class OpenTelemetryCallbackHandler(BaseCallbackHandler):
    """callback handler for observation"""
    def __init__(self):
        self._spans: Dict[str, Span] = {}
        self._timers: Dict[str, float] = {}

    # pylint: disable=too-many-arguments
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID,
                     parent_run_id: UUID | None = None, tags: list[str] | None = None,
                     metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        uid = run_id.hex
        if parent_run_id is not None and self._spans.get(parent_run_id.hex) is not None:
            span_context = set_span_in_context(self._spans[parent_run_id.hex])
            span = tracer.start_span(name=uid,context=span_context)
        else:
            span = tracer.start_span(name=uid)
        self._spans[uid] = span

        llm_name = serialized["name"]
        span.set_attribute("langchain.component", "llm")
        span.set_attribute("langchain.llm.name", llm_name)
        span.set_attribute("langchain.run_id", uid)
        span.set_attribute("langchain.parent_run_id", parent_run_id.hex)

        prompt_char_amount = sum(len(prompt) for prompt in prompts)
        span.set_attribute("langchain.llm.prompt_char_amount", prompt_char_amount)

        self._timers[uid] = time.perf_counter()

    # pylint: disable=too-many-arguments
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None,
                   tags: list[str] | None = None, **kwargs: Any) -> Any:
        span = self._spans[run_id.hex]
        start_time = self._timers[run_id.hex] or 0
        if span is not None:
            time_count = (time.perf_counter() - start_time) * 1000
            span.set_attribute("langchain.llm.duration_ms", time_count)
            span.set_status(StatusCode.OK)
            keys = str(response.llm_output.keys())
            span.set_attribute("langchain.llm.output_keys", keys)

            if hasattr(response, "llm_output") and response.llm_output is not None:
                span.set_attribute("langchain.llm.llm_ouput", str(response.llm_output))

            span.end(end_time=time.time_ns())

    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None,
                     tags: list[str] | None = None, **kwargs: Any) -> Any:
        span = self._spans[run_id.hex]
        if span is not None:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
            span.end(end_time=time.time_ns())


    # pylint: disable=too-many-arguments
    def on_chain_start(self, serialized: dict[str, Any],
                       inputs: dict[str, Any], *, run_id: UUID,
                       parent_run_id: UUID | None = None, tags: list[str] | None = None,
                       metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if parent_run_id is not None and self._spans.get(parent_run_id.hex) is not None:
            span_context = set_span_in_context(self._spans[parent_run_id.hex])
            span = tracer.start_span(name=run_id.hex,context=span_context)
        else:
            span = tracer.start_span(name=run_id.hex)
        self._spans[run_id.hex] = span

        if kwargs is not None and kwargs["name"] is not None:
            chain_name = kwargs["name"]
            for key, value in kwargs.items():
                span.set_attribute(key, value)
        else:
            chain_name = run_id.hex
        span.set_attribute("langchain.component", "chain")
        span.set_attribute("langchain.chain.name", chain_name)
        span.set_attribute("langchain.run_id", run_id.hex)

        if isinstance(inputs, dict):
            span.set_attribute("langchain.input_keys", str(list(inputs.keys())))

        self._timers[run_id.hex] = time.perf_counter()

    def on_chain_end(self, outputs: dict[str, Any], *,
                     run_id: UUID, parent_run_id: UUID | None = None,
                     **kwargs: Any) -> Any:
        span = self._spans[run_id.hex]
        start_time = self._timers[run_id.hex] or 0
        if span is not None:
            time_count = (time.perf_counter() - start_time) * 1000
            span.set_attribute("langchain.chain.duration_ms", time_count)
            span.set_status(StatusCode.OK)

            if isinstance(outputs, dict):
                span.set_attribute("langchain.output_keys", str(list(outputs.keys())))
            span.end(end_time=time.time_ns())

    # pylint: disable=too-many-arguments
    def on_chain_error(self, error: BaseException, *,
                       run_id: UUID, parent_run_id: UUID | None = None,
                       **kwargs: Any) -> Any:
        span = self._spans[run_id.hex]
        if span is not None:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
            span.end(end_time=time.time_ns())
