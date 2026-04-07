import time
from typing import Any, Dict
from uuid import UUID

from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.trace import Span, StatusCode, set_span_in_context

from infra.tracer import tracer


class OpenTelemetryCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self._spans: Dict[str, Span] = {}
        self._timers: Dict[str, float] = {}

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
        span.set_attribute("langchain.parent_run_id", parent_run_id)

        prompt_char_amount = sum(len(prompt) for prompt in prompts)
        span.set_attribute("langchain.llm.prompt_char_amount", prompt_char_amount)

        self._timers[uid] = time.perf_counter()


    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None,
                   tags: list[str] | None = None, **kwargs: Any) -> Any:
        span = self._spans[run_id.hex]
        start_time = self._timers[run_id.hex] or 0
        if span is not None:
            time_count = (time.perf_counter() - start_time) * 1000
            span.set_attribute("langchain.llm.duration_ms", time_count)
            span.set_status(StatusCode.OK)

            if hasattr(response, "llm_output") and response.llm_output is not None:
                token_usage = response.llm_output.token_usage
                if token_usage is not None:
                    span.set_attribute("langchain.llm.token_usage", token_usage)

            span.end(end_time=time.time_ns())

    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None,
                     tags: list[str] | None = None, **kwargs: Any) -> Any:
        span = self._spans[run_id.hex]
        if span is not None:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
            span.end(end_time=time.time_ns())

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *, run_id: UUID,
                            parent_run_id: UUID | None = None, tags: list[str] | None = None,
                            metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return super().on_chat_model_start(serialized, messages, run_id=run_id, parent_run_id=parent_run_id, tags=tags,
                                           metadata=metadata, **kwargs)

    def on_retriever_start(self, serialized: dict[str, Any], query: str, *, run_id: UUID,
                           parent_run_id: UUID | None = None, tags: list[str] | None = None,
                           metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return super().on_retriever_start(serialized, query, run_id=run_id, parent_run_id=parent_run_id, tags=tags,
                                          metadata=metadata, **kwargs)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID,
                      parent_run_id: UUID | None = None, tags: list[str] | None = None,
                      metadata: dict[str, Any] | None = None, inputs: dict[str, Any] | None = None,
                      **kwargs: Any) -> Any:
        return super().on_tool_start(serialized, input_str, run_id=run_id, parent_run_id=parent_run_id, tags=tags,
                                     metadata=metadata, inputs=inputs, **kwargs)

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], *, run_id: UUID,
                       parent_run_id: UUID | None = None, tags: list[str] | None = None,
                       metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if parent_run_id is not None and self._spans.get(parent_run_id.hex) is not None:
            span_context = set_span_in_context(self._spans[parent_run_id.hex])
            span = tracer.start_span(name=run_id.hex,context=span_context)
        else:
            span = tracer.start_span(name=run_id.hex)
        self._spans[run_id.hex] = span

        chain_name = run_id.hex
        span.set_attribute("langchain.component", "chain")
        span.set_attribute("langchain.chain.name", chain_name)
        span.set_attribute("langchain.run_id", run_id.hex)

        if isinstance(inputs, dict):
            span.set_attribute("langchain.input_keys", str(list(inputs.keys())))

        self._timers[run_id.hex] = time.perf_counter()

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None,
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

    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None,
                       **kwargs: Any) -> Any:
        span = self._spans[run_id.hex]
        if span is not None:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
            span.end(end_time=time.time_ns())

    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: UUID | None = None,
                        **kwargs: Any) -> Any:
        return super().on_agent_action(action, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: UUID | None = None,
                        **kwargs: Any) -> Any:
        return super().on_agent_finish(finish, run_id=run_id, parent_run_id=parent_run_id, **kwargs)