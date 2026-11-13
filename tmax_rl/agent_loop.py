from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
)

from .sandbox_tool import (
    LocalSandboxBashTool,
    SandboxInfrastructureError,
)

_FORMAT_ERROR = """Format error: Your last response did not include a valid `bash` tool call.

Please always provide EXACTLY ONE call to the `bash` tool. If you want to
end the task, issue `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` through
the `bash` tool, with no other command.
"""


class TMaxSandboxAgentLoop(ToolAgentLoop):
    """TMax Vanillux loop with a persistent local container per trajectory."""

    def __init__(
        self, *args, max_format_errors: int = 64, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_format_errors = max_format_errors
        bash_tool = self.tools.get("bash")
        if bash_tool is None or not callable(
            getattr(bash_tool, "finalize", None)
        ):
            raise TypeError(
                "TMaxSandboxAgentLoop requires a bash tool with finalize()"
            )

    @property
    def bash_tool(self) -> LocalSandboxBashTool:
        return self.tools["bash"]

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        state = await super()._handle_generating_state(
            agent_data,
            sampling_params,
            ignore_termination=ignore_termination,
        )
        if state is not AgentState.TERMINATED or agent_data.tool_calls:
            return state
        if len(agent_data.response_mask) >= self.response_length:
            return state
        if (
            self.max_assistant_turns
            and agent_data.assistant_turns >= self.max_assistant_turns
        ):
            return state
        if (
            self.max_user_turns
            and agent_data.user_turns >= self.max_user_turns
        ):
            return state

        format_errors = (
            int(agent_data.extra_fields.get("tmax_format_errors", 0)) + 1
        )
        agent_data.extra_fields["tmax_format_errors"] = format_errors
        if format_errors > self.max_format_errors:
            return state

        previous_messages = list(agent_data.messages)
        agent_data.messages.append(
            {"role": "user", "content": _FORMAT_ERROR}
        )
        schemas = getattr(
            agent_data, "_active_tool_schemas", self.tool_schemas
        )
        merge_result, response_mask, response_logprobs = (
            await self.ct_merge_context_msg(
                previous_messages,
                agent_data.messages,
                agent_data.prompt_ids,
                agent_data.response_mask,
                (
                    agent_data.response_logprobs
                    if agent_data.response_logprobs
                    else None
                ),
                tools=schemas,
            )
        )
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        if response_logprobs is not None:
            agent_data.response_logprobs = response_logprobs
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_processing_tools_state(
        self, agent_data: AgentData
    ) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        infra_error = await self.bash_tool.pop_infrastructure_error(
            agent_data.request_id
        )
        if infra_error is not None:
            raise SandboxInfrastructureError(
                f"{type(infra_error).__name__}: {infra_error}"
            ) from infra_error
        if agent_data.extra_fields.get("tmax_submitted"):
            return AgentState.TERMINATED
        return state

    async def run(
        self, sampling_params: dict[str, Any], priority: int = 0, **kwargs
    ):
        create_kwargs = (
            kwargs.get("tools_kwargs", {})
            .get("bash", {})
            .get("create_kwargs", {})
        )
        output = await super().run(
            sampling_params, priority=priority, **kwargs
        )
        output.extra_fields.setdefault("sandbox_request_id", None)
        output.extra_fields.setdefault("sandbox_id", None)
        output.extra_fields.setdefault("sandbox_runtime", None)
        output.extra_fields.setdefault(
            "tmax_task_id", create_kwargs.get("task_id")
        )
        output.extra_fields.setdefault("tmax_submitted", False)
        output.extra_fields.setdefault("tmax_format_errors", 0)
        request_id = output.extra_fields.get("sandbox_request_id")
        reward, verifier_info = await self.bash_tool.finalize(request_id)
        output.reward_score = reward
        output.extra_fields["tmax_verifier"] = verifier_info
        output.extra_fields["tmax_infra_attempt"] = 1
        return output
