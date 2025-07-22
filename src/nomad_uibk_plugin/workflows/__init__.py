from nomad.orchestrator.base import BaseWorkflowHandler
from nomad.orchestrator.shared.constant import TaskQueue
from pydantic import BaseModel


class IFMLLMEntryPoint(BaseModel):
    entry_point_type: str = 'workflow'

    def load(self):
        from nomad_uibk_plugin.workflows.activities import (
            construct_model_input,
            get_model,
            run_inference,
            write_results,
        )
        from nomad_uibk_plugin.workflows.workflow import InferenceWorkflow

        return BaseWorkflowHandler(
            task_queue=TaskQueue.GPU,
            workflows=[InferenceWorkflow],
            activities=[get_model, construct_model_input, run_inference, write_results],
        )


ifmllm = IFMLLMEntryPoint()