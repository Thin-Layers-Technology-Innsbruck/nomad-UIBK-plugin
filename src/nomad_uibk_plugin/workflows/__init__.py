from nomad.orchestrator.base import BaseWorkflowHandler
from nomad.orchestrator.shared.constant import TaskQueue
from pydantic import BaseModel


class IFMLLMEntryPoint(BaseModel):
    entry_point_type: str = 'workflow'

    def load(self):
        from nomad_uibk_plugin.workflows.activities import (
            read_file,
            run_inference,
            write_to_archive,
        )
        from nomad_uibk_plugin.workflows.workflow import InferenceWorkflow

        return BaseWorkflowHandler(
            task_queue=TaskQueue.GPU,
            workflows=[InferenceWorkflow],
            activities=[read_file, run_inference, write_to_archive],
        )


ifmllm = IFMLLMEntryPoint()