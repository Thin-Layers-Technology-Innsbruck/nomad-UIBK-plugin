from nomad.actions import TaskQueue
from pydantic import Field
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class IFMInferenceEntryPoint(ActionEntryPoint):
    """
    Entry point for the nomad-crystallm inference action.
    """

    task_queue: str = Field(
        default=TaskQueue.CPU, description='Determines the task queue for this action'
    )

    def load(self):
        from nomad.actions import Action

        from nomad_uibk_plugin.actions.activities import (
            read_file,
            run_ifm_inference,
            write_to_archive,
        )
        from nomad_uibk_plugin.actions.workflow import InferenceWorkflow

        return Action(
            task_queue=self.task_queue,
            workflow=InferenceWorkflow,
            activities=[read_file, run_ifm_inference, write_to_archive],
        )


ifm_inference = IFMInferenceEntryPoint()
