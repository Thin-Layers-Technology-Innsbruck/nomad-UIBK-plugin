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
            generate_csv_path,
            mask_image,
            process_new_files,
            read_file_and_write_archive,
            run_ifm_inference,
        )
        from nomad_uibk_plugin.actions.workflow import IFMInferenceWorkflow

        return Action(
            task_queue=self.task_queue,
            workflow=IFMInferenceWorkflow,
            activities=[
                generate_csv_path,
                mask_image,
                process_new_files,
                read_file_and_write_archive,
                run_ifm_inference,
            ],
        )


ifm_inference = IFMInferenceEntryPoint()
