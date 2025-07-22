from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad_uibk_plugin.workflows.activities import (
        construct_model_input,
        get_model,
        run_inference,
        write_results,
    )
    from nomad_uibk_plugin.workflows.shared import (
        InferenceInput,
        # InferenceModelInput,
#         InferenceResultsInput,
    )


@workflow.defn(name='nomad_UIBK_plugin.workflows.InferenceWorkflow')
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput):
        await workflow.execute_activity(
            run_inference,
            data,
            start_to_close_timeout=timedelta(seconds=600),
        )