from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_uibk_plugin.actions.activities import (
        read_file_and_write_archive,
        run_ifm_inference,
    )
    from nomad_uibk_plugin.actions.shared import (
        InferenceInput,
        WriteArchiveInput,
    )


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput):
        await workflow.execute_activity(
            run_ifm_inference,
            data,
            start_to_close_timeout=timedelta(seconds=600),
            retry_policy=RetryPolicy(
                maximum_attempts=5,
            ),
        )

        input_for_writer = WriteArchiveInput(
            csv_path=data.csv_path,
            upload_id=data.upload_id,
            user_id=data.user_id,
            triggering_entry_id=data.triggering_entry_id,
            sample_id=data.sample_id,
        )

        result_reference = await workflow.execute_activity(
            read_file_and_write_archive,
            input_for_writer,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                maximum_attempts=5,
            ),
        )

        return result_reference
