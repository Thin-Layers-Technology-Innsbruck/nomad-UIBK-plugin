from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_uibk_plugin.workflows.activities import (
        read_file,
        run_inference,
        write_to_archive,
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
            retry_policy=RetryPolicy(
                maximum_attempts=5,
            )
        )
        result_from_csv = await workflow.execute_activity(
            read_file,
            data.csv_path,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                maximum_attempts=5,
            )
        )
        result_from_csv["user_id"] = data.user_id
        result_from_csv["upload_id"] = data.upload_id
        await workflow.execute_activity(
            write_to_archive,
            result_from_csv,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                maximum_attempts=5,
            )
        )
