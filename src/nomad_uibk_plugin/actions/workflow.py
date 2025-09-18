from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_uibk_plugin.actions.activities import (
        generate_csv_path,
        mask_image,
        read_file_and_write_archive,
        run_ifm_inference,
    )
    from nomad_uibk_plugin.actions.shared import (
        InferenceInput,
        MaskingInput,
        WriteArchiveInput,
    )


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput):
        input_for_masking = MaskingInput(
            input_path=data.image_file_name,
            overwrite_existing_results=data.overwrite_existing_results,
            mask_input_images=data.mask_input_images,
        )

        image_after_masking_name = await workflow.execute_activity(
            mask_image,
            input_for_masking,
            start_to_close_timeout=timedelta(seconds=600),
            retry_policy=RetryPolicy(
                maximum_attempts=5,
            ),
        )

        csv_path = await workflow.execute_activity(
            generate_csv_path,
            image_after_masking_name,
            start_to_close_timeout=timedelta(seconds=600),
            retry_policy=RetryPolicy(
                maximum_attempts=5,
            ),
        )

        data.image_file_name = image_after_masking_name
        data.csv_path = csv_path

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
            sample_id=data.sample_id,
            mask_input_images=data.mask_input_images
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
