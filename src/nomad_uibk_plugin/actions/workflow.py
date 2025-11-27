from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_uibk_plugin.actions.activities import (
        generate_paths,
        mask_image,
        process_new_files,
        read_file_and_write_archive,
        run_ifm_inference,
    )
    from nomad_uibk_plugin.actions.shared import (
        ActivityInferenceInput,
        InferenceInput,
        MaskingInput,
        ProcessNewFilesInput,
        WriteArchiveInput,
    )

MAXIMUM_ATTEMPTS = 3
TIMEOUT_LARGE = 3600
TIMEOUT_SMALL = 300


@workflow.defn
class IFMInferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput):
        input_num = len(data.sample_id)
        if (
            (input_num != len(data.image_file_name))
            or (input_num != len(data.csv_path))
            or (input_num != len(data.output_path))
            or (input_num == 0)
        ):
            return None
        else:
            result_path = []
            for i in range(input_num):
                input_for_masking = MaskingInput(
                    input_path=data.image_file_name[i],
                    overwrite_existing_results=data.overwrite_existing_results,
                    mask_input_images=data.mask_input_images,
                )

                image_after_masking_name = await workflow.execute_activity(
                    mask_image,
                    input_for_masking,
                    start_to_close_timeout=timedelta(seconds=TIMEOUT_LARGE),
                    retry_policy=RetryPolicy(
                        maximum_attempts=MAXIMUM_ATTEMPTS,
                    ),
                )

                output_path, csv_path, h5_path = await workflow.execute_activity(
                    generate_paths,
                    image_after_masking_name,
                    start_to_close_timeout=timedelta(seconds=TIMEOUT_SMALL),
                    retry_policy=RetryPolicy(
                        maximum_attempts=MAXIMUM_ATTEMPTS,
                    ),
                )

                data.image_file_name[i] = image_after_masking_name
                data.csv_path[i] = csv_path
                data.h5_path[i] = h5_path
                data.output_path[i] = output_path

                data_inference_run = ActivityInferenceInput(
                    image_file_name=data.image_file_name[i],
                    model_name=data.model_name,
                    pixel_size=data.pixel_size[i],
                    csv_path=data.csv_path[i],
                    h5_path=data.h5_path[i],
                    output_path=data.output_path[i],
                    overwrite_existing_results=data.overwrite_existing_results,
                    mask_input_images=data.mask_input_images,
                    save_resulting_image=data.save_resulting_image,
                )

                await workflow.execute_activity(
                    run_ifm_inference,
                    data_inference_run,
                    start_to_close_timeout=timedelta(seconds=TIMEOUT_LARGE),
                    retry_policy=RetryPolicy(
                        maximum_attempts=MAXIMUM_ATTEMPTS,
                    ),
                )

                input_for_writer = WriteArchiveInput(
                    pixel_size=data.pixel_size[i],
                    csv_path=data.csv_path[i],
                    h5_path=data.h5_path[i],
                    output_path=data.output_path[i],
                    upload_id=data.upload_id,
                    user_id=data.user_id,
                    sample_id=data.sample_id[i],
                    mask_input_images=data.mask_input_images,
                )

                result_path_this_input = await workflow.execute_activity(
                    read_file_and_write_archive,
                    input_for_writer,
                    start_to_close_timeout=timedelta(seconds=TIMEOUT_LARGE),
                    retry_policy=RetryPolicy(
                        maximum_attempts=MAXIMUM_ATTEMPTS,
                    ),
                )

                result_path.append(result_path_this_input)

            input_for_process = ProcessNewFilesInput(
                upload_id=data.upload_id,
                user_id=data.user_id,
                result_path=result_path,
            )

            result_refs = await workflow.execute_activity(
                process_new_files,
                input_for_process,
                start_to_close_timeout=timedelta(seconds=TIMEOUT_SMALL),
                retry_policy=RetryPolicy(
                    maximum_attempts=MAXIMUM_ATTEMPTS,
                ),
            )

            return {'refs': result_refs}
