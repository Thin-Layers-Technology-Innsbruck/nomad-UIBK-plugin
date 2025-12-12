#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
from typing import TYPE_CHECKING

from nomad_measurements.utils import create_archive

from nomad_uibk_plugin.schema_packages.sample import UIBKSample, UIBKSampleReference

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger


def find_reference_by_id(
    target_id: str, target_type: str, archive: 'EntryArchive', logger: 'BoundLogger'
) -> str | None:
    """
    Extracts the target references from the metadata.

    Parameters:
        target_id: The ID of the target.
        target_type: The type of the target.
        archive: The archive containing the metadata.
        logger: The logger object.

    Returns:
        A reference to the target.
    """

    from nomad.search import search

    if target_id is None or target_type is None or archive is None:
        logger.warning('Target ID, target type, or archive not provided.')
        return None

    # search for target entry in database
    search_result = search(
        owner='all',
        query={
            'results.eln.sections:any': [target_type],
            'results.eln.lab_ids:any': [target_id],
        },
        user_id=archive.metadata.main_author.user_id,  # type: ignore
    )

    # Logger checks
    if not search_result.data:
        logger.warning(f'{target_type} entry with {target_id} not found in database.')
        return None
    else:
        if len(search_result.data) > 1:
            logger.warning(
                f'Multiple {target_type} entries found for ID {target_id}.'
                f'Using the first one.'
            )

        # create reference string
        entry_id = search_result.data[0]['entry_id']
        upload_id = search_result.data[0]['upload_id']

        return f'../uploads/{upload_id}/archive/{entry_id}#data'


def safe_float(input) -> float | None:
    "like float(), but returns None if conversion is not possible or results in NaN"
    try:
        output = float(input)
    except (ValueError, TypeError):
        return None
    if math.isnan(output):
        return None
    else:
        return output


def update_sample_refs(
    sample: UIBKSampleReference, archive: 'EntryArchive', logger: 'BoundLogger'
) -> tuple[str | None, str | None]:
    """
    If sample reference has lab-id but no reference,
    search for sample entry by id, create a new one if none found

    Returns name and reference for the sample reference class
    """
    if sample.lab_id and not sample.reference:
        from nomad.search import MetadataPagination, search

        query = {'results.eln.lab_ids': sample.lab_id}
        search_result = search(
            owner='all',
            query=query,  # type: ignore
            pagination=MetadataPagination(page_size=1),
            user_id=archive.metadata.main_author.user_id,  # type: ignore
        )
        if search_result.pagination.total == 0:
            new_sample = UIBKSample(lab_id=sample.lab_id)
            sample_file_name = f'Sample_{sample.lab_id}.archive.json'
            sample_ref = create_archive(
                entity=new_sample,
                archive=archive,
                file_name=sample_file_name,
                overwrite=False,
            )
        else:
            entry_id = search_result.data[0]['entry_id']
            upload_id = search_result.data[0]['upload_id']
            sample_ref = f'../uploads/{upload_id}/archive/{entry_id}#data'
            try:
                sample_file_name = search_result.data[0]['results']['eln']['names'][0]
            except Exception as e:
                sample_file_name = str(sample.lab_id)
                logger.warn(f'Found no sample name, using sample id instead: {e}')
            if search_result.pagination.total > 1:
                logger.warn(
                    f'Found {search_result.pagination.total} entries with '
                    f'lab_id: "{sample.lab_id}". Will use the first one found.'
                )
    else:
        sample_file_name = None
        sample_ref = None
    return sample_file_name, sample_ref
