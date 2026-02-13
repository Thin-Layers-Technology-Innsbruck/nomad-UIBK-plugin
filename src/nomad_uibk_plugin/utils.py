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

from nomad.datamodel.metainfo.basesections import SectionReference

from nomad_uibk_plugin.schema_packages.sample import (
    SampleActivities,
    UIBKSample,
    UIBKSampleReference,
)

if TYPE_CHECKING:
    from nomad.datamodel.data import ArchiveSection
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger


def get_reference(upload_id: str, entry_id: str) -> str:
    return f'../uploads/{upload_id}/archive/{entry_id}#data'


def get_entry_id_from_file_name(file_name: str, archive: 'EntryArchive') -> str:
    from nomad.utils import hash

    return hash(archive.metadata.upload_id, file_name)


def create_archive(
    entity: 'ArchiveSection',
    archive: 'EntryArchive',
    file_name: str,
    overwrite: bool = False,
    reprocess: bool = False,
) -> str:
    if overwrite or not archive.m_context.raw_path_exists(file_name):
        with archive.m_context.update_entry(
            file_name, write=True, process=True
        ) as entry:
            entry['data'] = entity.m_to_dict(with_root_def=True)
    elif reprocess:
        with archive.m_context.update_entry(
            file_name, write=True, process=True
        ) as entry:
            pass  # just trigger reprocessing
    return get_reference(
        archive.metadata.upload_id, get_entry_id_from_file_name(file_name, archive)
    )


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


def update_backward_refs_in_sample(  # noqa: PLR0913
    entry_id: str,
    upload_id: str,
    archive: 'EntryArchive',
    sample_file_name: str,
    activity_type: str | None,
    activity_name: str | None,
) -> bool:
    """
    Here entry_id and upload_id refer to the sample entry to be updated; archive is
    from the activity that is referencing the sample.
    Might create race condition (multiple activities referring to the same sample
    entry), use carefully!!!

    returns `False` if no upload found or authorization failed, `True` on success
    """

    from nomad.processing.data import Upload

    user_id = archive.metadata.authors[0].user_id  # type: ignore
    upload_with_sample = Upload.get(upload_id)

    if upload_with_sample is None:
        return False
    is_coauthor = (
        isinstance(upload_with_sample.coauthors, list)
        and user_id in upload_with_sample.coauthors
    )
    is_authorized = upload_with_sample.main_author == user_id or is_coauthor
    if not is_authorized:
        return False

    new_activity_ref = get_reference(
        archive.metadata.upload_id,
        archive.metadata.entry_id,  # pyright: ignore[reportArgumentType]
    )
    sample_archive = archive.m_context.load_archive(
        entry_id, upload_id, archive.m_context.installation_url
    )
    sample_entry_needs_reprocessing = False
    with sample_archive.m_context.update_entry(
        sample_file_name, write=True, process=False
    ) as sample_entry:
        if 'activities_performed' not in sample_entry['data']:
            sample_entry['data']['activities_performed'] = {}
            sample_entry_needs_reprocessing = True
        if activity_type not in sample_entry['data']['activities_performed']:
            sample_entry['data']['activities_performed'][activity_type] = []
            sample_entry_needs_reprocessing = True
        # check if activity already listed
        for old_activity in sample_entry['data']['activities_performed'][activity_type]:
            if old_activity['reference'] == new_activity_ref:
                break
        else:
            sample_entry['data']['activities_performed'][activity_type].append(
                {
                    'name': activity_name,
                    'reference': new_activity_ref,
                }
            )
            sample_entry_needs_reprocessing = True
    if sample_entry_needs_reprocessing:
        upload_with_sample.process_updated_raw_file(sample_file_name, allow_modify=True)

    return True


def update_sample_refs(  # noqa: PLR0913
    sample: UIBKSampleReference,
    archive: 'EntryArchive',
    logger: 'BoundLogger',
    activity_type: str | None = None,
    activity_name: str | None = None,
    update_backward_refs: bool = False,
) -> None:
    """
    If sample reference has lab-id but no reference,
    search for sample entry by id, create a new one if none found

    Returns name and reference for the sample reference class
    """
    from nomad.search import MetadataPagination, search

    if update_backward_refs and (activity_name is None or activity_type is None):
        update_backward_refs = False
        logger.warn(
            'missing activity_name or activity_type, will not update backward refs'
        )

    sample_id = str(sample.lab_id)
    # TODO check that entry is UIBKSample or inherits from it in the query ?
    query = {
        'results.eln.lab_ids': sample_id,
    }
    new_activity_ref = get_reference(
        archive.metadata.upload_id,
        archive.metadata.entry_id,  # pyright: ignore[reportArgumentType]
    )
    # find existing sample entry
    search_result = search(
        owner='all',
        query=query,  # type: ignore
        pagination=MetadataPagination(page_size=1),
        user_id=archive.metadata.main_author.user_id,  # type: ignore
    )
    # create new sample entry if none found
    if search_result.pagination.total == 0:
        if update_backward_refs:
            new_sample = UIBKSample(
                lab_id=sample_id,
                activities_performed=SampleActivities(
                    **{
                        activity_type: [
                            SectionReference(
                                name=activity_name, reference=new_activity_ref
                            )
                        ]
                    }  # pyright: ignore[reportArgumentType]
                ),
            )
        else:
            new_sample = UIBKSample(lab_id=sample_id)
        sample_file_name = f'Sample_{sample_id}.archive.json'
        sample_ref = create_archive(
            entity=new_sample,
            archive=archive,
            file_name=sample_file_name,
            overwrite=True,
        )
    else:
        entry_id = search_result.data[0]['entry_id']
        upload_id = search_result.data[0]['upload_id']
        sample_ref = f'../uploads/{upload_id}/archive/{entry_id}#data'
        try:
            sample_file_name = search_result.data[0]['mainfile']
        except Exception as e:
            sample_file_name = f'Sample_{sample_id}.archive.json'
            logger.warn(f'Found no sample file name, using sample id instead: {e}')
        if search_result.pagination.total > 1:
            logger.warn(
                f'Found {search_result.pagination.total} entries with '
                f'lab_id: "{sample_id}". Will use the first one found.'
            )
        # update activities performed in sample entry
        # potential race conditions if multiple activities refer to the same sample
        if update_backward_refs:
            back_refs_status = update_backward_refs_in_sample(
                entry_id,
                upload_id,
                archive,
                sample_file_name,
                activity_type,
                activity_name,
            )
            if not back_refs_status:
                logger.warn(
                    'Could not update backward references in sample entry with id '
                    + f'{entry_id} and upload id {upload_id} - no upload found or '
                    + 'authorization failed.'
                )

    sample.name = sample_file_name
    sample.reference = sample_ref
