from typing import TYPE_CHECKING

from nomad.config import config
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity
from nomad.parsing.parser import MatchingParser

from nomad_uibk_plugin.schema_packages.XRFschema import ELNXRayFluorescence
from nomad_uibk_plugin.utils import create_archive

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

configuration = config.get_plugin_entry_point('nomad_uibk_plugin.parsers:xrfparser')


class RawFileXRFData(EntryData):
    """
    Section for an XRF data file.
    """

    measurement = Quantity(
        type=ELNXRayFluorescence,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )


class XRFParser(MatchingParser):
    """
    Parser for matching XRF files and creating instances of XRayFlourescence.
    """

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('XRFParser.parse')
        data_file = mainfile.split('/')[-1]
        entry = ELNXRayFluorescence.m_from_dict(ELNXRayFluorescence.m_def.a_template)
        entry.data_file = data_file
        file_name = f'{"".join(data_file.split(".")[:-1])}.archive.json'
        archive.data = RawFileXRFData(
            measurement=create_archive(entry, archive, file_name)
        )
        archive.metadata.entry_name = f'{data_file} data file'
