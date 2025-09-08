import re
from typing import TYPE_CHECKING

from nomad.config import config
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity
from nomad.parsing.parser import MatchingParser
from nomad_measurements.utils import create_archive

from nomad_uibk_plugin.schema_packages.IFMschema_extra import IFMMeasurement

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

configuration = config.get_plugin_entry_point('nomad_uibk_plugin.parsers:xrfparser')


class RawFileIFMMeasData(EntryData):
    """
    Section for an IFM Measurements data file.
    """

    measurement = Quantity(
        type=IFMMeasurement,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )


class IFMParser(MatchingParser):
    """
    Parser for matching IFM .bmp files and creating instances of IFMMeasurement.
    """

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        print('1')
        logger.info('IFMParser.parse')
        data_file = mainfile.split('/')[-1]
        entry = IFMMeasurement.m_from_dict(IFMMeasurement.m_def.a_template)
        # entry.data_file = data_file
        file_str = "".join(data_file.split(".")[:-1])
        if data_file.split(".")[-1] == 'bmp':
            print('2')
            file_name = f'{file_str}.archive.json'
        else:
            print('3')
            file_str = re.sub(r'_info$', '', file_str)
            file_name = f'{file_str}.archive.json'
        # entry.image_file = data_file
        # archive.data = ElnParserRawFile()
        create_archive(
            entity=entry,
            archive=archive,
            file_name=file_name,
            overwrite=True,
        )
        # archive.data = RawFileIFMMeasData(
        #     measurement = create_archive(entry, archive, file_name)
        # )
        # archive.metadata.entry_name = f'{data_file} data file'
