from typing import TYPE_CHECKING

from nomad.config import config
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity
from nomad.parsing.parser import MatchingParser
from nomad_measurements.utils import create_archive

from nomad_uibk_plugin.schema_packages.JVschema import UIBK_JVMeasurement

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

configuration = config.get_plugin_entry_point('nomad_uibk_plugin.parsers:jvjsonparser')


class RawFileJVMeasData(EntryData):
    """
    Section for an JV Measurements data .json file.
    """

    measurement = Quantity(
        type=UIBK_JVMeasurement,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )


class JVParser(MatchingParser):
    """
    Parser for matching JV .json files and creating instances of UIBK_JVMeasurement.
    """

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        logger.info('JVParser.parse')
        data_file = mainfile.split('/raw/')[-1]
        archive.metadata.entry_type = 'RawJVMeasurementFile'
        entry = UIBK_JVMeasurement.m_from_dict(UIBK_JVMeasurement.m_def.a_template)
        file_str = ''.join(data_file.split('.')[:-1])
        file_name = f'{file_str}.archive.json'
        create_archive(
            entity=entry,
            archive=archive,
            file_name=file_name,
            overwrite=True,
        )
