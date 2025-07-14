from typing import TYPE_CHECKING

# import plotly.graph_objects as go
from nomad.datamodel.data import (
    # ArchiveSection,
    EntryData,
)
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
# from nomad.datamodel.metainfo.basesections import (
    # CompositeSystem,
    # CompositeSystemReference,
# )
# from nomad.datamodel.metainfo.plot import PlotlyFigure, PlotSection
from nomad.metainfo import Quantity, SchemaPackage, Section

from nomad_uibk_plugin.schema_packages import UIBKCategory

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger


m_package = SchemaPackage()


class EfficienciesSchema(EntryData):
    m_def = Section(
        categories=[UIBKCategory],
        label='UIBK Efficiencies',
    )

    sample_id = Quantity(
        type=str,
        description='ID of the sample.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )

    efficiency = Quantity(
        type=float,
        description='Measured efficiency.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)

































m_package.__init_metainfo__()