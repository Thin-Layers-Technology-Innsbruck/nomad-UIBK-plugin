from typing import TYPE_CHECKING

import plotly.graph_objects as go
from nomad.datamodel.data import (
    ArchiveSection,
    EntryData,
)
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.basesections import (
    CompositeSystem,
    CompositeSystemReference,
)
from nomad.datamodel.metainfo.plot import PlotlyFigure, PlotSection
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

from nomad_uibk_plugin.schema_packages import UIBKCategory

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger


m_package = SchemaPackage()


class MicroCell(CompositeSystem):
    m_def = Section()

    x = Quantity(
        type=float,
        description='X position of the microcell on the sample',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )
    y = Quantity(
        type=float,
        description='Y position of the microcell on the sample',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )


class ArrayGenerator(ArchiveSection):
    create_array = Quantity(
        type=bool,
        default=False,
        label='Create a new microcell array?',
        description='Check this box to trigger the generation of a new microcell array',
        a_eln=ELNAnnotation(component=ELNComponentEnum.BoolEditQuantity),
    )

    x = Quantity(
        type=int,
        default=0,
        label='Number of microcells in x direction',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )

    y = Quantity(
        type=int,
        default=0,
        label='Number of microcells in y direction',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )


class MicroCellArray(ArchiveSection):
    cells = SubSection(section_def=MicroCell, label='MicroCells', repeats=True)


class AlternativeID(ArchiveSection):
    m_def = Section(label_quantity='alternative_id')

    alternative_id = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    location = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    description = Quantity(type=str, a_eln=dict(component='RichTextEditQuantity'))

    url_to_resource = Quantity(type=str, a_eln=dict(component='URLEditQuantity'))


class UIBKSample(CompositeSystem, EntryData):
    m_def = Section(
        categories=[UIBKCategory],
        label='UIBK Sample',
    )

    alternative_id = SubSection(
        section_def=AlternativeID,
        description="""An alternative ID for the sample used in the different lab,
            unique for that lab""",
        repeats=True,
    )

    def normalize(self, archive, logger: 'BoundLogger'):
        super().normalize(archive, logger)
        archive.metadata.entry_name = self.name

        for alt_id in self.alternative_id:
            if alt_id not in archive.results.eln.lab_ids:
                archive.results.eln.lab_ids.append(alt_id.alternative_id)  # pyright: ignore[reportAttributeAccessIssue]


class UIBKSampleWithArray(UIBKSample, PlotSection):
    m_def = Section(
        categories=[UIBKCategory],
        label='UIBK Sample With Array',
    )

    array_generator = SubSection(
        section_def=ArrayGenerator,
        label='Array Generator',
        description='Helper section to generate a new microcell array',
    )

    arrays = SubSection(
        section_def=MicroCellArray, label='MicroCell Arrays', repeats=True
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # generate a new microcell array if requested
        if (
            self.array_generator
            and self.array_generator.create_array
            and self.array_generator.array_x > 0
            and self.array_generator.array_y > 0
        ):
            cells = [
                MicroCell(x=x, y=y, name=f'Cell {str(x)} {str(y)}')
                for x in range(1, self.array_generator.x + 1)
                for y in range(1, self.array_generator.y + 1)
            ]
            self.arrays.append(MicroCellArray(cells=cells))
            self.array_generator = None

        # plot microcell array
        self.figures = []
        if self.arrays:
            self.plot(archive, logger)

    def plot(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        x_values, y_values = self.list_microcell_positions(
            archive.data.arrays[0], logger
        )
        fig = go.Figure(
            data=go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(color='#2A4CDF', size=10),
            )
        )
        annotations = []
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            annotations.append(
                dict(
                    x=x,
                    y=y,
                    # text=f'<a href="{url}">MicroCell {int(x)}-{int(y)}</a>',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-20,
                )
            )
        fig.update_layout(
            title='Sample Overview',
            template='plotly_white',
            # hovermode='closest',
            dragmode=False,
            annotations=annotations,
            # clickmode='event+select',
        )
        plot_json = fig.to_plotly_json()
        plot_json['config'] = dict(
            scrollZoom=False,
        )
        self.figures.append(
            PlotlyFigure(
                label='Sample Overview',
                figure=plot_json,
            )
        )

    def list_microcell_positions(self, array, logger: 'BoundLogger'):
        x_values, y_values = [], []
        for cell in array.cells:
            x_values.append(cell.x)
            y_values.append(cell.y)
        return x_values, y_values


class UIBKSampleReference(CompositeSystemReference):
    """
    Reference section to an UIBK sample.
    """

    reference = Quantity(
        type=UIBKSample,
        description='Reference to an UIBK sample',
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
            label='Link to Sample',
        ),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        super().normalize(archive, logger)

        # Update name
        if self.reference and self.name is None:
            self.name = self.reference.name


m_package.__init_metainfo__()
