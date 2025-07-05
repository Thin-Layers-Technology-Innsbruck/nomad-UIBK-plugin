from typing import TYPE_CHECKING

import numpy as np
from ase.data import chemical_symbols
from nomad.datamodel.data import ArchiveSection, EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.basesections import (
    CompositeSystem,
    ElementalComposition,
    EntityReference,
    PubChemPureSubstanceSection,
    PureSubstanceComponent,
    SectionReference,
    # Component,
    # CompositeSystemReference,
    # InstrumentReference,
    # PureSubstanceSection,
)
from nomad.metainfo import (
    Datetime,
    MEnum,
    Quantity,
    # SectionProxy,
    SchemaPackage,
    # Package,
    Section,
    # MSection,
    SubSection,
)

# from nomad_material_processing.vapor_deposition.pvd.general import (
#     PVDSource, TimeSeries
# )
from nomad_material_processing.vapor_deposition.pvd.sputtering import SputterDeposition
from nomad_measurements.utils import merge_sections  # create_archive
from pint import UnitRegistry
from structlog.stdlib import BoundLogger

from nomad_uibk_plugin.schema_packages import UIBKCategory
from nomad_uibk_plugin.schema_packages.basesections import (
    Current,
    GasFlow,
    Power,
    Pressure,
    Voltage,
)
from nomad_uibk_plugin.schema_packages.sample import UIBKSample

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger


m_package = SchemaPackage()

ureg = UnitRegistry()


class Substrate(EntryData):
    """
    - type
    - subtype
    - manufacturer
    - Batch
    - size (x, y, z)
    -
    """

    m_def = Section()

    type = Quantity(
        type=MEnum(['c-Si die', 'glass', 'polyimide']),
        description='Type of substrate material',
        default='',
        a_eln={'component': 'AutocompleteEditQuantity'},
    )
    subtype = Quantity(
        type=str,
        description='Subtype of the substrate.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    batch = Quantity(
        type=str,
        description='Batch ID',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    mountingType = Quantity(
        type=MEnum(['retainers', 'double sided tape', 'retainers FlexPecs Narrow']),
        description='How is the substrate mounted in the chamber?',
        default='',
        a_eln={'component': 'RadioEnumEditQuantity'},
    )

    size_x = Quantity(
        type=float,
        description='Radius or x-dimension of the sample.',
        a_eln={'component': 'NumberEditQuantity', 'defaultDisplayUnit': 'mm'},
        unit='meter',
    )
    size_y = Quantity(
        type=float,
        description='Y-dimension of the sample.',
        a_eln={'component': 'NumberEditQuantity', 'defaultDisplayUnit': 'mm'},
        unit='meter',
    )

    size_z = Quantity(
        type=float,
        description='Height of the sample.',
        a_eln={'component': 'NumberEditQuantity', 'defaultDisplayUnit': 'mm'},
        unit='meter',
    )


class SubstrateReference(SectionReference):
    reference = Quantity(type=Substrate, a_eln=dict(component='ReferenceEditQuantity'))


class Operator(EntryData):
    author = Quantity(
        type=str,  ## < ?
        description='Authors/Operators',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,  # AuthorEditQuantity?
        ),
    )


class OperatorReference(SectionReference):
    reference = Quantity(type=Operator, a_eln=dict(component='ReferenceEditQuantity'))


class PSU(EntryData):
    """
    static
    - NOMAD internal ID
    - Manufacturer
    - Reseller
    - Model
    - Serial Number
    - prior use
    - [DC only, RF, Pulsed]

    variable
    - Operating Hours
    - Operating Joules

    time based
    - Setpoint type [W, V, A]
    - Setpoint value
    - RF settings
    - Pulsed settingss
        - Frequency
        - pulse reverse time
    """

    m_def = Section()

    internal_id = Quantity(
        type=int,
        description='NOMAD ID of the PSU',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
        a_display={'visible': False},
    )
    manufacturer_name = Quantity(
        type=str,
        description='Name of manufacturing company.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    reseller_name = Quantity(
        type=str,
        description='Name of reseller.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    model = Quantity(
        type=str,
        description='Name of reseller.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    serial = Quantity(
        type=str,
        description='Name of reseller.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    installation_date = Quantity(
        type=Datetime,
        description='',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.DateEditQuantity,
        ),
    )
    DC_capability = Quantity(
        type=bool,
        description='',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    rf_capability = Quantity(
        type=bool,
        description='',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    pulsed_capability = Quantity(
        type=bool,
        description='',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    prior_use = Quantity(
        type=bool,
        description='The PSU was in use before being added to NOMAD.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    operating_hours = Quantity(
        type=int,
        description='NOMAD ID of the PSU',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
        unit='hour',
    )
    operating_joules = Quantity(
        type=int,
        description='NOMAD ID of the PSU',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
        unit='joule',
    )
    m_def = Section()
    setpoint_type = Quantity(
        type=MEnum(['W', 'V', 'A']),
        description='',
        default='W',
        a_eln={'component': 'RadioEnumEditQuantity'},
    )
    setpoint_value = Quantity(
        type=float,
        description='',
    )
    m_def = Section()
    # RF settings
    m_def = Section()
    # Pulsed settings
    pulse_frequency = Quantity(
        type=int,
        description='',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            defaultDisplayUnit='kHz',
            label='Pulse frequency',
        ),
        unit='Hz',
    )
    pulse_reverse_time = Quantity(
        type=int,
        description='',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            defaultDisplayUnit='ms',
            label='Pulse reverse time',
        ),
        unit='s',
    )


class PsuReference(SectionReference):
    reference = Quantity(type=PSU, a_eln=dict(component='ReferenceEditQuantity'))


class PressureGauge(EntryData):
    """
    static

    time based
    - pressure
    """

    internal_id = Quantity(
        type=int,
        description='NOMAD ID of the PSU',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
        a_display={'visible': False},
    )
    internal_name = Quantity(
        type=str,
        description='Internal name',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity, label='Internal name'
        ),
    )
    manufacturer_name = Quantity(
        type=str,
        description='Name of manufacturing company.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    reseller_name = Quantity(
        type=str,
        description='Name of reseller.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    model = Quantity(
        type=str,
        description='Name of reseller.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    serial = Quantity(
        type=str,
        description='Name of reseller.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    installation_date = Quantity(
        type=Datetime,
        description='',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.DateEditQuantity,
        ),
    )
    pressure = Quantity(
        type=float,
        description='',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            defaultDisplayUnit='mBar',
            label='Pressure',
        ),
        unit='Pa',
    )


class PressureGaugeReference(SectionReference):
    reference = Quantity(type=PSU, a_eln=dict(component='ReferenceEditQuantity'))


class MassFlowController(EntryData):
    """
    static
    - Type
    - Medium [Ar, O2, N2]
    ...

    time based
    - mass flow
    """

    pass


class SputterParameters(ArchiveSection):
    """
    static
    - Operator (multiple) (->Operator)
    - samples (multiple) (-> UIBKSample)
        - ids
            - internal
            - external (multiple)
                - partner
                - ID
        - Substrate (->Substrate)
    - Installed targets (multiple)(->Target)
    - Installed PSUs (multiple) (-> PSU)
    - sample distance_to_source
    - Comment
    - LN² (bool)

    time based
    - timestamp
    - PSU(n) -> Target(n)
    - Setpoint type [W, V, A]
    - Setpoint Value
    - Pressure gauge (multiple) (->PressureGauge)
    - sample rotation speed
    """

    test = Quantity(type=str)


#     """
#     Measurements
#     - Photos
#     - IFM
#     - EDX

#     Associations
#     - List of all VPs a target was used in
#     - A chronological list of all processes
#     """


class TargetComponent(PureSubstanceComponent):
    """
    A component of a target.
    Usefull to describe the creation of a target from multiple components
    each with their own mass or mass fraction,
    """

    pure_substance = SubSection(
        section_def=PubChemPureSubstanceSection,
        description="""
        Section describing the pure substance that is the component.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        """
        Creates a new PubChemPureSubstanceSection if necessary
        to trigger the population of elemental composition in Target.
        """
        if not self.name:
            return
        if (
            self.pure_substance is None
            or self.name != self.pure_substance.molecular_formula
        ):
            self.pure_substance = PubChemPureSubstanceSection(
                molecular_formula=self.name
            )
            self.pure_substance.normalize(archive, logger)
        if (
            self.substance_name is None
            or self.substance_name != self.pure_substance.name
        ):
            self.substance_name = self.pure_substance.name


class Target(CompositeSystem, EntryData):
    """
    A Target which is used in a sputter deposition process.
    """

    m_def = Section(
        label='Target',
        categories=[UIBKCategory],
    )

    components = SubSection(section_def=TargetComponent, repeats=True)

    total_depositions = Quantity(
        type=int,
        description='Number of depositions the target has been used in',
        default=0,
    )
    total_dose = Quantity(
        type=float,
        unit='joule',
        description='Total dose of the target',
        default=0,
    )
    cross_contamination = Quantity(
        type=MEnum(chemical_symbols[1:]),
        description=(
            'Elements that are originally not part of the target'
            'but may be present due to cross contamination from the previous deposition'
        ),
        shape=['*'],
    )

    elemental_composition = SubSection(
        description="""
        A list of all the elements and their respective atomic or mass fractions 
        found in the target.
        Either filled automatically by populating the components or manually by editing.
        """,
        section_def=ElementalComposition,
        repeats=True,
    )

    def update_from_reference(
        self, archive: 'EntryArchive', logger: 'BoundLogger'
    ) -> None:
        """
        Update with information from references.
        """

        self.total_dose = 0
        self.total_depositions = 0

        from nomad.search import MetadataPagination, search

        query = {
            'section_defs.definition_qualified_name:all': [
                'nomad_uibk_plugin.schema_packages.sputtering.TargetReference',
            ],
            'entry_references.target_entry_id:all': [
                archive.metadata.entry_id,
            ],
        }

        # Query a single reference at a time
        page_after_value = None
        while True:
            search_result = search(
                owner='all',
                query=query,
                pagination=MetadataPagination(
                    page_size=1, page_after_value=page_after_value
                ),
                user_id=archive.metadata.main_author.user_id,
            )

            for result in search_result.data:
                # flatten result list to dict
                quantity_dict = {
                    item['path_archive']: item for item in result['search_quantities']
                }

                # Get target names, doses and upate properties
                left_name = quantity_dict.get('data.left_target.name')
                left_dose = quantity_dict.get('data.left_target.dose')
                right_name = quantity_dict.get('data.right_target.name')
                right_dose = quantity_dict.get('data.right_target.dose')

                if left_name and left_dose and left_name['string_value'] == self.name:
                    self.total_depositions += 1
                    self.total_dose += left_dose['float_value']
                elif (
                    right_name
                    and right_dose
                    and right_name['string_value'] == self.name
                ):
                    self.total_depositions += 1
                    self.total_dose += right_dose['float_value']

            if search_result.pagination.next_page_after_value:
                page_after_value = search_result.pagination.next_page_after_value
            else:
                break

    def normalize(self, archive, logger: 'BoundLogger') -> None:
        if self.components:
            self.elemental_composition = []

        super().normalize(archive, logger)
        self.update_from_reference(archive, logger)


class TargetReference(EntityReference):
    """
    Describes the way the target is used in a sputter deposition process.
    Includes:
    - Link to the entry of the target
    - Parameters of the mode of operation
    """

    reference = Quantity(type=Target, a_eln=dict(component='ReferenceEditQuantity'))

    mode = Quantity(
        type=MEnum(['DC Continuous', 'DC Pulsed']),
        description='Mode of operation this target was used in',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.RadioEnumEditQuantity,
            label='Operation Mode',
        ),
    )

    setpoint_unit = Quantity(
        type=MEnum(['W', 'V', 'A']),
        description='Setpoint mode for this target',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.RadioEnumEditQuantity,
            label='Setpoint mode',
        ),
    )

    power = SubSection(
        section_def=Power,
        description='Power at the target during deposition',
    )

    current = SubSection(
        section_def=Current,
        description='Current at the target during deposition',
    )

    voltage = SubSection(
        section_def=Voltage,
        description='Voltage at the target during deposition',
    )

    duration = Quantity(
        type=float,
        description='Duration for which the target is used during deposition',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            defaultDisplayUnit='second',
        ),
        unit='second',
    )

    pulse_frequency = Quantity(
        type=float,
        description='Frequency of the target during deposition if not DC',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
        unit='Hz',
    )

    pulse_reverse_time = Quantity(
        type=float,
        description=(
            'Pulse reverse time of the target during deposition if Pulsed mode is used'
        ),
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            defaultDisplayUnit='microsecond',
        ),
        unit='microsecond',
    )

    dose = Quantity(
        type=float,
        description="""Estimated energy input (dose)
        that the target has experienced during deposition""",
        unit='joule',
        a_eln=ELNAnnotation(
            defaultDisplayUnit='joule',
        ),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        """
        Automations of TargetReference:
        - Update the name of the target and estimate the dose if possible.
        - Estimate the energy input (dose) the target has experienced during deposition.
        - Update the target entry with the new information.
        """

        # Update name
        if self.reference and self.reference.name:
            self.name = self.reference.name

        # Estimate the dose
        if self.duration is not None and self.power is not None:
            if self.mode == 'DC':
                dose = np.median(self.power.value) * self.duration
                self.dose = dose.to('joule')  # TODO: Check if this is necessary
            elif self.mode == 'DC Pulsed':
                active = self.frequency * (1 * ureg.s - self.pulse_reverse_time)
                dose = np.median(self.power.value) * self.duration * active
                self.dose = dose.to('joule')

        # Update the target entry


class Atmosphere(ArchiveSection):
    """
    Describes the state of the atmosphere during the deposition.
    """

    pressure = SubSection(
        section_def=Pressure,
        description='Pressure of the atmosphere during deposition',
    )

    Ar = SubSection(
        section_def=GasFlow,
        description='Pressure of Argon during deposition',
    )

    O2 = SubSection(
        section_def=GasFlow,
        description='Pressure of Oxygen during deposition',
    )

    N2 = SubSection(
        section_def=GasFlow,
        description='Pressure of Nitrogen during deposition',
    )


class UIBKSputterDeposition(SputterDeposition, EntryData):
    m_def = Section(
        categories=[UIBKCategory],
        label='Sputter Deposition',
    )

    log_file = Quantity(
        type=str,
        description='Path to the data file',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
    )

    samples = SubSection(section_def=UIBKSample, repeats=True)

    target_left = SubSection(section_def=TargetReference)
    target_right = SubSection(section_def=TargetReference)
    atmosphere = SubSection(section_def=Atmosphere)

    distance_to_target = Quantity(
        type=float,
        description='Distance between target and sample',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            defaultDisplayUnit='mm',
        ),
        unit='m',
    )

    substrate_rotation = Quantity(
        type=float,
        description='Rotation speed of the substrate retainer in rpm',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            defaultDisplayUnit='rpm',
        ),
        unit='revolutions_per_second',
    )

    parameters = SubSection(section_def=SputterParameters)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        """Update the data from the data file if it is not already set"""

        from nomad_uibk_plugin.filereader.sputterreader import read_sputter_deposition

        # if self.data is None and self.data_file is not None:
        if self.log_file is not None:
            with archive.m_context.raw_file(self.log_file) as file:
                sputter_entry = read_sputter_deposition(file, archive, logger)
                merge_sections(self, sputter_entry, logger)

        # Add cross contamination to the targets and update dose
        # if (self.target_left and self.target_right and
        #         self.target_left.reference is not None
        #         and self.target_right.reference is not None
        #     ):
        #     for composition in self.target_right.reference.elemental_composition:
        #         element = composition.element
        #         if element not in self.target_left.reference.cross_contamination:
        #             self.target_left.reference.cross_contamination.append(element)

        #     for composition in self.target_left.reference.elemental_composition:
        #         element = composition.element
        #         if element not in self.target_right.reference.cross_contamination:
        #             self.target_right.reference.cross_contamination.append(element)

        super().normalize(archive, logger)


m_package.__init_metainfo__()
