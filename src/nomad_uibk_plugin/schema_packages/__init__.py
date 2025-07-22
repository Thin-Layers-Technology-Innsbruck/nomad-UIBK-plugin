from nomad.config.models.plugins import SchemaPackageEntryPoint
from nomad.datamodel.data import EntryDataCategory
from nomad.metainfo.metainfo import Category


class UIBKCategory(EntryDataCategory):
    """
    A category for all measurements defined in the UIBK nomad plugin.
    """

    m_def = Category(label='UIBK', categories=[EntryDataCategory])


class SampleSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_uibk_plugin.schema_packages.sample import m_package

        return m_package


sample = SampleSchemaPackageEntryPoint(
    name='SampleSchema',
    description='Schema package for UIBK samples with MicroCell arrays.',
)


class XRFSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_uibk_plugin.schema_packages.XRFschema import m_package

        return m_package


xrfschema = XRFSchemaPackageEntryPoint(
    name='XRFSchema',
    description='XRF Schema package defined using the new plugin mechanism.',
)


class IFMSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_uibk_plugin.schema_packages.IFMschema import m_package

        return m_package


ifmschema = IFMSchemaPackageEntryPoint(
    name='IFMSchema',
    description='IFM Schema package.',
)


class EfficienciesSchemaEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_uibk_plugin.schema_packages.EfficienciesSchema import m_package

        return m_package


effschema = EfficienciesSchemaEntryPoint(
    name='EfficienciesSchema',
    description='test',
)
