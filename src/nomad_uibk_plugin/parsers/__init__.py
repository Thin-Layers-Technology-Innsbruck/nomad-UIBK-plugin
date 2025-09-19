from nomad.config.models.plugins import ParserEntryPoint


class XRFParserEntryPoint(ParserEntryPoint):
    """
    XRF Parser plugin entry point.
    """

    def load(self):
        # lazy import to avoid circular dependencies
        from nomad_uibk_plugin.parsers.XRFparser import XRFParser

        return XRFParser(**self.dict())


xrfparser = XRFParserEntryPoint(
    name='XRFParser',
    description='XRF Parser for UIBK .txt files.',
    mainfile_name_re=r'.*\.txt',
    mainfile_content_re=r'PositionType\s+Application\s+Sample\s+name\s+Date\s+\n[A-Z0-9-]+\s+Quant\s+analysis',
)

# # Microcell parser entry points
# class EBICParserEntryPoint(ParserEntryPoint):
#     """
#     EBIC Parser plugin entry point.
#     """

#     def load(self):
#         # lazy import to avoid circular dependencies
#         from nomad_uibk_plugin.parsers.microcellparsers import EBICParser

#         return EBICParser(**self.dict())


# ebicparser = EBICParserEntryPoint(
#     name='EBICParser',
#     description='Parser for EBIC tiff files.',
#     mainfile_name_re='.*\.(tif|tiff)',
#     mainfile_content_re='pointelectronic\.com',
# )


class IFMParserEntryPoint(ParserEntryPoint):
    """
    IFM Parser plugin entry point.
    """

    def load(self):
        # lazy import to avoid circular dependencies
        from nomad_uibk_plugin.parsers.IFMparser import IFMParser

        return IFMParser(**self.model_dump())


ifmparser = IFMParserEntryPoint(
    name='IFMParser',
    description='IFM Parser for bmp and xml files.',
    mainfile_name_re=r'.*(?:/texture\.bmp|/info\.xml)',
    # mainfile_name_re=r'.*/IFM_.*(?:\.bmp|_info\.xml)',
)


class IFMModelParserEntryPoint(ParserEntryPoint):
    """
    IFM Parser plugin entry point.
    """

    def load(self):
        # lazy import to avoid circular dependencies
        from nomad_uibk_plugin.parsers.IFMparser import IFMModelParser

        return IFMModelParser(**self.model_dump())


ifmmodelparser = IFMModelParserEntryPoint(
    name='IFMModelParser',
    description='IFM Model Parser for keras files.',
    mainfile_name_re=r'.*/Model_.*(?:_binary\.keras|_classification\.keras)',
)
