from nomad.config.models.plugins import AppEntryPoint

from nomad_uibk_plugin.apps.ifm_analysis_app import ifm_analysis_app
from nomad_uibk_plugin.apps.uibk_sample_app import uibk_sample_app

ifmanalysisapp_ep = AppEntryPoint(
    name='IFM Analysis',
    description="""
      This app allows you to search for the results of IFM Two Step Analysis 
      within NOMAD.
    """,
    app=ifm_analysis_app,
)

uibksamplesapp_ep = AppEntryPoint(
    name='Samples',
    description="""
      This app allows you to search for samples within UIBK definition.
    """,
    app=uibk_sample_app,
)
