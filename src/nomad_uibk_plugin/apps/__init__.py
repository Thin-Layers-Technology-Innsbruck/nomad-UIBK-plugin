from nomad.config.models.plugins import AppEntryPoint

from nomad_uibk_plugin.apps.ifm_analysis_app import ifm_analysis_app

ifmanalysisapp_ep = AppEntryPoint(
    name='IFM Analysis',
    description="""
      This app allows you to search for the results of IFM Two Step Analysis 
      within NOMAD.
    """,
    app=ifm_analysis_app,
)
