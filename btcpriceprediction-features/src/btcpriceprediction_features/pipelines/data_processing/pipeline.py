"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import download_data, preprocess_btc_raw

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=download_data,
            inputs=None,
            outputs="btc_raw_data",
            name="download_data_node"
        ),
        node(
            func=preprocess_btc_raw,
            inputs="btc_raw_data",
            outputs="btc_preprocessed_data",
            name="preprocess_btc_data_node",
        ),
    ])
