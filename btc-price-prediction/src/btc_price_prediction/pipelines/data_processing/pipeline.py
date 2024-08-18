"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import preprocess_btc_raw, download_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=download_data,
            inputs=["params:crypto_currency", "params:against_currency", "params:start_date", "params:end_date",
                    "params:force_download"],
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
