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
            outputs=["btc_raw_dataset_1d","btc_raw_dataset_1w",'train_dates'],
            name="download_data_node"
        ),
        node(
            func=preprocess_btc_raw,
            inputs=["btc_raw_dataset_1d","btc_raw_dataset_1w"],
            outputs=["btc_preprocessed_data_1d","btc_preprocessed_data_1w","scaler1d","scaler1w"],
            name="preprocess_btc_data_node",
        ),
    ])
