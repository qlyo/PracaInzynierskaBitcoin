"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["btc_preprocessed_data_1d","btc_preprocessed_data_1w"],
            outputs=["X_train_1d", "y_train_1d", "X_train_1w", "y_train_1w"],
            name="split_data_node",
        ),
        node(
            func=train_model,
            inputs=["X_train_1d", "y_train_1d", "X_train_1w", "y_train_1w"],
            outputs=["model_1d","model_1w"],
            name="train_model_node",
        ),
        node(
            func=evaluate_model,
            inputs=["model_1d", "btc_preprocessed_data_1d", "model_1w","btc_preprocessed_data_1w"],
            outputs=None,
            name="evaluate_model_node",
        ),
    ])
