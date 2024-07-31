"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs="btc_preprocessed_data",
            outputs=["X_train", "y_train", "data"],
            name="split_data_node",
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train"],
            outputs="model",
            name="train_model_node",
        ),
        node(
            func=evaluate_model,
            inputs=["model", "data"],
            outputs=None,
            name="evaluate_model_node",
        ),
    ])
