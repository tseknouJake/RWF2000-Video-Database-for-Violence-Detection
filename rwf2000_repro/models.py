from __future__ import annotations

from pathlib import Path

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="rwf2000_repro")
def get_rgb(input_x):
    return input_x[..., :3]


@tf.keras.utils.register_keras_serializable(package="rwf2000_repro")
def get_opt(input_x):
    return input_x[..., 3:5]


def _stacked_p3d_block(x, filters: int, optical_gate: bool = False):
    activation = "sigmoid" if optical_gate else "relu"
    x = tf.keras.layers.Conv3D(
        filters,
        kernel_size=(1, 3, 3),
        strides=(1, 1, 1),
        kernel_initializer="he_normal",
        activation=activation,
        padding="same",
    )(x)
    x = tf.keras.layers.Conv3D(
        filters,
        kernel_size=(3, 1, 1),
        strides=(1, 1, 1),
        kernel_initializer="he_normal",
        activation=activation,
        padding="same",
    )(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    return x


def _merge_head(x):
    x = tf.keras.layers.MaxPooling3D(pool_size=(8, 1, 1))(x)

    for filters, pool_size in ((64, (2, 2, 2)), (64, (2, 2, 2)), (128, (2, 3, 3))):
        x = tf.keras.layers.Conv3D(
            filters,
            kernel_size=(1, 3, 3),
            strides=(1, 1, 1),
            kernel_initializer="he_normal",
            activation="relu",
            padding="same",
        )(x)
        x = tf.keras.layers.Conv3D(
            filters,
            kernel_size=(3, 1, 1),
            strides=(1, 1, 1),
            kernel_initializer="he_normal",
            activation="relu",
            padding="same",
        )(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=pool_size)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    return outputs


def build_rgb_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(64, 224, 224, 3))
    x = inputs
    for filters in (16, 16, 32, 32):
        x = _stacked_p3d_block(x, filters)
    outputs = _merge_head(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="rgb_only")


def build_opt_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(64, 224, 224, 2))
    x = inputs
    for filters in (16, 16, 32, 32):
        x = _stacked_p3d_block(x, filters)
    outputs = _merge_head(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="opt_only")


def build_flow_gated_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(64, 224, 224, 5))

    rgb = tf.keras.layers.Lambda(get_rgb, name="rgb_slice")(inputs)
    opt = tf.keras.layers.Lambda(get_opt, name="opt_slice")(inputs)

    for filters in (16, 16, 32):
        rgb = _stacked_p3d_block(rgb, filters)
        opt = _stacked_p3d_block(opt, filters)

    rgb = _stacked_p3d_block(rgb, 32)
    opt = _stacked_p3d_block(opt, 32, optical_gate=True)

    x = tf.keras.layers.Multiply()([rgb, opt])
    outputs = _merge_head(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="flow_gated")


def build_model(architecture: str) -> tf.keras.Model:
    if architecture == "rgb":
        return build_rgb_model()
    if architecture == "opt":
        return build_opt_model()
    if architecture == "flow-gated":
        return build_flow_gated_model()
    raise ValueError(f"Unsupported architecture: {architecture}")


def load_model(model_path: str | Path, architecture: str = "flow-gated") -> tf.keras.Model:
    model_path = Path(model_path)
    custom_objects = {"get_rgb": get_rgb, "get_opt": get_opt}

    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False, custom_objects=custom_objects)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception:
        model = build_model(architecture)
        model.load_weights(model_path)
        return model
