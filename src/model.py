from __future__ import annotations

import tensorflow as tf

from src.config import IMAGE_SIZE


def build_model(
    num_classes: int,
    image_size: tuple[int, int] = IMAGE_SIZE,
    arch: str = "mobilenetv2",
    base_weights: str | None = "imagenet",
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    arch_lower = arch.lower()
    if arch_lower == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*image_size, 3),
            include_top=False,
            weights=base_weights,
        )
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
    elif arch_lower == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=(*image_size, 3),
            include_top=False,
            weights=base_weights,
        )
        preprocess_fn = tf.keras.applications.resnet50.preprocess_input
    elif arch_lower == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(*image_size, 3),
            include_top=False,
            weights=base_weights,
        )
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(
            f"Unsupported architecture: '{arch}'. Choose from 'mobilenetv2', 'resnet50', 'efficientnetb0'."
        )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = data_augmentation(inputs)
    x = preprocess_fn(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=f"crop_classifier_{arch_lower}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
