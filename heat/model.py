import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_cnn_lstm(input_shape, n_classes=1, wd=1e-4, dr=0.4):
    """
    Build a regularized CNN–LSTM model.
    input_shape: (T, H, W, C)
    n_classes: 1 (binary with sigmoid) or >1 (multiclass with softmax)
    wd: L2 weight decay
    dr: dropout rate
    """
    inp = layers.Input(shape=input_shape)

    # CNN applied to each timestep
    cnn = models.Sequential([
        layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(wd)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(wd)),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dr),
    ])
    x = layers.TimeDistributed(cnn)(inp)  # (T, F)

    # Temporal dynamics with LSTM
    x = layers.LSTM(128, return_sequences=False,
                    kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.Dropout(dr)(x)

    # Output layer
    if n_classes == 1:
        out = layers.Dense(1, activation="sigmoid",
                           kernel_regularizer=regularizers.l2(wd))(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auroc")]
    else:
        out = layers.Dense(n_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(wd))(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=loss, metrics=metrics)
    return model
