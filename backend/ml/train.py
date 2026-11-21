"""CPU-friendly training entry point for custom CNN and MobileNetV2."""
import argparse
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from constants import CLASS_NAMES, IMAGE_SIZE, SEED
from evaluate import save_evaluation

def datasets(data_dir: Path, batch_size: int):
    options = dict(image_size=IMAGE_SIZE, batch_size=batch_size, label_mode="int", class_names=CLASS_NAMES, seed=SEED)
    train = tf.keras.utils.image_dataset_from_directory(data_dir / "train", shuffle=True, **options)
    validation = tf.keras.utils.image_dataset_from_directory(data_dir / "val", shuffle=False, **options)
    test = tf.keras.utils.image_dataset_from_directory(data_dir / "test", shuffle=False, **options)
    autotune = tf.data.AUTOTUNE
    return train.prefetch(autotune), validation.prefetch(autotune), test.prefetch(autotune)

def augmentation():
    return tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.08), tf.keras.layers.RandomZoom(0.1)], name="training_augmentation")

def custom_cnn():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation()(inputs)
    x = tf.keras.layers.Rescaling(1 / 255.0)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn"), "normalized"

def mobilenet_v2():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2"), "mobilenet_v2"

def graph(history, destination: Path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train"); axes[0].plot(history.history["val_accuracy"], label="validation"); axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train"); axes[1].plot(history.history["val_loss"], label="validation"); axes[1].set_title("Loss"); axes[1].legend()
    figure.tight_layout(); figure.savefig(destination, dpi=160); plt.close(figure)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../../data/processed")
    parser.add_argument("--model", choices=["custom_cnn", "mobilenetv2"], required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16, help="Use 8 if RAM is constrained.")
    parser.add_argument("--publish", action="store_true", help="Make this trained model the API model.")
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(SEED)
    data, backend = Path(args.data).resolve(), Path(__file__).resolve().parents[1]
    reports, models = backend / "artifacts" / "reports", backend / "artifacts" / "models"
    train, validation, test = datasets(data, args.batch_size)
    model, preprocessing = custom_cnn() if args.model == "custom_cnn" else mobilenet_v2()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    checkpoint = models / f"{args.model}_best.keras"; models.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True)]
    history = model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks)
    model = tf.keras.models.load_model(checkpoint)
    graph(history, reports / f"{args.model}_training_graph.png")
    metrics = save_evaluation(model, test, args.model, reports)
    metadata = {"architecture": args.model, "model_file": checkpoint.name, "class_names": CLASS_NAMES, "image_size": list(IMAGE_SIZE), "preprocessing": preprocessing, "metrics_file": f"{args.model}_metrics.json"}
    (models / f"{args.model}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.publish:
        shutil.copy2(checkpoint, models / "best_model.keras")
        metadata["model_file"] = "best_model.keras"
        (models / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
