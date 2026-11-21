"""Grad-CAM utilities used after a real Keras model has been loaded."""
import base64
from io import BytesIO
import numpy as np
from PIL import Image

def _find_last_convolution(model):
    for layer in reversed(model.layers):
        # Keras 3 no longer guarantees ``output_shape`` on every layer.
        # Tensor rank is the portable way to identify convolutional feature maps.
        output = getattr(layer, "output", None)
        if output is not None and len(output.shape) == 4:
            return layer.name
    raise ValueError("No convolutional feature layer was found for Grad-CAM.")

def make_gradcam_data_url(model, prepared_batch: np.ndarray, original_rgb: np.ndarray, class_index: int) -> str:
    """Create a JPEG data URL. It is computed for the current prediction, never cached."""
    import tensorflow as tf
    import cv2
    layer_name = _find_last_convolution(model)
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(prepared_batch)
        loss = predictions[:, class_index]
    gradients = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * weights, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    maximum = heatmap.max()
    if maximum > 0:
        heatmap /= maximum
    heatmap = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR), 0.55, heatmap_color, 0.45, 0)
    image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
