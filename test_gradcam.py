import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Load trained model
# ---------------------------------
model = tf.keras.models.load_model(
    "models/resnet50_best.keras"
)

print("Model loaded successfully")

# ---------------------------------
# 2. Class names
# ---------------------------------
class_names = [
    "basophil",
    "erythroblast",
    "monocyte",
    "myeloblast",
    "seg_neutrophil"
]

# ---------------------------------
# 3. Image
# ---------------------------------
image_path = r"dataset/basophil/BA_100102.jpg"

img = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

img_array = tf.keras.utils.img_to_array(img)

# IMPORTANT:
# Do NOT divide by 255.
# This matches the model's training pipeline.
input_tensor = tf.expand_dims(img_array, axis=0)

# ---------------------------------
# 4. Get ResNet50 base model
# ---------------------------------
base_model = model.get_layer("resnet50")

print("ResNet50 base model found")

# ---------------------------------
# 5. Grad-CAM
# ---------------------------------
with tf.GradientTape() as tape:

    # Get convolutional feature maps
    conv_outputs = base_model(
        input_tensor,
        training=False
    )

    # Watch the feature maps
    tape.watch(conv_outputs)

    # Pass feature maps through the classifier
    x = conv_outputs

    for layer in model.layers[1:]:
        x = layer(x, training=False)

    predictions = x

    # Get predicted class
    predicted_index = tf.argmax(
        predictions[0]
    )

    # Score of predicted class
    class_output = predictions[:, predicted_index]

# ---------------------------------
# 6. Calculate gradients
# ---------------------------------
grads = tape.gradient(
    class_output,
    conv_outputs
)

# Average gradients across width and height
weights = tf.reduce_mean(
    grads,
    axis=(1, 2)
)

# Weighted combination of feature maps
cam = tf.reduce_sum(
    conv_outputs *
    weights[:, tf.newaxis, tf.newaxis, :],
    axis=-1
)

# ReLU
cam = tf.maximum(cam, 0)

# Normalize
cam = cam / (
    tf.reduce_max(
        cam,
        axis=(1, 2),
        keepdims=True
    ) + 1e-8
)

heatmap = cam[0].numpy()

# ---------------------------------
# 7. Resize heatmap
# ---------------------------------
heatmap = tf.image.resize(
    heatmap[..., np.newaxis],
    (224, 224)
).numpy().squeeze()

# ---------------------------------
# 8. Prediction information
# ---------------------------------
predicted_class = class_names[
    predicted_index.numpy()
]

confidence = (
    predictions[0][predicted_index].numpy()
    * 100
)

print()
print("Prediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

print()
print("All probabilities:")

for name, probability in zip(
    class_names,
    predictions[0].numpy()
):
    print(
        f"{name}: {probability * 100:.4f}%"
    )

# ---------------------------------
# 9. Display Grad-CAM
# ---------------------------------
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)

plt.imshow(img)

plt.title(
    f"Original\n"
    f"{predicted_class} ({confidence:.2f}%)"
)

plt.axis("off")

# Grad-CAM
plt.subplot(1, 2, 2)

plt.imshow(img)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.45
)

plt.title("Grad-CAM")

plt.axis("off")

plt.tight_layout()

# ---------------------------------
# 10. Save result
# ---------------------------------
output_path = "gradcam_result.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print()
print("Grad-CAM saved to:", output_path)