from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np

from PIL import Image
import io
import base64
from pathlib import Path


# ============================================================
# 1. Create FastAPI application
# ============================================================

app = FastAPI(
    title="SynthMicro API",
    description="Leukemia cell classification and Grad-CAM XAI API",
    version="1.0.0"
)


# ============================================================
# 2. Allow frontend to communicate with backend
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. Load model
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_best.keras"

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully")


# ============================================================
# 4. Class names
# ============================================================

class_names = [
    "basophil",
    "erythroblast",
    "monocyte",
    "myeloblast",
    "seg_neutrophil"
]


# ============================================================
# 5. Get ResNet50 base model
# ============================================================

base_model = model.get_layer("resnet50")

last_conv_layer = base_model.get_layer(
    "conv5_block3_out"
)

print("Grad-CAM layer found")


# ============================================================
# 6. Health check
# ============================================================

@app.get("/")
def root():
    return {
        "message": "SynthMicro API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "resnet50_best.keras"
    }


# ============================================================
# 7. Grad-CAM generation
# ============================================================

def generate_gradcam(input_tensor):
    """
    Generate Grad-CAM for the predicted class.
    """

    with tf.GradientTape() as tape:

        # Get convolutional feature maps
        conv_outputs = base_model(
            input_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        # Pass feature maps through classifier
        x = conv_outputs

        for layer in model.layers[1:]:
            x = layer(
                x,
                training=False
            )

        predictions = x

        # Find predicted class
        predicted_index = tf.argmax(
            predictions[0]
        )

        class_output = predictions[
            :,
            predicted_index
        ]

    # Calculate gradients
    grads = tape.gradient(
        class_output,
        conv_outputs
    )

    # Average gradients
    weights = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Weighted combination
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

    # Resize heatmap to image size
    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (224, 224)
    ).numpy().squeeze()

    return (
        heatmap,
        predictions[0].numpy(),
        predicted_index.numpy()
    )


# ============================================================
# 8. Create Grad-CAM overlay
# ============================================================

def create_overlay(original_image, heatmap):
    """
    Create a colored Grad-CAM overlay.
    """

    # Convert heatmap to 0-255
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # Create a simple RGB heatmap
    heatmap_rgb = np.zeros(
        (224, 224, 3),
        dtype=np.uint8
    )

    # Red = high activation
    heatmap_rgb[:, :, 0] = heatmap_uint8

    # Blue = low activation
    heatmap_rgb[:, :, 2] = 255 - heatmap_uint8

    # Convert to PIL
    heatmap_image = Image.fromarray(
        heatmap_rgb
    )

    # Resize original
    original_image = original_image.resize(
        (224, 224)
    ).convert("RGB")

    # Blend original + heatmap
    overlay = Image.blend(
        original_image,
        heatmap_image,
        alpha=0.45
    )

    return overlay


# ============================================================
# 9. Convert PIL image to Base64
# ============================================================

def image_to_base64(image):
    """
    Convert image to Base64 so frontend can display it.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


# ============================================================
# 10. Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # ---------------------------------------------
    # Check file type
    # ---------------------------------------------

    if not file.content_type or not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:

        # -----------------------------------------
        # Read uploaded image
        # -----------------------------------------

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # -----------------------------------------
        # Resize to model input size
        # -----------------------------------------

        resized_image = image.resize(
            (224, 224)
        )

        # -----------------------------------------
        # Convert to NumPy
        #
        # IMPORTANT:
        # No /255.0 here.
        # This matches model training.
        # -----------------------------------------

        img_array = np.array(
            resized_image,
            dtype=np.float32
        )

        input_tensor = tf.expand_dims(
            img_array,
            axis=0
        )

        # -----------------------------------------
        # Generate prediction + Grad-CAM
        # -----------------------------------------

        heatmap, probabilities, predicted_index = generate_gradcam(
            input_tensor
        )

        # -----------------------------------------
        # Prediction details
        # -----------------------------------------

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index] * 100
        )

        # -----------------------------------------
        # All class probabilities
        # -----------------------------------------

        probability_dict = {}

        for name, probability in zip(
            class_names,
            probabilities
        ):
            probability_dict[name] = float(
                probability * 100
            )

        # -----------------------------------------
        # Create Grad-CAM overlay
        # -----------------------------------------

        overlay = create_overlay(
            resized_image,
            heatmap
        )

        # -----------------------------------------
        # Convert overlay to Base64
        # -----------------------------------------

        overlay_base64 = image_to_base64(
            overlay
        )

        # -----------------------------------------
        # Return response
        # -----------------------------------------

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probability_dict,
            "gradcam": overlay_base64
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )