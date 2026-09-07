import io
import os
import base64
import uuid
import sqlite3

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SynthMicro API",
    description="AI-powered microscopic blood-cell classification with Grad-CAM",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PROJECT PATHS
# ============================================================

# main.py:
# C:\Users\Sunny\SynthMicro\backend\main.py
#
# BASE_DIR:
# C:\Users\Sunny\SynthMicro

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "resnet50_best.keras"

UPLOAD_DIR = BASE_DIR / "uploads"

XAI_DIR = BASE_DIR / "xai_results"

DB_PATH = BASE_DIR / "synthmicro.db"


# Create required directories
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

XAI_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "basophil",
    "erythroblast",
    "monocyte",
    "myeloblast",
    "seg_neutrophil"
]


# ============================================================
# DATABASE
# ============================================================

def init_db():

    conn = sqlite3.connect(
        str(DB_PATH)
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT,
            heatmap_path TEXT,
            xai_path TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


# Initialize database
init_db()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("SynthMicro")
print("Loading model...")
print("=" * 60)

print(f"Model path: {MODEL_PATH}")


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


print("Model loaded successfully.")

print(
    "Model layers:",
    [layer.name for layer in model.layers]
)

print("=" * 60)


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(
    image: Image.Image
) -> str:

    """
    Convert a PIL image to PNG Base64.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image: Image.Image
):
    """
    Prepare image for SynthMicro ResNet50.

    IMPORTANT:

    The SynthMicro model was trained using raw pixel values.

    Therefore:
        - Do NOT divide by 255
        - Do NOT use preprocess_input()
    """

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(img_array, model):
    """
    Generate a localized Grad-CAM heatmap for the
    class predicted by the ResNet50 model.

    Final convolutional layer:
        conv5_block3_out
    """

    # --------------------------------------------------------
    # Get nested ResNet50
    # --------------------------------------------------------

    base_model = model.get_layer("resnet50")

    last_conv_layer = base_model.get_layer(
        "conv5_block3_out"
    )

    # --------------------------------------------------------
    # Build Grad-CAM model
    # --------------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, features = grad_model(
            img_array,
            training=False
        )

        # Classification head
        x = features

        for layer in model.layers[1:]:

            x = layer(
                x,
                training=False
            )

        predictions = x

        # Predicted class
        pred_index = tf.argmax(
            predictions[0]
        )

        class_channel = predictions[
            :,
            pred_index
        ]

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    if grads is None:

        raise ValueError(
            "Could not calculate Grad-CAM gradients."
        )

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    conv_outputs = conv_outputs[0]

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap.numpy()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)

    if heatmap_max <= heatmap_min:

        return np.zeros_like(
            heatmap,
            dtype=np.float32
        )

    heatmap = (
        heatmap - heatmap_min
    ) / (
        heatmap_max - heatmap_min
    )

    # ========================================================
    # IMPORTANT:
    # Suppress weak activation
    # ========================================================

    threshold = np.percentile(
        heatmap,
        60
    )

    heatmap[heatmap < threshold] = 0

    # --------------------------------------------------------
    # Re-normalize after thresholding
    # --------------------------------------------------------

    max_value = np.max(heatmap)

    if max_value > 0:

        heatmap = heatmap / max_value

    # --------------------------------------------------------
    # Slight smoothing
    #
    # Don't over-blur because we want the activation
    # to remain localized around the cell.
    # --------------------------------------------------------

    heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=1.0
    )

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)

    if heatmap_max > heatmap_min:

        heatmap = (
            heatmap - heatmap_min
        ) / (
            heatmap_max - heatmap_min
        )

    return heatmap.astype(
        np.float32
    )
# ============================================================
# CREATE JET HEATMAP
# ============================================================

def create_heatmap_image(
    heatmap,
    analysis_id
):
    """
    Convert Grad-CAM values into a JET heatmap.

    Color scale:

        Blue   = low activation
        Cyan   = low/medium activation
        Green  = medium activation
        Yellow = high activation
        Red    = strongest activation
    """

    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    heatmap_resized = cv2.resize(
        heatmap,
        IMG_SIZE,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # Convert 0-1 → 0-255
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # --------------------------------------------------------
    # Apply JET
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # OpenCV BGR → RGB
    # --------------------------------------------------------

    heatmap_rgb = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    heatmap_filename = (
        f"{analysis_id}_heatmap.png"
    )

    heatmap_path = (
        XAI_DIR / heatmap_filename
    )

    Image.fromarray(
        heatmap_rgb
    ).save(
        heatmap_path
    )

    # --------------------------------------------------------
    # Base64
    # --------------------------------------------------------

    heatmap_base64 = image_to_base64(
        Image.fromarray(
            heatmap_rgb
        )
    )

    return (
        heatmap_base64,
        str(heatmap_path)
    )


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_overlay(
    original_pil,
    heatmap,
    analysis_id
):
    """
    Create the final Grad-CAM visualization.

    The original microscopy image remains visible while
    the Grad-CAM activation is displayed over it.

    Original image:
        60%

    Grad-CAM:
        40%
    """

    # --------------------------------------------------------
    # Resize original image
    # --------------------------------------------------------

    original = (
        original_pil
        .convert("RGB")
        .resize(
            IMG_SIZE,
            Image.Resampling.BICUBIC
        )
    )

    original_rgb = np.array(
        original,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Resize Grad-CAM
    # --------------------------------------------------------

    heatmap_resized = cv2.resize(
        heatmap,
        IMG_SIZE,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # Convert Grad-CAM 0-1 → 0-255
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # --------------------------------------------------------
    # Apply JET colormap
    # --------------------------------------------------------

    heatmap_color_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # Convert heatmap BGR → RGB
    # --------------------------------------------------------

    heatmap_color_rgb = cv2.cvtColor(
        heatmap_color_bgr,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_rgb,
        0.60,
        heatmap_color_rgb,
        0.40,
        0
    )

    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    xai_filename = (
        f"{analysis_id}_xai.png"
    )

    xai_path = (
        XAI_DIR / xai_filename
    )

    Image.fromarray(
        overlay
    ).save(
        xai_path
    )

    # --------------------------------------------------------
    # Convert overlay to Base64
    # --------------------------------------------------------

    overlay_image = Image.fromarray(
        overlay
    )

    overlay_base64 = image_to_base64(
        overlay_image
    )

    return (
        overlay_base64,
        str(xai_path)
    )

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "SynthMicro API is running",
        "model": "resnet50_best.keras",
        "classes": CLASS_NAMES
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "resnet50_best.keras",
        "classes": CLASS_NAMES
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png"
    }

    if file.content_type not in allowed_types:

        return {
            "success": False,
            "error": (
                "Please upload a JPG, JPEG, "
                "or PNG image."
            )
        }


    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        contents = await file.read()

        if not contents:

            return {
                "success": False,
                "error": "Uploaded file is empty."
            }


        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")


        # Keep original
        original_image = image.copy()


        # ====================================================
        # GENERATE ANALYSIS ID
        # ====================================================

        analysis_id = uuid.uuid4().hex


        # ====================================================
        # SAVE ORIGINAL IMAGE
        # ====================================================

        upload_filename = (
            f"{analysis_id}.png"
        )

        upload_path = (
            UPLOAD_DIR / upload_filename
        )

        original_image.save(
            upload_path
        )


        # ====================================================
        # PREPROCESS
        # ====================================================

        input_array = preprocess_image(
            original_image
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        predictions = model.predict(
            input_array,
            verbose=0
        )


        probabilities = predictions[0]


        # ====================================================
        # FIND PREDICTED CLASS
        # ====================================================

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_label = (
            CLASS_NAMES[predicted_index]
        )


        # ====================================================
        # CONFIDENCE
        #
        # Returned as 0-1 because your React frontend
        # converts it to percentage.
        # ====================================================

        confidence = float(
            probabilities[predicted_index]
        )


        # ====================================================
        # ALL CLASS PROBABILITIES
        #
        # Also returned as 0-1.
        # ====================================================

        probability_dict = {

            CLASS_NAMES[i]:
                float(probabilities[i])

            for i in range(
                len(CLASS_NAMES)
            )
        }


        # ====================================================
        # GRAD-CAM
        # ====================================================

        heatmap = make_gradcam_heatmap(
            input_array,
            model
        )


        # ====================================================
        # CREATE STANDALONE HEATMAP
        # ====================================================

        heatmap_base64, heatmap_path = (
            create_heatmap_image(
                heatmap,
                analysis_id
            )
        )


        # ====================================================
        # CREATE OVERLAY
        # ====================================================

        overlay_base64, xai_path = (
            create_overlay(
                original_image,
                heatmap,
                analysis_id
            )
        )


        # ====================================================
        # SAVE DATABASE RECORD
        # ====================================================

        conn = sqlite3.connect(
            str(DB_PATH)
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO analyses (
                id,
                prediction,
                confidence,
                image_path,
                heatmap_path,
                xai_path,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,

                predicted_label,

                round(
                    confidence * 100,
                    2
                ),

                str(upload_path),

                heatmap_path,

                xai_path,

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        conn.commit()
        conn.close()


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "success": True,

            "analysis_id":
                analysis_id,

            "prediction":
                predicted_label,

            "confidence":
                confidence,

            "probabilities":
                probability_dict,

            # Existing frontend compatibility
            "gradcam":
                overlay_base64,

            # Standalone heatmap
            "gradcam_heatmap":
                heatmap_base64,

            # Overlay
            "gradcam_overlay":
                overlay_base64,

            # Compatibility with older frontend
            "xai_image":
                f"data:image/png;base64,{overlay_base64}"

        }


    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return {

            "success": False,

            "error":
                str(e)

        }


# ============================================================
# HISTORY ENDPOINT
# ============================================================

@app.get("/history")
def history():

    try:

        conn = sqlite3.connect(
            str(DB_PATH)
        )

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                prediction,
                confidence,
                image_path,
                heatmap_path,
                xai_path,
                created_at
            FROM analyses
            ORDER BY created_at DESC
            LIMIT 20
            """
        )

        rows = cursor.fetchall()

        conn.close()


        results = []


        for row in rows:

            # =================================================
            # ORIGINAL IMAGE
            # =================================================

            image_base64 = None

            if row["image_path"]:

                image_path = Path(
                    row["image_path"]
                )

                if image_path.exists():

                    with open(
                        image_path,
                        "rb"
                    ) as f:

                        image_base64 = (
                            "data:image/png;base64,"
                            +
                            base64.b64encode(
                                f.read()
                            ).decode("utf-8")
                        )


            # =================================================
            # HEATMAP
            # =================================================

            heatmap_base64 = None

            if row["heatmap_path"]:

                heatmap_path = Path(
                    row["heatmap_path"]
                )

                if heatmap_path.exists():

                    with open(
                        heatmap_path,
                        "rb"
                    ) as f:

                        heatmap_base64 = (
                            "data:image/png;base64,"
                            +
                            base64.b64encode(
                                f.read()
                            ).decode("utf-8")
                        )


            # =================================================
            # OVERLAY
            # =================================================

            overlay_base64 = None

            if row["xai_path"]:

                xai_path = Path(
                    row["xai_path"]
                )

                if xai_path.exists():

                    with open(
                        xai_path,
                        "rb"
                    ) as f:

                        overlay_base64 = (
                            "data:image/png;base64,"
                            +
                            base64.b64encode(
                                f.read()
                            ).decode("utf-8")
                        )


            # =================================================
            # HISTORY RESULT
            # =================================================

            results.append({

                "id":
                    row["id"],

                "prediction":
                    row["prediction"],

                "confidence":
                    row["confidence"],

                "created_at":
                    row["created_at"],

                "image":
                    image_base64,

                "heatmap":
                    heatmap_base64,

                "xai_image":
                    overlay_base64

            })


        return results


    except Exception as e:

        print(
            "History error:",
            str(e)
        )

        return {

            "success": False,

            "error":
                str(e)

        }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )