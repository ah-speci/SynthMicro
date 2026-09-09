import subprocess
import sys
import tempfile
import os
import io
import base64
import uuid
import sqlite3
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Optional heavy dependencies are imported lazily where possible so the API
# can start with a useful error message if one of them is missing
cellpose_models = None
CELLPPOSE_IMPORT_ERROR = None

try:
    import shap
except Exception as exc:
    shap = None
    SHAP_IMPORT_ERROR = str(exc)

try:
    from google import genai
    from google.genai import types as genai_types
except Exception as exc:
    genai = None
    genai_types = None
    GEMINI_IMPORT_ERROR = str(exc)

from tensorflow.keras.applications.resnet50 import preprocess_input

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="SynthMicro API",
    description=(
        "Whole blood-smear analysis using Cellpose, ResNet50, "
        "Grad-CAM, SHAP and an optional Hugging Face VLM report."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Phase 7 notebook uses this final 384x384 model.
MODEL_PATH = BASE_DIR / "models" / "resnet50_frozen_384.keras"

# Fallback for an older checkout that has only the original model.
FALLBACK_MODEL_PATH = BASE_DIR / "models" / "resnet50_best.keras"

UPLOAD_DIR = BASE_DIR / "uploads"
XAI_DIR = BASE_DIR / "xai_results"
DB_PATH = BASE_DIR / "synthmicro.db"

IMG_SIZE = 384
CLASS_NAMES = [
    "basophil",
    "erythroblast",
    "monocyte",
    "myeloblast",
    "seg_neutrophil",
]

PURPLE_THRESHOLD = 35.0
MAX_XAI_CELLS = 3
SHAP_BACKGROUND_SIZE = 2

# Fast multimodal explanation layer.
# Flash-Lite was verified with the project's test image at ~2 seconds.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MAX_OUTPUT_TOKENS = 180
GEMINI_THINKING_LEVEL = "minimal"
GEMINI_STATUS_DIR = XAI_DIR / "gemini_status"
GEMINI_STATUS_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
XAI_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def init_db() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT,
            heatmap_path TEXT,
            xai_path TEXT,
            report_path TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    # Existing synthmicro.db may have been created by the older backend.
    # Add the new report_path column without destroying old data.
    cur.execute("PRAGMA table_info(analyses)")
    columns = {row[1] for row in cur.fetchall()}
    if "report_path" not in columns:
        cur.execute("ALTER TABLE analyses ADD COLUMN report_path TEXT")

    conn.commit()
    conn.close()


init_db()


# ============================================================
# LOAD RESNET50
# ============================================================

if MODEL_PATH.exists():
    ACTIVE_MODEL_PATH = MODEL_PATH
elif FALLBACK_MODEL_PATH.exists():
    ACTIVE_MODEL_PATH = FALLBACK_MODEL_PATH
else:
    raise FileNotFoundError(
        "No ResNet50 model found. Expected either:\n"
        f"  {MODEL_PATH}\n"
        f"  {FALLBACK_MODEL_PATH}"
    )

# Keep TensorFlow from reserving the entire GPU up front.
# This is especially useful on the 6-GB RTX 4050 because SHAP and
# ResNet50 temporarily need additional VRAM during gradient calculation.
try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow GPU memory growth enabled:", gpu)
except Exception as exc:
    print("GPU memory-growth configuration warning:", exc)

print("=" * 70)
print("SynthMicro Phase 7 backend")
print(f"Loading model: {ACTIVE_MODEL_PATH}")
print("=" * 70)

model = tf.keras.models.load_model(
    ACTIVE_MODEL_PATH,
    compile=False,
)

print("Model loaded:", ACTIVE_MODEL_PATH.name)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# SMALL UTILITIES
# ============================================================

def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def file_to_data_url(path: Path) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def prepare_cell(crop_bgr: np.ndarray) -> np.ndarray:
    """384x384 aspect-ratio preserving cell crop with white padding."""
    h, w = crop_bgr.shape[:2]
    scale = min(IMG_SIZE / max(w, 1), IMG_SIZE / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        crop_bgr,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.ones(
        (IMG_SIZE, IMG_SIZE, 3),
        dtype=np.uint8,
    ) * 255

    x0 = (IMG_SIZE - new_w) // 2
    y0 = (IMG_SIZE - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def calculate_purple_score(crop_bgr: np.ndarray) -> float:
    """Same purple/blue score used by the Phase 7 notebook."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    purple = ((r + b) / 2.0) - g
    foreground = np.mean(rgb, axis=2) < 245
    if np.sum(foreground) == 0:
        return 0.0
    return float(np.mean(purple[foreground]))


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ============================================================
# CELLPPOSE SEGMENTATION
# ============================================================

def segment_smear(smear_bgr: np.ndarray):
    """
    Run Cellpose in a completely separate Python process.

    This is intentional: TensorFlow/ResNet and Cellpose/PyTorch
    can cause native-library conflicts when loaded in the same process.
    """

    with tempfile.TemporaryDirectory(prefix="synthmicro_cellpose_") as tmpdir:

        tmpdir = Path(tmpdir)

        input_path = tmpdir / "smear.png"
        output_path = tmpdir / "masks.npy"

        # Save image for the Cellpose subprocess.
        success = cv2.imwrite(
            str(input_path),
            smear_bgr
        )

        if not success:
            raise RuntimeError(
                "Failed to create temporary image for Cellpose."
            )

        worker_path = Path(__file__).resolve().parent / "cellpose_worker.py"

        command = [
            sys.executable,
            str(worker_path),
            str(input_path),
            str(output_path),
        ]

        print("=" * 60)
        print("Starting isolated Cellpose process...")
        print("Command:", command)
        print("=" * 60)

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print("Cellpose process output:")
        print(process.stdout)

        if process.returncode != 0:
            raise RuntimeError(
                "Cellpose subprocess failed.\n\n"
                + process.stdout
            )

        if not output_path.exists():
            raise RuntimeError(
                "Cellpose finished but did not produce a mask file."
            )

        masks = np.load(output_path)

        print(
            f"Cellpose completed successfully: "
            f"{int(masks.max())} objects"
        )

        return masks


def make_cell_crops(smear_bgr: np.ndarray, masks: np.ndarray):
    crops = []
    info = []

    num_objects = int(masks.max())

    for cell_id in range(1, num_objects + 1):
        ys, xs = np.where(masks == cell_id)
        if len(xs) == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

        # Same small padding used in the notebook.
        pad = 10
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(smear_bgr.shape[1], x2 + pad)
        y2p = min(smear_bgr.shape[0], y2 + pad)

        crop = smear_bgr[y1p:y2p, x1p:x2p].copy()
        crop_mask = masks[y1p:y2p, x1p:x2p] == cell_id

        # Remove neighboring cells/background from the crop.
        clean = crop.copy()
        clean[~crop_mask] = 255

        area = int(np.sum(crop_mask))
        width = int(x2 - x1)
        height = int(y2 - y1)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        crops.append(clean)
        info.append(
            {
                "cell_id": cell_id,
                "area": area,
                "x": cx,
                "y": cy,
                "width": width,
                "height": height,
            }
        )

    return crops, pd.DataFrame(info)


# ============================================================
# RESNET CLASSIFICATION
# ============================================================

def classify_cells(cell_crops):
    if not cell_crops:
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), np.empty((0, 5))

    X = np.stack([prepare_cell(c) for c in cell_crops])
    X_preprocessed = preprocess_input(X.astype(np.float32))

    predictions = model.predict(
        X_preprocessed,
        batch_size=16,
        verbose=0,
    )

    return X, X_preprocessed, predictions


# ============================================================
# GRAD-CAM — SAME HEAD AS PHASE 7 NOTEBOOK
# ============================================================

base_model = model.get_layer("resnet50")
target_layer = base_model.get_layer("conv5_block3_out")
feature_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=target_layer.output,
)


def generate_gradcam(image_input: np.ndarray, class_index: int) -> np.ndarray:
    with tf.GradientTape() as tape:
        conv_outputs = feature_model(
            image_input,
            training=False,
        )

        x = tf.reduce_mean(
            conv_outputs,
            axis=[1, 2],
        )
        x = model.get_layer("dropout_2")(x, training=False)
        x = model.get_layer("dense_2")(x)
        x = model.get_layer("dropout_3")(x, training=False)
        predictions_out = model.get_layer("dense_3")(x)
        class_score = predictions_out[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    conv = conv_outputs[0]
    weights = pooled_grads[0]

    heatmap = tf.reduce_sum(conv * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy().astype(np.float32)


def save_gradcam(cell_bgr: np.ndarray, heatmap: np.ndarray, path: Path):
    heatmap_resized = cv2.resize(
        heatmap,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    original_rgb = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )

    # Ensure both images are exactly the same size and 3-channel RGB
    original_rgb = np.asarray(original_rgb)
    
    if original_rgb.ndim == 2:
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)
    elif original_rgb.shape[2] == 4:
        original_rgb = original_rgb[:, :, :3]
    
    heatmap_rgb = np.asarray(heatmap_rgb)
    
    if heatmap_rgb.ndim == 2:
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_GRAY2RGB)
    elif heatmap_rgb.shape[2] == 4:
        heatmap_rgb = heatmap_rgb[:, :, :3]
    
    heatmap_rgb = cv2.resize(
        heatmap_rgb,
        (original_rgb.shape[1], original_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    
    original_rgb = original_rgb.astype(np.uint8)
    heatmap_rgb = heatmap_rgb.astype(np.uint8)
    
    overlay = cv2.addWeighted(
        original_rgb,
        0.60,
        heatmap_rgb,
        0.40,
        0,
    )

    # 3-panel evidence image, matching the notebook's intent.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_shap(cell_bgr: np.ndarray, shap_display: np.ndarray, path: Path, label: str, confidence: float):
    original_rgb = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_rgb)
    axes[0].set_title(f"Original\n{label} ({confidence * 100:.1f}%)")
    axes[0].axis("off")
    axes[1].imshow(shap_display, cmap="hot")
    axes[1].set_title("SHAP Attribution Magnitude")
    axes[1].axis("off")
    axes[2].imshow(original_rgb)
    axes[2].imshow(shap_display, cmap="hot", alpha=0.5)
    axes[2].set_title("SHAP Overlay")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_shap_records(X_preprocessed, X, cell_df, selected_df, output_dir: Path):
    """
    Generate SHAP explanations for the selected cells.

    This configuration is deliberately tuned for a 6-GB GPU:
      * only a tiny background is used;
      * GradientExplainer's internal batch_size is forced to 1;
      * only the top-ranked model output is explained;
      * nsamples is kept modest.

    The important fix is ``batch_size=1`` on GradientExplainer itself.
    SHAP 0.52 accepts this in the constructor, while ``shap_values()``
    does not expose a batch_size argument.
    """

    if shap is None:
        return [], f"SHAP is unavailable: {SHAP_IMPORT_ERROR}"

    if len(selected_df) == 0 or len(X_preprocessed) == 0:
        return [], "No selected cells available for SHAP."

    background = X_preprocessed[:min(SHAP_BACKGROUND_SIZE, len(X_preprocessed))]

    print(
        f"SHAP: background={background.shape}, "
        f"selected_cells={len(selected_df)}"
    )

    # SHAP 0.52 constructor supports batch_size; setting it here prevents
    # the default batch of 50 from creating a large [50, 512, 48, 48]
    # activation tensor inside ResNet50.
    explainer = shap.GradientExplainer(
        model,
        background,
        batch_size=1,
        local_smoothing=0,
    )

    records = []

    for _, row in selected_df.iterrows():
        cell_id = int(row["cell_id"])
        matches = cell_df.index[cell_df["cell_id"] == cell_id]
        if len(matches) == 0:
            print(f"SHAP: Cell {cell_id} not found in cell dataframe; skipping.")
            continue

        idx = int(matches[0])
        image_input = X_preprocessed[idx:idx + 1]

        predicted_index = int(np.argmax(row["prediction_vector"]))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(row["confidence"])

        print(
            f"SHAP: Cell {cell_id} -> {predicted_class} "
            f"({confidence * 100:.1f}%), starting..."
        )

        # Explain only the highest-ranked output. For this classifier, the
        # highest-ranked output is the same class already stored in the row.
        shap_result = explainer.shap_values(
            image_input,
            nsamples=50,
            ranked_outputs=1,
            output_rank_order="max",
            rseed=0,
            return_variances=False,
        )

        # With ranked_outputs=1, SHAP returns:
        #   (shap_values, indexes)
        # The exact ndarray/list shape differs slightly between SHAP versions,
        # so normalize it defensively.
        if (
            isinstance(shap_result, tuple)
            and len(shap_result) == 2
        ):
            values_all, indexes = shap_result
            values_array = np.asarray(values_all)
            indexes_array = np.asarray(indexes)

            # For one ranked output the class dimension is the final axis.
            # Typical SHAP 0.52 output is (1, H, W, C, 1).
            if values_array.ndim == 5:
                values = values_array[0, :, :, :, 0]
            elif values_array.ndim == 4:
                values = values_array[0]
            elif values_array.ndim == 3:
                values = values_array
            else:
                raise RuntimeError(
                    f"Unexpected ranked SHAP output shape: {values_array.shape}"
                )

            # Sanity-check the ranked class when SHAP supplies indexes.
            try:
                shap_class_index = int(indexes_array.reshape(-1)[0])
                if shap_class_index != predicted_index:
                    print(
                        f"SHAP warning: ranked class {shap_class_index} differs "
                        f"from predicted class {predicted_index}."
                    )
            except Exception:
                pass

        elif isinstance(shap_result, list):
            # Compatibility path for older SHAP TensorFlow return formats.
            class_values = np.asarray(shap_result[predicted_index])
            if class_values.ndim == 4:
                values = class_values[0]
            else:
                values = class_values

        else:
            values_array = np.asarray(shap_result)
            if values_array.ndim == 5:
                values = values_array[0, :, :, :, predicted_index]
            elif values_array.ndim == 4:
                values = values_array[0]
            elif values_array.ndim == 3:
                values = values_array
            else:
                raise RuntimeError(
                    f"Unexpected SHAP output shape: {values_array.shape}"
                )

        if values.ndim == 3:
            shap_magnitude = np.mean(np.abs(values), axis=-1)
        else:
            shap_magnitude = np.abs(values)

        low = np.percentile(shap_magnitude, 1)
        high = np.percentile(shap_magnitude, 99)

        shap_display = np.clip(
            (shap_magnitude - low) / (high - low + 1e-8),
            0,
            1,
        )

        path = output_dir / f"cell_{cell_id}_shap.png"

        save_shap(
            X[idx],
            shap_display,
            path,
            predicted_class,
            confidence,
        )

        records.append(
            {
                "cell_id": cell_id,
                "class": predicted_class,
                "confidence": confidence,
                "path": str(path),
            }
        )

        # Release temporary references before the next cell.
        del shap_result
        del image_input
        print(f"SHAP: Cell {cell_id} finished -> {path}")

    return records, None


# ============================================================
# VLM REPORT EXPLANATIONS
# ============================================================

def gemini_explain_image(image_path: Path, prompt: str) -> str:
    """Fast Gemini multimodal explanation using the standard image API."""
    if genai is None or genai_types is None:
        raise RuntimeError(f"google-genai is unavailable: {GEMINI_IMPORT_ERROR}")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=genai_types.HttpOptions(timeout=30000),
    )
    try:
        image_bytes = image_path.read_bytes()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Part.from_text(text=prompt),
                genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=("image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"),
                ),
            ],
            config=genai_types.GenerateContentConfig(
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                thinking_config=genai_types.ThinkingConfig(
                    thinking_level=GEMINI_THINKING_LEVEL,
                ),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text
    finally:
        client.close()


def fallback_cell_explanation(row: dict) -> str:
    label = row["predicted_class"]
    confidence = row["confidence"] * 100
    if label == "myeloblast":
        return (
            f"The ResNet50 model classified this segmented candidate as a "
            f"myeloblast with {confidence:.1f}% confidence. This is a model-level "
            "cell classification; a myeloblast-like prediction alone does not "
            "establish AML or another cancer diagnosis."
        )
    return (
        f"The ResNet50 model classified this segmented candidate as {label} "
        f"with {confidence:.1f}% confidence. The result describes the model's "
        "learned image category and is not a clinical diagnosis."
    )


def _gemini_status_path(analysis_id: str) -> Path:
    return GEMINI_STATUS_DIR / f"{analysis_id}.json"


def write_gemini_status(analysis_id: str, payload: dict) -> None:
    path = _gemini_status_path(analysis_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_vlm_explanations(selected_records, cell_df):
    """Return instant fallback text; real Gemini work runs in background."""
    explanations = {}
    for rec in selected_records:
        cell_id = rec["cell_id"]
        row = cell_df.loc[cell_df["cell_id"] == cell_id].iloc[0].to_dict()
        explanations[cell_id] = fallback_cell_explanation(row)
    return explanations


def run_gemini_background(analysis_id: str, gradcam_records: list, cell_df: pd.DataFrame) -> None:
    """Generate Gemini explanations concurrently without blocking /predict."""
    try:
        cells = []
        for rec in gradcam_records:
            cell_id = int(rec["cell_id"])
            row = cell_df.loc[cell_df["cell_id"] == cell_id].iloc[0].to_dict()
            cells.append({
                "cell_id": cell_id,
                "class": str(row["predicted_class"]),
                "confidence": float(row["confidence"]),
                "image_path": str(rec["original_path"]),
            })

        status = {
            "analysis_id": analysis_id,
            "status": "processing" if cells else "completed",
            "model": GEMINI_MODEL,
            "results": {
                str(c["cell_id"]): {
                    "cell_id": c["cell_id"],
                    "status": "queued",
                    "text": None,
                    "class": c["class"],
                    "confidence": c["confidence"],
                }
                for c in cells
            },
        }
        write_gemini_status(analysis_id, status)

        def worker(cell):
            prompt = (
                f"Describe this segmented blood-smear cell for a research-oriented "
                f"computer-vision system. The ResNet50 model predicts '{cell['class']}' "
                f"with {cell['confidence'] * 100:.1f}% confidence. "
                "In 3-4 concise sentences, describe only visible morphology, "
                "staining/color, nucleus/cytoplasm appearance, and image-quality "
                "limitations. Explain how visible features could relate to the "
                "model prediction, but do not diagnose leukemia, AML, cancer, or "
                "any disease. Do not invent morphology."
            )
            try:
                text = gemini_explain_image(Path(cell["image_path"]), prompt)
                return cell["cell_id"], text, None
            except Exception as exc:
                return cell["cell_id"], None, str(exc)

        with ThreadPoolExecutor(max_workers=min(3, max(1, len(cells)))) as executor:
            futures = [executor.submit(worker, cell) for cell in cells]
            for future in as_completed(futures):
                cell_id, text, error = future.result()
                key = str(cell_id)
                status = json.loads(_gemini_status_path(analysis_id).read_text(encoding="utf-8"))
                status["results"][key]["status"] = "completed" if text else "error"
                status["results"][key]["text"] = text
                if error:
                    status["results"][key]["error"] = error
                write_gemini_status(analysis_id, status)

        final = json.loads(_gemini_status_path(analysis_id).read_text(encoding="utf-8"))
        final["status"] = "completed"
        write_gemini_status(analysis_id, final)
        print(f"Gemini background analysis complete: {analysis_id}")
    except Exception as exc:
        print(f"Gemini background analysis failed: {analysis_id}: {exc}")
        write_gemini_status(analysis_id, {
            "analysis_id": analysis_id,
            "status": "error",
            "model": GEMINI_MODEL,
            "error": str(exc),
            "results": {},
        })


# ============================================================
# PDF REPORT
# ============================================================

def make_pdf_report(
    analysis_id: str,
    smear_path: Path,
    overlay_path: Path,
    cell_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    myeloblast_df: pd.DataFrame,
    gradcam_records: list,
    shap_records: list,
    explanations: dict,
    output_path: Path,
):
    """Create a detailed, research-oriented XAI report.

    The report is intentionally explanatory rather than diagnostic.  It documents
    what the pipeline measured, how candidate cells were selected, what the
    classifier predicted, and how Grad-CAM/SHAP should be interpreted.
    """
    from xml.sax.saxutils import escape as xml_escape

    total_segmented = len(cell_df)
    candidate_count = len(candidate_df)
    myeloblast_count = len(myeloblast_df)
    proportion = (
        myeloblast_count / candidate_count * 100
        if candidate_count > 0
        else 0.0
    )

    # ---------- Styles ----------
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["BodyText"], fontSize=10,
        leading=14, textColor=colors.HexColor("#4b5563"), spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SmallNote", parent=styles["BodyText"], fontSize=8.2,
        leading=10.5, textColor=colors.HexColor("#5b6470"),
    ))
    styles.add(ParagraphStyle(
        name="SectionIntro", parent=styles["BodyText"], fontSize=9.5,
        leading=14, textColor=colors.HexColor("#374151"), spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CellHeading", parent=styles["Heading2"], fontSize=13,
        leading=16, spaceBefore=6, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="BoxText", parent=styles["BodyText"], fontSize=9,
        leading=13, leftIndent=8, rightIndent=8, spaceBefore=5, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Metric", parent=styles["BodyText"], fontSize=15,
        leading=18, alignment=1, spaceAfter=2,
    ))

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=36, leftMargin=36, topMargin=48, bottomMargin=42,
        title=f"SynthMicro XAI Report - {analysis_id}",
        author="SynthMicro",
    )

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d1d5db"))
        canvas.line(36, 30, A4[0] - 36, 30)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawString(36, 19, "SynthMicro | Research-oriented computer-vision report | Not a clinical diagnosis")
        canvas.drawRightString(A4[0] - 36, 19, f"Page {doc_obj.page}")
        canvas.restoreState()

    story = []

    def section(title, intro=None):
        story.append(Paragraph(title, styles["Heading1"]))
        if intro:
            story.append(Paragraph(intro, styles["SectionIntro"]))

    def info_box(title, body, background="#f3f6fa"):
        data = [[
            Paragraph(f"<b>{xml_escape(title)}</b>", styles["BodyText"]),
            Paragraph(body, styles["BoxText"]),
        ]]
        t = Table(data, colWidths=[1.45 * inch, 4.95 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(t)
        story.append(Spacer(1, 9))

    def pct(v):
        return f"{float(v) * 100:.1f}%"

    # ---------- 1. Executive summary ----------
    story.append(Paragraph("SynthMicro — Whole Blood Smear XAI Report", styles["Title"]))
    story.append(Paragraph(
        "Detailed analysis of segmentation, cell classification and explainable-AI evidence",
        styles["ReportSubtitle"],
    ))
    story.append(Paragraph(
        f"<b>Analysis ID:</b> {xml_escape(analysis_id)}<br/>"
        f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        "<b>Input:</b> uploaded blood-smear microscopy image",
        styles["SmallNote"],
    ))
    story.append(Spacer(1, 12))

    pipeline_text = (
        "<b>Pipeline:</b> Cellpose segmentation &rarr; candidate-cell extraction at 384&times;384 &rarr; "
        "ResNet50 five-class classification &rarr; Grad-CAM &rarr; SHAP &rarr; explanatory report."
    )
    story.append(Paragraph(pipeline_text, styles["BodyText"]))
    story.append(Spacer(1, 12))

    summary = [
        ["Measurement", "Result", "What it means"],
        ["Cellpose objects", str(total_segmented), "Objects segmented from the uploaded smear."],
        ["Candidate cells", str(candidate_count), "Objects passing the purple-score candidate-selection rule."],
        ["Myeloblast-like candidates", str(myeloblast_count), "Candidates whose ResNet50 top class was myeloblast."],
        ["Myeloblast-like proportion", f"{proportion:.2f}%", "Myeloblast-like candidates divided by candidate cells."],
    ]
    table = Table(summary, colWidths=[1.75 * inch, 1.1 * inch, 3.75 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#9ca3af")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 13))

    info_box(
        "How to read this result",
        "The numbers above describe the behavior of a computer-vision pipeline on this image. "
        "A high classifier confidence means the ResNet50 model strongly preferred one of its "
        "available image classes; it does <b>not</b> mean a high probability of leukemia, AML, "
        "or any other disease. The candidate percentage is also not a cancer prevalence estimate.",
        "#fff8e7",
    )
    info_box(
        "Important safety interpretation",
        "This report is intended for research, model evaluation and screening-oriented exploration. "
        "It does not establish a diagnosis. Clinical interpretation requires expert hematopathology, "
        "appropriate laboratory testing and the complete clinical context.",
        "#fef2f2",
    )
    story.append(PageBreak())

    # ---------- 2. Methodology ----------
    section(
        "Analysis Methodology",
        "The following sections explain what each stage contributes and where uncertainty can enter the pipeline.",
    )
    method_rows = [
        ["Stage", "Operation", "Output used downstream"],
        ["1. Cellpose", "Segments visible cell/object regions in the whole smear.", "Object masks and cell geometry."],
        ["2. Candidate selection", "Computes a purple-score and retains objects at or above the configured threshold.", "Candidate-cell subset for reporting."],
        ["3. ResNet50", "Resizes each selected crop to 384×384 and predicts one of five closed-set classes.", "Class label and softmax confidence."],
        ["4. Grad-CAM", "Highlights image regions that contribute to the selected class activation.", "Spatial explanation of model behavior."],
        ["5. SHAP", "Estimates pixel-level attribution magnitude for the selected prediction.", "Pixel attribution explanation."],
        ["6. Gemini", "Optional multimodal commentary describes visible morphology for selected cells.", "Separate descriptive commentary; not a diagnosis."],
    ]
    mt = Table(method_rows, colWidths=[1.1 * inch, 3.0 * inch, 2.5 * inch], repeatRows=1)
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(mt)
    story.append(Spacer(1, 12))

    info_box(
        "Candidate-selection rule",
        "Only segmented objects with the configured purple-score threshold are treated as candidate "
        "cells for the screening indicator. This is an image-processing rule, not a laboratory definition "
        "of a nucleated cell. Objects outside the threshold can therefore be missed, while visually similar "
        "objects can be retained.",
    )
    info_box(
        "Closed-set classifier limitation",
        "The ResNet50 model has five outputs: <b>basophil, erythroblast, monocyte, myeloblast, and "
        "segmented neutrophil</b>. RBCs and unknown cell types are not explicit outputs, so an unsuitable "
        "input can still be assigned one of these five labels.",
    )
    story.append(PageBreak())

    # ---------- 3. Whole smear ----------
    section("Whole Smear", "Original uploaded image used as the input to the segmentation stage.")
    story.append(RLImage(str(smear_path), width=6.7 * inch, height=5.1 * inch))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Visual context: the complete field contains numerous erythrocytes and several strongly stained "
        "nucleated-looking objects. Downstream results depend on what Cellpose can segment reliably in this field.",
        styles["SectionIntro"],
    ))
    story.append(PageBreak())

    # ---------- 4. Segmentation ----------
    section(
        "Segmentation and Candidate Overlay",
        "Labels show the candidate-cell ID, predicted class and classifier confidence used by the report.",
    )
    story.append(RLImage(str(overlay_path), width=6.7 * inch, height=5.1 * inch))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Cellpose produced <b>{total_segmented}</b> objects. <b>{candidate_count}</b> met the candidate "
        f"selection rule, and <b>{myeloblast_count}</b> of those were classified as myeloblast by the model. "
        "The overlay is a bookkeeping and visualization aid; it does not certify that every label corresponds "
        "to a biologically correct cell type.",
        styles["SectionIntro"],
    ))
    story.append(PageBreak())

    # ---------- 5. Candidate results ----------
    section(
        "Cell-Level Results",
        "Candidate measurements and model predictions. Area is the segmented-pixel count; purple score is the "
        "candidate-selection feature; confidence is the model's top softmax output.",
    )
    rows = [["ID", "Area", "X", "Y", "Purple", "Class", "Confidence"]]
    for _, row in candidate_df.sort_values("cell_id").iterrows():
        rows.append([
            str(int(row["cell_id"])), str(int(row["area"])), str(int(row["x"])), str(int(row["y"])),
            f"{row['purple_score']:.1f}", str(row["predicted_class"]), pct(row["confidence"]),
        ])
    results_table = Table(rows, colWidths=[0.45*inch, 0.62*inch, 0.48*inch, 0.48*inch, 0.62*inch, 1.65*inch, 0.82*inch], repeatRows=1)
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9ca3af")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 12))

    distribution = candidate_df["predicted_class"].value_counts().to_dict() if candidate_count else {}
    dist_rows = [["Predicted class", "Candidate count", "Share of candidates"]]
    for cls in CLASS_NAMES:
        count = int(distribution.get(cls, 0))
        share = count / candidate_count * 100 if candidate_count else 0.0
        dist_rows.append([cls, str(count), f"{share:.1f}%"])
    dt = Table(dist_rows, colWidths=[2.2*inch, 1.6*inch, 2.0*inch], repeatRows=1)
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9ca3af")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Paragraph("Candidate class distribution", styles["Heading3"]))
    story.append(dt)
    story.append(Spacer(1, 10))

    info_box(
        "Confidence versus probability of disease",
        "For each crop, the reported confidence is the largest value among the five ResNet50 output "
        "classes. It is a measure of model preference within this closed label set. It has not been presented "
        "as a calibrated probability of AML, leukemia or any other clinical outcome.",
    )
    story.append(PageBreak())

    # ---------- 6. Per-cell probability detail ----------
    section(
        "Per-Cell Model Probability Detail",
        "The table below exposes the complete five-class output for each candidate when that vector is available.",
    )
    prob_rows = [["Cell", "Basophil", "Erythroblast", "Monocyte", "Myeloblast", "Seg. neutrophil"]]
    for _, row in candidate_df.sort_values("cell_id").iterrows():
        vec = row.get("prediction_vector")
        if vec is None:
            values = ["—"] * 5
        else:
            try:
                values = [f"{float(v)*100:.1f}%" for v in list(vec)[:5]]
                if len(values) < 5:
                    values += ["—"] * (5-len(values))
            except Exception:
                values = ["—"] * 5
        prob_rows.append([str(int(row["cell_id"]))] + values)
    pt = Table(prob_rows, colWidths=[0.55*inch, 1.1*inch, 1.15*inch, 1.0*inch, 1.1*inch, 1.25*inch], repeatRows=1)
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9ca3af")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pt)
    story.append(Spacer(1, 12))
    info_box(
        "Why show all five outputs?",
        "A single top-class label can hide how close the model was to another class. The full vector gives "
        "a better picture of the model's internal competition between the five available labels. Even so, these "
        "values remain model outputs and should not be interpreted as biological probabilities.",
    )
    story.append(PageBreak())

    # ---------- 7. Grad-CAM ----------
    section(
        "Grad-CAM Explanations",
        "Grad-CAM is presented here as a visual audit of the ResNet50 decision. Each cell receives a dedicated "
        "report page so the reader can examine the original crop, the heatmap and the overlay together with a "
        "detailed explanation of what the visualization can and cannot establish.",
    )

    if gradcam_records:
        for idx, rec in enumerate(gradcam_records):
            cell_id = int(rec["cell_id"])
            cls = str(rec["class"])
            conf = float(rec["confidence"])
            row = cell_df.loc[cell_df["cell_id"] == cell_id].iloc[0]
            area = int(row["area"])
            purple = float(row["purple_score"])
            shap_present = any(int(s["cell_id"]) == cell_id for s in shap_records)

            if idx > 0:
                story.append(PageBreak())

            story.append(Paragraph(
                f"Cell {cell_id} — {xml_escape(cls)} prediction ({pct(conf)})",
                styles["CellHeading"],
            ))

            metrics = [
                ["Cell ID", str(cell_id), "Predicted class", xml_escape(cls)],
                ["Segmented area", f"{area:,} pixels", "Purple score", f"{purple:.1f}"],
                ["Top-class confidence", pct(conf), "SHAP available", "Yes" if shap_present else "No"],
            ]
            metric_table = Table(metrics, colWidths=[1.25*inch, 1.55*inch, 1.45*inch, 2.25*inch])
            metric_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef2f7")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c4cbd4")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(metric_table)
            story.append(Spacer(1, 9))

            story.append(RLImage(rec["path"], width=6.7 * inch, height=2.35 * inch))
            story.append(Spacer(1, 8))

            story.append(Paragraph(
                "<b>What the three panels represent.</b> The <b>Original</b> panel is the 384×384 image crop "
                "presented to the classifier after candidate-cell extraction. The <b>Grad-CAM Heatmap</b> converts "
                "the class-specific activation information into a spatial visualization. Brighter or warmer regions "
                "represent areas with stronger contribution in the Grad-CAM representation. The <b>Grad-CAM Overlay</b> "
                "places that information back over the cell image so the reader can judge whether the highlighted "
                "areas appear to lie inside the segmented object or near its boundary/background. These panels should "
                "always be interpreted together rather than treating the heatmap alone as a biological map.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            story.append(Paragraph(
                f"<b>What the model decided.</b> For Cell {cell_id}, ResNet50 selected <b>{xml_escape(cls)}</b> "
                f"as its highest-scoring class with a reported confidence of <b>{pct(conf)}</b>. The segmented "
                f"object contains <b>{area:,} pixels</b>, and its purple score is <b>{purple:.1f}</b>, which is "
                "the image-processing feature used by this pipeline when selecting candidate cells. These values "
                "describe the computational pathway leading to the prediction. They are not measurements of disease "
                "severity, blast percentage in blood or bone marrow, or a probability of AML.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            story.append(Paragraph(
                "<b>How to read the highlighted regions.</b> The most useful question is whether the strongest "
                "activation is concentrated on the segmented cell rather than on empty background, neighboring "
                "erythrocytes, image borders, debris or other unrelated structures. A heatmap that broadly follows "
                "the cell can provide more intuitive evidence that the model used information contained within the "
                "candidate crop. Conversely, strong activation outside the cell or at a visually unusual artifact "
                "can indicate that the prediction may be sensitive to contextual or acquisition-related features. "
                "Grad-CAM cannot by itself determine whether a highlighted area is a nucleus, nucleolus, chromatin "
                "pattern, cytoplasmic feature or another specific hematological structure.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            story.append(Paragraph(
                "<b>Relationship to the cell-level measurement.</b> The segmented area and purple score explain why "
                "the object entered the downstream analysis, while the ResNet50 prediction describes the model's "
                "preferred class after the crop was created. Grad-CAM adds a spatial explanation of that decision. "
                "Keeping these stages separate is important: a correctly drawn mask does not guarantee a correct "
                "classification, and a visually plausible heatmap does not validate the segmentation or the class "
                "label. The evidence is therefore best understood as a chain of computational observations.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            story.append(Paragraph(
                "<b>Cross-check recommended.</b> Compare this Grad-CAM result with the original smear and, where "
                "available, the SHAP attribution for the same cell. Look for consistency in the general location of "
                "model-relevant pixels, while remembering that Grad-CAM and SHAP use different explanation mechanisms. "
                "Agreement can make the model decision easier to audit, but disagreement is not automatically evidence "
                "that the model is wrong; it is a reason to inspect the crop, segmentation and prediction more closely.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            if explanations.get(cell_id):
                story.append(Paragraph(
                    f"<b>Automated descriptive commentary.</b> {xml_escape(str(explanations[cell_id]))}",
                    styles["BodyText"],
                ))
                story.append(Spacer(1, 7))

            story.append(Paragraph(
                "<b>Interpretive boundary.</b> A myeloblast prediction, even when the classifier confidence is very "
                "high, remains a learned image-classification output within a five-class closed set. It is not a "
                "confirmed morphological diagnosis and should not be converted into a leukemia or AML probability. "
                "Clinical interpretation requires appropriate hematopathology review, laboratory testing and the "
                "broader clinical context.",
                styles["BodyText"],
            ))

    else:
        story.append(Paragraph(
            "No Grad-CAM records were available for this analysis. This means the local visual explanation stage "
            "did not produce a usable record for the selected candidates; it should not be interpreted as evidence "
            "that the classifier had no basis for its prediction.",
            styles["BodyText"],
        ))
    story.append(PageBreak())

    # ---------- 8. SHAP ----------
    section(
        "SHAP Explanations",
        "SHAP provides a second, pixel-level view of model attribution. Each available cell receives a dedicated "
        "page with an explanation of attribution magnitude, how to inspect the overlay and how SHAP should be "
        "distinguished from a biological or clinical measurement.",
    )

    if shap_records:
        for idx, rec in enumerate(shap_records):
            cell_id = int(rec["cell_id"])
            cls = str(rec["class"])
            conf = float(rec["confidence"])
            row = cell_df.loc[cell_df["cell_id"] == cell_id].iloc[0]
            area = int(row["area"])
            purple = float(row["purple_score"])

            if idx > 0:
                story.append(PageBreak())

            story.append(Paragraph(
                f"Cell {cell_id} — SHAP attribution for {xml_escape(cls)} ({pct(conf)})",
                styles["CellHeading"],
            ))

            metrics = [
                ["Cell ID", str(cell_id), "Predicted class", xml_escape(cls)],
                ["Segmented area", f"{area:,} pixels", "Purple score", f"{purple:.1f}"],
                ["Model confidence", pct(conf), "Explanation type", "SHAP pixel attribution"],
            ]
            metric_table = Table(metrics, colWidths=[1.25*inch, 1.55*inch, 1.45*inch, 2.25*inch])
            metric_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef2f7")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c4cbd4")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(metric_table)
            story.append(Spacer(1, 9))

            story.append(RLImage(rec["path"], width=6.7 * inch, height=2.35 * inch))
            story.append(Spacer(1, 8))

            story.append(Paragraph(
                "<b>What SHAP is showing.</b> The Original panel is the exact candidate crop used for the "
                "classification/explanation process. The SHAP Attribution Magnitude panel visualizes the strength "
                "of pixel-level attribution associated with the selected class. The SHAP Overlay combines the "
                "attribution visualization with the original image, making it easier to judge where the model's "
                "decision may have been influenced. High attribution magnitude means that the corresponding pixels "
                "were influential in the explanation; it does not mean that the pixels are abnormal, malignant or "
                "biologically diagnostic.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            story.append(Paragraph(
                "<b>Why attribution magnitude matters.</b> A classifier makes its decision from many visual signals "
                "at once. SHAP attempts to distribute the model output across input features so the reader can see "
                "which parts of the image were most influential under the explanation procedure. This is useful for "
                "auditing whether the model appears to be responding to the cell itself, staining patterns, texture, "
                "edges, background context or other visually salient information. It is not a segmentation algorithm "
                "and it should not be used to trace a biological structure pixel by pixel.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            story.append(Paragraph(
                "<b>How to inspect this cell.</b> First locate the segmented cell in the Original panel. Next compare "
                "the brightest attribution regions with the same locations in the SHAP Overlay. Then ask whether the "
                "attribution is concentrated within the cell or whether substantial signal appears along boundaries, "
                "background areas or neighboring material. Finally compare the SHAP pattern with Grad-CAM for the same "
                "cell. This sequence helps separate an interpretable cell-centered decision from a decision that may "
                "be influenced by image context or acquisition artifacts.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            story.append(Paragraph(
                f"<b>Connection to the reported prediction.</b> Cell {cell_id} was assigned <b>{xml_escape(cls)}</b> "
                f"with <b>{pct(conf)}</b> model confidence. The segmented area is {area:,} pixels and the purple "
                f"score is {purple:.1f}. SHAP does not independently reclassify the cell; instead, it explains the "
                "selected model output. Therefore, the attribution map should be read as supporting evidence about "
                "model behavior, not as a second diagnostic test.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            story.append(Paragraph(
                "<b>Important limitation.</b> Pixel attribution can be affected by the image background, preprocessing, "
                "model architecture and the particular baseline/background used by the SHAP explainer. A visually "
                "striking attribution does not necessarily represent the most clinically meaningful feature. Likewise, "
                "a weak or diffuse attribution does not prove that the prediction is invalid. The correct use of SHAP "
                "in this report is model auditing: it helps the reader understand and question the computational "
                "decision rather than replacing expert microscopy.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 7))

            story.append(Paragraph(
                "<b>Interpretive boundary.</b> SHAP attribution is not evidence of AML, leukemia or another disease. "
                "The five-class classifier can assign a label even when an object is outside the intended biological "
                "class distribution. For that reason, the original smear, segmentation quality, class probabilities "
                "and clinical context remain essential when interpreting the result.",
                styles["BodyText"],
            ))
    else:
        story.append(Paragraph(
            "SHAP output was unavailable for this analysis. The absence of a SHAP visualization should not be "
            "interpreted as absence of model evidence; it only indicates that the SHAP explanation stage did not "
            "produce a usable attribution record.",
            styles["BodyText"],
        ))
    story.append(PageBreak())

    # ---------- 9. Cell-by-cell synthesis ----------
    section(
        "Cell-by-Cell Evidence Synthesis",
        "This section combines the measurable properties of each candidate with its classifier output and the "
        "availability of Grad-CAM/SHAP evidence. The purpose is to make the reasoning traceable from image object "
        "to model output to explanation, without converting model evidence into a clinical diagnosis.",
    )

    synthesis_cells = candidate_df.sort_values("cell_id").head(MAX_XAI_CELLS)
    if len(synthesis_cells):
        for _, row in synthesis_cells.iterrows():
            cell_id = int(row["cell_id"])
            cls = str(row["predicted_class"])
            conf = float(row["confidence"])
            area = int(row["area"])
            purple = float(row["purple_score"])
            has_gc = any(int(g["cell_id"]) == cell_id for g in gradcam_records)
            has_shap = any(int(s["cell_id"]) == cell_id for s in shap_records)

            story.append(Paragraph(
                f"Cell {cell_id} — {xml_escape(cls)} ({pct(conf)})",
                styles["CellHeading"],
            ))
            story.append(Paragraph(
                f"<b>Step 1 — Object selection:</b> Cell {cell_id} entered the candidate workflow with a segmented "
                f"area of <b>{area:,} pixels</b> and a purple score of <b>{purple:.1f}</b>. These measurements describe "
                "the image-processing stage and are used to decide which segmented objects proceed to classification. "
                "They do not independently identify the cell type.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 5))
            story.append(Paragraph(
                f"<b>Step 2 — Classification:</b> The ResNet50 model selected <b>{xml_escape(cls)}</b> as the top class "
                f"with <b>{pct(conf)}</b> confidence. This confidence should be understood as the model's preference "
                "among its five available labels. It is not a calibrated disease probability and should not be "
                "interpreted as a percentage likelihood of AML or leukemia.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 5))
            story.append(Paragraph(
                f"<b>Step 3 — Local explanation:</b> Grad-CAM is "
                f"{'available' if has_gc else 'not available'} for this cell and SHAP is "
                f"{'available' if has_shap else 'not available'}. When both are available, they provide complementary "
                "ways of inspecting the spatial evidence associated with the same prediction. Their role is to help "
                "identify whether the model appears to rely on the cell region or on potentially confounding visual "
                "context.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 5))
            story.append(Paragraph(
                "<b>Evidence quality:</b> The strength of this computational evidence depends on the quality of the "
                "segmentation, the representativeness of the crop, staining and imaging conditions, and whether the "
                "candidate is actually one of the biological categories represented in the training label set. "
                "A high-confidence prediction can still be wrong when the input is out-of-distribution or when the "
                "classifier is forced to choose among unsuitable classes.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 5))
            story.append(Paragraph(
                "<b>Practical reading:</b> Treat the cell result as a traceable research observation: an object was "
                "segmented, passed a heuristic selection rule, received a model label, and generated one or more "
                "visual explanations. The appropriate next action for a suspicious or unexpected result is review of "
                "the source image and expert laboratory/hematopathology assessment—not escalation of the model score "
                "into a standalone diagnosis.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 9))

    if candidate_count > MAX_XAI_CELLS:
        story.append(Paragraph(
            f"The analysis contained {candidate_count} candidates, while detailed local XAI was limited to "
            f"{MAX_XAI_CELLS} cells by configuration. The remaining candidates are represented in the aggregate "
            "tables but do not have individual Grad-CAM/SHAP pages in this report.",
            styles["SmallNote"],
        ))

    story.append(PageBreak())

    # ---------- 10. Interpretation ----------
    section(
        "Screening-Oriented Interpretation",
        "A concise interpretation of what this run suggests at the model level, followed by the constraints that "
        "must be considered before any biological or clinical conclusion.",
    )
    if candidate_count == 0:
        interpretation = (
            "No candidate cells passed the configured selection rule. This run therefore does not provide a "
            "meaningful candidate-cell class distribution. A lack of candidates can reflect image quality, "
            "segmentation behavior or the selection threshold and should not be treated as evidence that abnormal "
            "cells are absent."
        )
    elif myeloblast_count == candidate_count:
        interpretation = (
            f"All {candidate_count} candidate cells in this run received <b>myeloblast</b> as the top ResNet50 "
            f"class, producing a model-level myeloblast-like proportion of <b>{proportion:.2f}%</b>. This is a "
            "strongly concentrated model output within the available label set. It should prompt careful review "
            "of the original cells, segmentation quality, staining/domain compatibility and XAI maps rather than "
            "being converted into a disease probability."
        )
    else:
        interpretation = (
            f"The pipeline identified {myeloblast_count} myeloblast-like predictions among {candidate_count} "
            f"candidate cells ({proportion:.2f}%). This is a research-oriented model indicator. The meaning of "
            "that indicator depends on segmentation quality, the candidate-selection rule and how well the input "
            "resembles the data used to train the classifier."
        )
    story.append(Paragraph(interpretation, styles["BodyText"]))
    story.append(Spacer(1, 10))

    limitations = [
        "Cellpose segmentation errors propagate into crop selection and every downstream prediction.",
        "The purple-score threshold is a heuristic candidate-selection rule, not a validated hematology criterion.",
        "The classifier is closed-set: RBCs and unknown/non-target cells can be forced into one of five classes.",
        "Stain, microscope, illumination, focus, magnification and acquisition differences can produce domain shift.",
        "Small sample counts in a single field can make percentages unstable and should not be generalized to a whole blood sample.",
        "Classifier confidence is not a calibrated probability of disease.",
        "Grad-CAM and SHAP explain model behavior but do not prove that highlighted pixels are disease-specific structures.",
        "A myeloblast-like prediction is not equivalent to an AML or leukemia diagnosis.",
        "Clinical assessment requires appropriate hematopathology review and laboratory testing.",
    ]
    story.append(Paragraph("Key limitations", styles["Heading3"]))
    for item in limitations:
        story.append(Paragraph("- " + item, styles["BodyText"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 8))
    info_box(
        "Recommended use",
        "Use this report to inspect segmentation, compare cell-level model outputs, audit XAI evidence and identify "
        "cases that merit expert review. Do not use the report alone to diagnose or rule out disease.",
        "#eff6ff",
    )

    # ---------- 11. Technical appendix ----------
    story.append(PageBreak())
    section(
        "Technical Appendix",
        "Reproducibility-oriented details captured from the running SynthMicro pipeline.",
    )
    appendix_rows = [
        ["Parameter", "Configured value"],
        ["Cell extraction size", "384 × 384 pixels"],
        ["Candidate purple threshold", "35.0"],
        ["Maximum detailed XAI cells", "3"],
        ["SHAP background size", "2"],
        ["Classifier classes", ", ".join(CLASS_NAMES)],
        ["Segmentation model", "Cellpose CPSAM (GPU-enabled when available)"],
        ["Classifier", "ResNet50 Keras model"],
        ["Local XAI", "Grad-CAM + SHAP GradientExplainer"],
        ["Optional visual commentary", "Gemini Flash-Lite (asynchronously generated)"],
    ]
    at = Table(appendix_rows, colWidths=[2.45*inch, 4.0*inch], repeatRows=1)
    at.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(at)
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This appendix describes the software pipeline configuration used to generate the report. It does not "
        "constitute validation of the model for clinical use. Model performance should be established separately "
        "using an appropriate held-out dataset, calibration analysis and clinically meaningful evaluation metrics.",
        styles["SmallNote"],
    ))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ============================================================
# WHOLE-SMEAR OVERLAY
# ============================================================

def save_classification_overlay(smear_bgr, candidate_df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = cv2.cvtColor(smear_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(rgb)

    for _, row in candidate_df.iterrows():
        ax.text(
            int(row["x"]),
            int(row["y"]),
            f"{int(row['cell_id'])}: {row['predicted_class']}\n"
            f"{row['confidence'] * 100:.1f}%",
            fontsize=7,
            color="black",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    ax.axis("off")
    ax.set_title("Whole Smear — Candidate Cell Classification")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ANALYSIS PIPELINE
# ============================================================

def run_analysis(original_image: Image.Image, analysis_id: str):
    analysis_dir = XAI_DIR / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    original_path = UPLOAD_DIR / f"{analysis_id}.png"
    original_image.save(original_path)

    smear_bgr = cv2.cvtColor(
        np.array(original_image.convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )

    # 1. Cellpose
    masks = segment_smear(smear_bgr)
    cell_crops, cell_df = make_cell_crops(smear_bgr, masks)

    if len(cell_crops) == 0:
        raise RuntimeError("Cellpose did not produce any usable cell objects.")

    # 2. ResNet50
    X, X_preprocessed, predictions = classify_cells(cell_crops)
    predicted_indices = np.argmax(predictions, axis=1)
    cell_df["predicted_class"] = [CLASS_NAMES[i] for i in predicted_indices]
    cell_df["confidence"] = np.max(predictions, axis=1)
    cell_df["purple_score"] = [calculate_purple_score(c) for c in cell_crops]
    cell_df["prediction_vector"] = list(predictions)

    # 3. Candidate selection — same rule as Phase 7 notebook.
    candidate_df = cell_df[cell_df["purple_score"] >= PURPLE_THRESHOLD].copy()
    candidate_df = candidate_df.sort_values("purple_score", ascending=False)

    myeloblast_df = candidate_df[
        candidate_df["predicted_class"] == "myeloblast"
    ].copy()
    myeloblast_df = myeloblast_df.sort_values("confidence", ascending=False)

    # 4. Save overlay.
    overlay_path = analysis_dir / "classification_overlay.png"
    save_classification_overlay(smear_bgr, candidate_df, overlay_path)

    # 5. XAI cells = up to six highest-confidence myeloblast-like candidates.
    xai_cells = myeloblast_df.head(MAX_XAI_CELLS).copy()
    gradcam_records = []

    for _, row in xai_cells.iterrows():
        cell_id = int(row["cell_id"])
        idx = int(cell_df.index[cell_df["cell_id"] == cell_id][0])
        class_index = int(np.argmax(predictions[idx]))
        image_input = np.expand_dims(X_preprocessed[idx], axis=0)
        heatmap = generate_gradcam(image_input, class_index)

        gradcam_path = analysis_dir / f"cell_{cell_id}_gradcam.png"
        save_gradcam(cell_crops[idx], heatmap, gradcam_path)

        original_path_for_vlm = analysis_dir / f"cell_{cell_id}_original.png"
        Image.fromarray(cv2.cvtColor(cell_crops[idx], cv2.COLOR_BGR2RGB)).save(
            original_path_for_vlm
        )

        gradcam_records.append(
            {
                "cell_id": cell_id,
                "class": CLASS_NAMES[class_index],
                "confidence": float(predictions[idx, class_index]),
                "path": str(gradcam_path),
                "original_path": str(original_path_for_vlm),
            }
        )
	
    # 6. SHAP on the same selected cells.
    print("========== SHAP START ==========")
    shap_records, shap_error = generate_shap_records(
        X_preprocessed,
        X,
        cell_df,
        xai_cells,
        analysis_dir,
    )
    print("========== SHAP FINISHED ==========")

    # 7. VLM explanations for the selected cell crops.
    explanations = build_vlm_explanations(gradcam_records, cell_df)

    # 8. PDF.
    report_path = XAI_DIR / f"{analysis_id}_Phase7_XAI_Report.pdf"
    make_pdf_report(
        analysis_id=analysis_id,
        smear_path=original_path,
        overlay_path=overlay_path,
        cell_df=cell_df,
        candidate_df=candidate_df,
        myeloblast_df=myeloblast_df,
        gradcam_records=gradcam_records,
        shap_records=shap_records,
        explanations=explanations,
        output_path=report_path,
    )

    # 9. CSV with candidate results.
    candidate_csv = analysis_dir / "cell_results.csv"
    export_df = candidate_df[
        ["cell_id", "area", "x", "y", "width", "height", "purple_score", "predicted_class", "confidence"]
    ].copy()
    export_df["confidence_percent"] = export_df["confidence"] * 100
    export_df.to_csv(candidate_csv, index=False)

    # 10. Screening indicator. This is intentionally not named a diagnosis.
    candidate_count = len(candidate_df)
    myeloblast_count = len(myeloblast_df)
    myeloblast_proportion = (
        myeloblast_count / candidate_count * 100
        if candidate_count > 0
        else 0.0
    )

    if myeloblast_count == 0:
        screening_label = "No myeloblast-like candidates detected"
        screening_level = "LOWER_SCREENING_CONCERN"
    elif myeloblast_proportion >= 20:
        screening_label = "Elevated myeloblast-like cell burden"
        screening_level = "ELEVATED_SCREENING_INDICATOR"
    else:
        screening_label = "Myeloblast-like candidates detected"
        screening_level = "SCREENING_FLAG"

    # Store paths in DB.
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO analyses
        (id, prediction, confidence, image_path, heatmap_path, xai_path, report_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_id,
            "myeloblast-like" if myeloblast_count else "no myeloblast-like candidate",
            float(myeloblast_proportion / 100.0),
            str(original_path),
            str(analysis_dir),
            str(overlay_path),
            str(report_path),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()

    # Remove prediction vectors from JSON response.
    response_candidates = []
    for _, row in candidate_df.iterrows():
        response_candidates.append({
            "cell_id": int(row["cell_id"]),
            "area": int(row["area"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "purple_score": float(row["purple_score"]),
            "predicted_class": str(row["predicted_class"]),
            "confidence": float(row["confidence"]),
        })

    # Whole-smear class distribution over candidate cells.
    distribution = (
        candidate_df["predicted_class"].value_counts().to_dict()
        if candidate_count
        else {}
    )

    # Top cell = highest-confidence candidate, not necessarily myeloblast.
    top_candidate = None
    if candidate_count:
        top = candidate_df.sort_values("confidence", ascending=False).iloc[0]
        top_candidate = {
            "cell_id": int(top["cell_id"]),
            "predicted_class": str(top["predicted_class"]),
            "confidence": float(top["confidence"]),
        }

    return {
        "analysis_id": analysis_id,
        "prediction": top_candidate["predicted_class"] if top_candidate else None,
        "confidence": top_candidate["confidence"] if top_candidate else 0.0,
        "probabilities": (
            {
                CLASS_NAMES[i]: float(predictions[int(cell_df.index[cell_df["cell_id"] == top_candidate["cell_id"]][0]), i])
                for i in range(5)
            }
            if top_candidate is not None else {}
        ),
        "cellpose_objects": len(cell_df),
        "candidate_cells": candidate_count,
        "myeloblast_like_cells": myeloblast_count,
        "myeloblast_like_proportion": myeloblast_proportion,
        "screening_label": screening_label,
        "screening_level": screening_level,
        "class_distribution": distribution,
        "candidates": response_candidates,
        "gradcam_records": [
            {
                "cell_id": r["cell_id"],
                "class": r["class"],
                "confidence": r["confidence"],
                "image_url": f"/analysis/{analysis_id}/files/{Path(r['path']).name}",
            }
            for r in gradcam_records
        ],
        "shap_records": [
            {
                "cell_id": r["cell_id"],
                "class": r["class"],
                "confidence": r["confidence"],
                "image_url": f"/analysis/{analysis_id}/files/{Path(r['path']).name}",
            }
            for r in shap_records
        ],
        "shap_available": bool(shap_records),
        "shap_error": shap_error,
        "gemini_model": GEMINI_MODEL,
        "gemini_enabled": bool(GEMINI_API_KEY and genai is not None),
        "gemini_status_url": f"/analysis/{analysis_id}/gemini",
        "report_url": f"/reports/{analysis_id}",
        "overlay_url": f"/analysis/{analysis_id}/files/{overlay_path.name}",
        "original_url": f"/analysis/{analysis_id}/files/{original_path.name}",
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "pipeline": "Cellpose -> ResNet50 -> Grad-CAM -> SHAP -> Gemini Flash-Lite -> PDF",
        "model": ACTIVE_MODEL_PATH.name,
        "input_size": IMG_SIZE,
        "classes": CLASS_NAMES,
        "gemini_model": GEMINI_MODEL,
        "gemini_enabled": bool(GEMINI_API_KEY and genai is not None),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": ACTIVE_MODEL_PATH.name,
        "cellpose_available": True,
        "shap_available": shap is not None,
        "gemini_available": genai is not None,
        "gemini_key_configured": bool(GEMINI_API_KEY),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if file.content_type not in allowed_types:
        return {
            "success": False,
            "error": "Please upload a JPG, JPEG, or PNG blood-smear image.",
        }

    try:
        contents = await file.read()
        if not contents:
            return {"success": False, "error": "Uploaded file is empty."}

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        analysis_id = uuid.uuid4().hex

        result = run_analysis(image, analysis_id)
        # Gemini is deliberately decoupled from the critical ML/XAI path.
        # The UI receives Cellpose/ResNet/Grad-CAM/SHAP immediately and polls
        # this background job for the optional visual commentary.
        if background_tasks is not None and result.get("gemini_enabled"):
            # Reconstruct the lightweight inputs needed by the Gemini worker.
            analysis_dir = XAI_DIR / analysis_id
            records = []
            for item in result.get("gradcam_records", []):
                cell_id = int(item["cell_id"])
                original_cell = analysis_dir / f"cell_{cell_id}_original.png"
                records.append({
                    "cell_id": cell_id,
                    "class": item["class"],
                    "confidence": item["confidence"],
                    "original_path": str(original_cell),
                })
            # Persist enough metadata for the background worker without keeping
            # the pandas DataFrame alive after the request.
            meta = pd.DataFrame([
                {
                    "cell_id": r["cell_id"],
                    "predicted_class": r["class"],
                    "confidence": r["confidence"],
                } for r in records
            ])
            background_tasks.add_task(run_gemini_background, analysis_id, records, meta)
        else:
            write_gemini_status(analysis_id, {
                "analysis_id": analysis_id,
                "status": "disabled",
                "model": GEMINI_MODEL,
                "results": {},
            })
        result["success"] = True
        return result

    except Exception as exc:
        print("Analysis error:", repr(exc))
        return {
            "success": False,
            "error": str(exc),
        }


@app.get("/analysis/{analysis_id}/gemini")
def get_gemini_status(analysis_id: str):
    path = _gemini_status_path(analysis_id)
    if not path.exists():
        return {
            "analysis_id": analysis_id,
            "status": "queued",
            "model": GEMINI_MODEL,
            "results": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/reports/{analysis_id}")
def get_report(analysis_id: str):
    path = XAI_DIR / f"{analysis_id}_Phase7_XAI_Report.pdf"
    if not path.exists():
        return {"success": False, "error": "Report not found."}
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"SynthMicro_{analysis_id}_XAI_Report.pdf",
    )


@app.get("/analysis/{analysis_id}/files/{filename}")
def get_analysis_file(analysis_id: str, filename: str):
    path = XAI_DIR / analysis_id / filename
    if not path.exists() or not path.is_file():
        return {"success": False, "error": "Analysis file not found."}
    return FileResponse(path)


@app.get("/history")
def history():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, prediction, confidence, image_path, heatmap_path,
               xai_path, report_path, created_at
        FROM analyses
        ORDER BY created_at DESC
        LIMIT 20
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    for row in rows:
        row["report_url"] = f"/reports/{row['id']}"

    return rows


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
