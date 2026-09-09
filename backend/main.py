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
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, File, UploadFile
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
    from huggingface_hub import InferenceClient
except Exception as exc:
    InferenceClient = None
    HF_IMPORT_ERROR = str(exc)

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

# Hugging Face model used by the explanatory report layer.
HF_VLM_MODEL = os.getenv(
    "HF_VLM_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

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

def hf_explain_image(image_path: Path, prompt: str) -> str:
    """Use HF Qwen2.5-VL when HF_TOKEN is configured."""
    if InferenceClient is None:
        raise RuntimeError(f"huggingface_hub is unavailable: {HF_IMPORT_ERROR}")
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not configured. Set HF_TOKEN in the backend environment."
        )

    client = InferenceClient(
        model=HF_VLM_MODEL,
	provider="featherless-ai",
        token=HF_TOKEN,
        timeout=120,
    )

    data_url = file_to_data_url(image_path)

    response = client.chat.completions.create(
        model=HF_VLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an explanatory assistant for a research blood-smear "
                    "image-analysis system. Do not diagnose cancer or leukemia. "
                    "Do not invent morphology that is not visible. Explain model "
                    "results conservatively and explicitly distinguish model output "
                    "from clinical diagnosis."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
        max_tokens=350,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


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


def build_vlm_explanations(selected_records, cell_df):
    explanations = {}
    for rec in selected_records:
        cell_id = rec["cell_id"]
        row = cell_df.loc[cell_df["cell_id"] == cell_id].iloc[0].to_dict()
        prompt = (
            f"This segmented microscopy crop was classified by ResNet50 as "
            f"'{row['predicted_class']}' with {row['confidence'] * 100:.1f}% confidence. "
            "Explain in 3-5 sentences what is visibly present in the image and how "
            "the result can be interpreted as model behavior. Mention uncertainty "
            "and domain limitations. Do not diagnose leukemia/cancer and do not "
            "claim a cell type that cannot be supported from the image."
        )
        try:
            explanations[cell_id] = hf_explain_image(
                Path(rec["original_path"]),
                prompt,
            )
        except Exception as exc:
            explanations[cell_id] = fallback_cell_explanation(row)
            print(f"HF explanation unavailable for Cell {cell_id}: {exc}")
    return explanations


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
    total_segmented = len(cell_df)
    candidate_count = len(candidate_df)
    myeloblast_count = len(myeloblast_df)
    proportion = (
        myeloblast_count / candidate_count * 100
        if candidate_count > 0
        else 0.0
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#555555"),
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    story = []
    story.append(Paragraph("SynthMicro — Whole Blood Smear XAI Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Analysis ID: {analysis_id}<br/>"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["SmallNote"],
    ))
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Pipeline: Cellpose segmentation → 384×384 cell extraction → "
        "ResNet50 classification → Grad-CAM → SHAP → explanatory report.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 14))

    summary = [
        ["Measurement", "Result"],
        ["Cellpose objects", str(total_segmented)],
        ["Nucleated-cell candidates", str(candidate_count)],
        ["Myeloblast-like candidates", str(myeloblast_count)],
        ["Myeloblast-like proportion", f"{proportion:.2f}%"],
    ]
    table = Table(summary, colWidths=[3.3 * inch, 2.2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        "<b>Screening interpretation:</b> the proportion above is a "
        "research-oriented model indicator based on candidate cells selected "
        "by the notebook's purple-score rule. It is not a cancer probability "
        "and does not establish AML or another diagnosis.",
        styles["BodyText"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Whole Smear", styles["Heading1"]))
    story.append(RLImage(str(smear_path), width=6.7 * inch, height=5.1 * inch))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Original uploaded blood-smear image analyzed by Cellpose.",
        styles["SmallNote"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Segmentation and Candidate Overlay", styles["Heading1"]))
    story.append(RLImage(str(overlay_path), width=6.7 * inch, height=5.1 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Cell-Level Results", styles["Heading1"]))
    rows = [["ID", "Area", "Purple", "Class", "Confidence"]]
    for _, row in candidate_df.sort_values("cell_id").iterrows():
        rows.append([
            str(int(row["cell_id"])),
            str(int(row["area"])),
            f"{row['purple_score']:.1f}",
            str(row["predicted_class"]),
            f"{row['confidence'] * 100:.1f}%",
        ])
    results_table = Table(
        rows,
        repeatRows=1,
        colWidths=[0.55 * inch, 0.7 * inch, 0.8 * inch, 1.8 * inch, 1.0 * inch],
    )
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(results_table)
    story.append(PageBreak())

    story.append(Paragraph("Grad-CAM Explanations", styles["Heading1"]))
    story.append(Paragraph(
        "Grad-CAM shows regions contributing to the selected model class. "
        "It is an explanation of model behavior, not a direct measurement "
        "of a biological structure.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 10))

    for rec in gradcam_records:
        story.append(Paragraph(
            f"Cell {rec['cell_id']} — {rec['class']} ({rec['confidence'] * 100:.1f}%)",
            styles["Heading3"],
        ))
        story.append(RLImage(rec["path"], width=6.7 * inch, height=2.2 * inch))
        story.append(Spacer(1, 7))
        if rec["cell_id"] in explanations:
            story.append(Paragraph(explanations[rec["cell_id"]], styles["BodyText"]))
        story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("SHAP Explanations", styles["Heading1"]))
    story.append(Paragraph(
        "SHAP attribution magnitude summarizes the magnitude of pixel-level "
        "contribution toward the selected class. It is a model explanation, "
        "not a clinical measurement.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 10))

    if shap_records:
        for rec in shap_records:
            story.append(Paragraph(
                f"Cell {rec['cell_id']} — {rec['class']} ({rec['confidence'] * 100:.1f}%)",
                styles["Heading3"],
            ))
            story.append(RLImage(rec["path"], width=6.7 * inch, height=2.2 * inch))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph(
            "SHAP output was unavailable for this analysis.",
            styles["BodyText"],
        ))

    story.append(PageBreak())
    story.append(Paragraph("Interpretation and Limitations", styles["Heading1"]))
    limitations = [
        "The classifier has five closed-set classes: basophil, erythroblast, monocyte, myeloblast and segmented neutrophil.",
        "RBCs and unknown cell types are not explicit classes and can therefore be forced into one of the five outputs.",
        "Cellpose segmentation quality affects every downstream result.",
        "Stain, microscope, illumination and acquisition differences can cause domain shift.",
        "Confidence is not a calibrated probability of disease.",
        "A myeloblast-like prediction is not equivalent to an AML diagnosis.",
        "Clinical assessment requires appropriate hematopathology and laboratory testing.",
    ]
    for item in limitations:
        story.append(Paragraph("• " + item, styles["BodyText"]))
        story.append(Spacer(1, 5))

    doc.build(story)


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
        "vlm_model": HF_VLM_MODEL,
        "vlm_enabled": bool(HF_TOKEN and InferenceClient is not None),
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
        "pipeline": "Cellpose -> ResNet50 -> Grad-CAM -> SHAP -> Hugging Face VLM -> PDF",
        "model": ACTIVE_MODEL_PATH.name,
        "input_size": IMG_SIZE,
        "classes": CLASS_NAMES,
        "hf_vlm_model": HF_VLM_MODEL,
        "hf_vlm_enabled": bool(HF_TOKEN and InferenceClient is not None),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": ACTIVE_MODEL_PATH.name,
        "cellpose_available": True,
        "shap_available": shap is not None,
        "huggingface_available": InferenceClient is not None,
        "hf_token_configured": bool(HF_TOKEN),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
        result["success"] = True
        return result

    except Exception as exc:
        print("Analysis error:", repr(exc))
        return {
            "success": False,
            "error": str(exc),
        }


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
