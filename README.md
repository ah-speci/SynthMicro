# SynthMicro

### Explainable AI for Microscopic Blood-Cell Analysis

SynthMicro is an AI-assisted blood-smear analysis system designed to classify microscopic blood-cell images and provide visual explanations for the model's predictions.

The project combines **Cellpose segmentation**, **ResNet50 transfer learning**, **Grad-CAM**, **SHAP**, and an optional **Hugging Face vision-language model (VLM)** into a web-based analysis pipeline.

> **Important:** SynthMicro is a research and educational system. Its predictions, confidence values, screening indicators, Grad-CAM maps, and generated reports are **not clinical diagnoses** and must not be used as a substitute for evaluation by a qualified healthcare professional.

---

## Overview

A major goal of SynthMicro is to move beyond simply classifying a single cropped blood cell.

For a microscopy image containing many cells, the application can:

1. Accept a whole blood-smear microscopy image.
2. Segment individual cells using **Cellpose**.
3. Extract and preprocess cell candidates.
4. Classify candidate cells using a trained **ResNet50** model.
5. Identify myeloblast-like candidates from the model output.
6. Aggregate cell-level results into a research-oriented screening indicator.
7. Generate **Grad-CAM** visualizations showing regions influencing predictions.
8. Generate **SHAP** attribution visualizations.
9. Optionally use a Hugging Face vision-language model to produce explanatory text.
10. Produce a detailed **PDF XAI report**.

The repository is structured as a full-stack application with a Python/FastAPI backend, React frontend, trained model artifacts, datasets, notebooks, and generated XAI results.

---

## System Architecture

```text
                    ┌──────────────────────────┐
                    │   Microscopy Image       │
                    │   JPG / JPEG / PNG       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       FastAPI API        │
                    │       /predict            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       Cellpose            │
                    │  Individual Cell Masks    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Cell Crop + Preprocessing│
                    │       384 × 384          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │        ResNet50          │
                    │  5-Class Cell Classifier  │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 │               │                │
                 ▼               ▼                ▼
          ┌────────────┐  ┌────────────┐  ┌──────────────┐
          │ Grad-CAM   │  │    SHAP    │  │ Probability  │
          │ Explain.   │  │ Attribution│  │ / Cell Stats │
          └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
                │                │                 │
                └────────────────┼─────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │   XAI / Screening Report │
                    │        PDF + JSON        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       React Frontend      │
                    │ Upload • Results • XAI    │
                    └──────────────────────────┘
```

---

## Supported Cell Classes

The current classifier uses five output classes:

| Class | Description |
|---|---|
| `basophil` | Mature granulocyte associated with inflammatory and allergic responses |
| `erythroblast` | Immature red-blood-cell precursor |
| `monocyte` | Mature agranulocyte involved in immune defense and phagocytosis |
| `myeloblast` | Immature myeloid precursor cell |
| `seg_neutrophil` | Mature segmented neutrophil |

The model is therefore a **closed-set five-class classifier**. A cell type that is not represented by one of these classes can still be assigned to one of the available classes by the softmax classifier. This is an important limitation when analyzing real-world whole-smear images.

---

## Cancer / Leukemia Screening Concept

SynthMicro should not be interpreted as a direct patient-level cancer diagnosis system.

The current pipeline uses the presence and proportion of **myeloblast-like model predictions** as a research-oriented screening indicator.

For a processed smear, the backend reports:

- total Cellpose objects
- candidate-cell count
- number of myeloblast-like candidates
- myeloblast-like proportion
- class distribution
- individual cell predictions
- confidence values
- Grad-CAM results
- SHAP results
- an XAI PDF report

The backend currently uses a **20% myeloblast-like proportion** as an elevated screening indicator. This should not be interpreted as a standalone diagnostic threshold because the value is calculated from the application's candidate-selection and classifier pipeline rather than from a validated clinical workflow.

A proper leukemia diagnosis requires appropriate clinical, laboratory, morphological, immunophenotypic, cytogenetic and/or molecular evaluation as determined by qualified specialists.

---

# Features

## 1. Whole-Smear Cell Segmentation

The backend launches Cellpose in a separate Python process.

This design isolates the Cellpose/PyTorch environment from the TensorFlow/ResNet process and reduces the possibility of native-library conflicts.

The worker is implemented in:

```text
backend/cellpose_worker.py
```

The worker:

- loads the uploaded smear
- runs Cellpose `cpsam`
- generates cell masks
- saves the mask array
- returns the result to the FastAPI process

---

## 2. ResNet50 Cell Classification

The project uses a trained ResNet50 model with an input size of:

```text
384 × 384 × 3
```

The model predicts five classes:

```text
basophil
erythroblast
monocyte
myeloblast
seg_neutrophil
```

The backend loads the preferred model from:

```text
models/resnet50_frozen_384.keras
```

and falls back to:

```text
models/resnet50_best.keras
```

if the preferred model is unavailable.

---

## 3. Grad-CAM Explainability

Grad-CAM is used to visualize the regions that contributed to the selected class prediction.

The current implementation targets the ResNet50 layer:

```text
conv5_block3_out
```

The generated visualization contains:

- original cell
- Grad-CAM heatmap
- Grad-CAM overlay

These visualizations are intended to help inspect whether the model is focusing on meaningful cell regions rather than background artifacts.

---

## 4. SHAP Explainability

SHAP is used as a second explainability method.

The system generates:

- original cell
- SHAP attribution magnitude
- SHAP overlay

This provides a complementary view of which image regions contribute to the selected class.

SHAP is optional. If the SHAP dependency is unavailable or an explanation cannot be generated, the main classification pipeline can still return results.

---

## 5. Optional Vision-Language Explanations

If a Hugging Face token is configured, SynthMicro can use:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

to generate explanatory text for selected cell crops.

The VLM is instructed to:

- describe visible image content conservatively
- explain model behavior
- distinguish model output from clinical diagnosis
- avoid inventing morphology
- avoid diagnosing leukemia or cancer

If the VLM is unavailable, the backend falls back to a deterministic cell-level explanation.

---

## 6. Detailed PDF XAI Reports

The backend generates a report containing:

- analysis ID and timestamp
- pipeline description
- whole-smear image
- segmentation/classification overlay
- cell-level classification table
- Grad-CAM explanations
- SHAP explanations
- screening-oriented summary
- limitations and clinical-use disclaimer

Reports are generated with ReportLab.

---

## 7. React Web Interface

The frontend provides:

- microscopy-image upload
- drag-and-drop support
- image preview
- model analysis button
- prediction and confidence display
- class probability bars
- original/heatmap/overlay views
- Grad-CAM visualization
- heatmap opacity control
- XAI comparison cards
- cell profile information
- research-use disclaimer

The frontend communicates with the FastAPI backend through:

```text
POST /predict
```

The main frontend application is:

```text
frontend/App.jsx
```

---

# Repository Structure

The repository currently contains the following major components:

```text
SynthMicro/
│
├── backend/
│   ├── main.py
│   └── cellpose_worker.py
│
├── dataset/
│
├── frontend/
│   └── App.jsx
│
├── images/
│
├── models/
│   ├── resnet50_frozen_384.keras
│   └── resnet50_best.keras
│
├── notebooks/
│   ├── phase1.ipynb
│   ├── phase2_eda.ipynb
│   ├── phase3.ipynb
│   ├── phase4_resnet50.ipynb
│   ├── phase5.ipynb
│   ├── phase6_gradcam.ipynb
│   ├── xai.ipynb
│   ├── xai_report.ipynb
│   └── test.ipynb
│
├── uploads/
│
├── xai_results/
│
├── test_gradcam.py
├── synthmicro.db
└── README.md
```

Some generated folders/files may be empty in a fresh checkout and will be populated when the application is executed.

---

# Installation

## Prerequisites

Recommended environment:

- Python 3.10+
- Node.js 18+
- npm
- Git
- Sufficient RAM for TensorFlow and Cellpose
- Optional GPU support for model development/training

Clone the repository:

```bash
git clone https://github.com/ah-speci/SynthMicro.git
cd SynthMicro
```

---

# Backend Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install the backend dependencies:

```bash
pip install \
  fastapi \
  uvicorn \
  python-multipart \
  tensorflow \
  opencv-python \
  numpy \
  pandas \
  pillow \
  reportlab \
  shap \
  cellpose \
  huggingface-hub
```

Depending on the TensorFlow, PyTorch, CUDA, Cellpose and operating-system combination, some dependencies may require platform-specific installation.

---

# Model Files

Place the trained model files under:

```text
models/
```

The backend looks for:

```text
models/resnet50_frozen_384.keras
```

first.

If it does not exist, it tries:

```text
models/resnet50_best.keras
```

The model must be compatible with the five-class output mapping used by the backend:

```python
CLASS_NAMES = [
    "basophil",
    "erythroblast",
    "monocyte",
    "myeloblast",
    "seg_neutrophil",
]
```

**Important:** the class ordering must match the ordering used during model training.

---

# Optional Hugging Face VLM Setup

The VLM layer is optional.

Set a Hugging Face token in your environment.

### Linux / macOS

```bash
export HF_TOKEN="your_huggingface_token"
```

### Windows PowerShell

```powershell
$env:HF_TOKEN="your_huggingface_token"
```

The default model configured by the backend is:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

You can override it with:

```bash
export HF_VLM_MODEL="your/model-name"
```

Without `HF_TOKEN`, the application still performs segmentation, classification and XAI processing, but VLM-generated explanations are disabled and the backend uses fallback explanations.

---

# Running the Backend

From the repository root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Useful endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API and model information |
| `/health` | GET | Backend health/dependency status |
| `/predict` | POST | Analyze a microscopy image |
| `/history` | GET | Retrieve recent analyses |
| `/reports/{analysis_id}` | GET | Download/view an XAI PDF |
| `/analysis/{analysis_id}/files/{filename}` | GET | Retrieve generated analysis assets |

FastAPI's interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# Frontend Setup

Move into the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The exact command depends on the frontend project's package configuration.

The React application expects the backend at:

```text
http://127.0.0.1:8000
```

This is configured in `frontend/App.jsx`.

If the backend is hosted elsewhere, update:

```javascript
const API_URL = "http://127.0.0.1:8000";
```

---

# Typical Usage

1. Start the FastAPI backend.
2. Start the React frontend.
3. Open the frontend in a browser.
4. Upload a whole blood-smear microscopy image.
5. Select **Analyze specimen**.
6. The backend:
   - segments cells with Cellpose
   - extracts candidate cells
   - classifies them with ResNet50
   - calculates model probabilities
   - selects myeloblast-like candidates
   - generates Grad-CAM
   - attempts SHAP explanations
   - optionally generates VLM explanations
   - creates a PDF report
7. The frontend displays the classification and XAI results.
8. The generated report can be accessed using the returned report URL.

---

# API Example

Example request:

```bash
curl -X POST \
  -F "file=@sample_smear.jpg" \
  http://127.0.0.1:8000/predict
```

A successful response contains information similar to:

```json
{
  "success": true,
  "analysis_id": "example-analysis-id",
  "prediction": "myeloblast",
  "confidence": 0.94,
  "cellpose_objects": 120,
  "candidate_cells": 76,
  "myeloblast_like_cells": 18,
  "myeloblast_like_proportion": 23.68,
  "screening_label": "Elevated myeloblast-like cell burden",
  "screening_level": "ELEVATED_SCREENING_INDICATOR",
  "class_distribution": {
    "myeloblast": 18,
    "monocyte": 20,
    "basophil": 8,
    "erythroblast": 12,
    "seg_neutrophil": 18
  },
  "report_url": "/reports/example-analysis-id"
}
```

The exact values depend on the uploaded specimen.

---

# Machine Learning Workflow

The project development is organized across several notebooks covering stages such as:

```text
Data preparation
      ↓
EDA / dataset analysis
      ↓
CNN / transfer-learning experiments
      ↓
ResNet50 training
      ↓
Model evaluation
      ↓
Grad-CAM
      ↓
SHAP / XAI
      ↓
Report generation
      ↓
FastAPI + React application
```

The notebooks are retained as the research/development record, while the backend contains the inference pipeline used by the application.

---

# Why ResNet50?

ResNet50 is used as the primary image classifier because residual connections allow deeper convolutional representations to be trained effectively.

For microscopic cell classification, the model can learn image features associated with:

- cell shape
- nucleus appearance
- cytoplasmic characteristics
- stain patterns
- granularity
- texture
- spatial morphology

The project uses a 384×384 input pipeline to preserve more fine-grained visual information than a smaller 224×224 input.

---

# Why Cellpose?

A whole-smear image can contain many cells.

Classifying the entire smear as a single image would make it difficult to associate predictions with individual cells.

Cellpose therefore acts as the segmentation stage:

```text
Whole smear
    ↓
Cell masks
    ↓
Individual cell crops
    ↓
ResNet50
```

This makes cell-level predictions and subsequent XAI analysis possible.

---

# Why Explainable AI?

A high classification score alone does not show what the model used to make its decision.

SynthMicro therefore includes multiple explainability techniques.

### Grad-CAM

Shows spatial regions associated with the selected CNN prediction.

### SHAP

Provides attribution information describing how image features contribute to the model output.

### VLM Explanation

When enabled, the VLM converts the visual evidence and model context into a human-readable explanation while being instructed not to present the model result as a clinical diagnosis.

Using multiple XAI methods makes it easier to investigate model behavior, failure cases and possible dataset biases.

---

# Important Limitations

## Closed-set classification

The ResNet50 model has only five classes. Real blood smears contain many other cell types and artifacts.

An unknown cell may therefore receive an incorrect five-class prediction.

## Segmentation dependency

Incorrect Cellpose masks can produce poor crops, which can then produce incorrect classifications and XAI results.

## Dataset bias

Microscopy models can learn stain, illumination, scanner, crop and background characteristics instead of clinically meaningful morphology.

## Confidence is not clinical probability

A model confidence such as `95%` means the model strongly prefers one of its available classes. It does **not** mean there is a 95% probability that the patient has cancer.

## Myeloblast ≠ leukemia diagnosis

A myeloblast is a cell type. Detecting a myeloblast-like cell does not, by itself, establish AML or another leukemia.

## Screening indicator ≠ diagnosis

The whole-smear myeloblast-like proportion generated by SynthMicro is an application-level research indicator. It should not be presented as a validated diagnostic rule.

---

# Research and Educational Use

SynthMicro is intended to support:

- machine-learning research
- blood-cell image classification experiments
- XAI research
- Grad-CAM visualization
- SHAP visualization
- segmentation/classification pipeline development
- educational demonstrations of medical-image AI
- investigation of model failure cases

It is not intended to independently diagnose, rule out, or stage cancer or leukemia.

---

# Troubleshooting

## Backend cannot find the model

Check:

```text
models/resnet50_frozen_384.keras
```

or:

```text
models/resnet50_best.keras
```

## Cellpose fails

Make sure Cellpose is installed in the same Python environment used to launch the backend.

The backend deliberately launches:

```text
backend/cellpose_worker.py
```

as a separate process.

Run the worker manually to test the environment:

```bash
python backend/cellpose_worker.py input.png output_masks.npy
```

## SHAP is unavailable

Check the SHAP installation:

```bash
pip install shap
```

SHAP failures do not necessarily prevent the main prediction pipeline from running.

## VLM explanations are unavailable

Check:

```bash
echo $HF_TOKEN
```

and ensure `huggingface-hub` is installed.

The application can run without the VLM.

## Frontend cannot connect to the API

Confirm the backend is running:

```text
http://127.0.0.1:8000/health
```

Then check that `frontend/App.jsx` uses the correct API URL.

---

# Project Status

SynthMicro is an active research/application project combining:

- computer vision
- deep learning
- medical-image segmentation
- transfer learning
- explainable AI
- FastAPI
- React
- automated report generation

The repository currently contains the backend, frontend, model, dataset, notebooks and XAI-result components needed to develop and run the system.

---

# Disclaimer

SynthMicro is an **AI-assisted research and educational prototype**.

Nothing in this repository should be interpreted as medical advice or as a validated diagnostic test. The system does not replace a hematologist, pathologist, laboratory investigation, or other qualified healthcare professional.

Any real clinical deployment would require appropriate validation, representative clinical data, calibration, external testing, regulatory review, security/privacy controls, and prospective clinical evaluation.

---

# License

No explicit license is currently specified in the repository. Until a license is added, users should not assume that the code is freely reusable or redistributable.

---

## Repository

[https://github.com/ah-speci/SynthMicro](https://github.com/ah-speci/SynthMicro)
