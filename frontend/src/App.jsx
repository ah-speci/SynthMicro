import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const CLASS_INFO = {
  basophil: {
    title: "Basophil",
    category: "White Blood Cell",
    description:
      "A granulocyte associated with immune and inflammatory responses.",
    short:
      "Model-detected basophil morphology",
  },

  erythroblast: {
    title: "Erythroblast",
    category: "Red Blood Cell Precursor",
    description:
      "An immature red blood cell precursor observed during erythropoiesis.",
    short:
      "Model-detected erythroblast morphology",
  },

  monocyte: {
    title: "Monocyte",
    category: "White Blood Cell",
    description:
      "A large white blood cell involved in immune defense and tissue response.",
    short:
      "Model-detected monocyte morphology",
  },

  myeloblast: {
    title: "Myeloblast",
    category: "Immature Myeloid Cell",
    description:
      "An immature myeloid precursor cell identified from microscopic morphology.",
    short:
      "Model-detected myeloblast morphology",
  },

  seg_neutrophil: {
    title: "Segmented Neutrophil",
    category: "White Blood Cell",
    description:
      "A mature neutrophil involved in the body's innate immune response.",
    short:
      "Model-detected segmented neutrophil morphology",
  },
};

function formatClassName(name) {
  if (!name) return "";

  return name
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [viewMode, setViewMode] = useState("overlay");
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.45);
  const [dragActive, setDragActive] = useState(false);

  // =========================================================
  // CLEANUP
  // =========================================================

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  // =========================================================
  // FILE HANDLING
  // =========================================================

  const processFile = (file) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid microscopy image.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("Image size must be less than 10 MB.");
      return;
    }

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
    setViewMode("overlay");
  };

  const handleFileChange = (event) => {
    processFile(event.target.files?.[0]);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragActive(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);

    const file = event.dataTransfer.files?.[0];

    processFile(file);
  };

  // =========================================================
  // ANALYZE
  // =========================================================

  const analyzeImage = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();

      formData.append("file", selectedFile);

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The backend returned an invalid response."
        );
      }

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.error ||
            `Backend error (${response.status})`
        );
      }

      if (!data.success) {
        throw new Error(
          data?.error ||
            "The image could not be analyzed."
        );
      }

      setResult(data);
      setViewMode("overlay");

      // Smoothly move user toward results
      setTimeout(() => {
        document
          .getElementById("analysis-results")
          ?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
      }, 100);
    } catch (err) {
      console.error(err);

      if (
        err.name === "TypeError" &&
        err.message.includes("fetch")
      ) {
        setError(
          "Unable to connect to the AI backend. Make sure FastAPI is running on port 8000."
        );
      } else {
        setError(
          err.message ||
            "Unable to analyze the image."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  // =========================================================
  // RESET
  // =========================================================

  const resetAnalysis = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    setViewMode("overlay");
    setHeatmapOpacity(0.45);
  };

  // =========================================================
  // RESULT HELPERS
  // =========================================================

  const confidencePercentage = result
    ? Math.min(
        Math.max(result.confidence * 100, 0),
        100
      )
    : 0;

  const predictedInfo = result
    ? CLASS_INFO[result.prediction]
    : null;

  const gradcamOverlay = result?.gradcam_overlay
    ? `data:image/png;base64,${result.gradcam_overlay}`
    : result?.gradcam
    ? `data:image/png;base64,${result.gradcam}`
    : null;

  const gradcamHeatmap = result?.gradcam_heatmap
    ? `data:image/png;base64,${result.gradcam_heatmap}`
    : null;

  const getConfidenceLabel = () => {
    if (confidencePercentage >= 95) return "Very high";
    if (confidencePercentage >= 80) return "High";
    if (confidencePercentage >= 60) return "Moderate";
    return "Low";
  };

  // =========================================================
  // RENDER
  // =========================================================

  return (
    <div className="app">

      {/* =====================================================
          HEADER
      ====================================================== */}

      <header className="topbar">
        <div className="topbar-inner">

          <div className="brand">

            <div className="brand-mark">
              <div className="brand-cell"></div>
              <div className="brand-cell"></div>
              <div className="brand-cell"></div>
              <div className="brand-cell"></div>
            </div>

            <div className="brand-text">
              <span className="brand-name">
                SynthMicro
              </span>

              <span className="brand-subtitle">
                Microscopic Cell Intelligence
              </span>
            </div>

          </div>

          <div className="topbar-right">

            <div className="model-pill">
              <span className="live-dot"></span>
              Model Online
            </div>

            <div className="model-name">
              ResNet50
            </div>

          </div>

        </div>
      </header>

      {/* =====================================================
          MAIN
      ====================================================== */}

      <main className="page">

        {/* ===================================================
            HERO
        ==================================================== */}

        <section className="hero">

          <div className="hero-grid">

            <div className="hero-content">

              <div className="eyebrow">
                <span className="eyebrow-line"></span>
                AI-ASSISTED MICROSCOPY
              </div>

              <h1>
                Leukemia Cell
                <span> Classification</span>
              </h1>

              <p className="hero-description">
                Analyze microscopic blood-cell images with
                a trained ResNet50 model and understand its
                predictions through explainable AI.
              </p>

              <div className="hero-tags">

                <span>
                  <b>01</b>
                  ResNet50
                </span>

                <span>
                  <b>02</b>
                  Grad-CAM
                </span>

                <span>
                  <b>03</b>
                  XAI
                </span>

              </div>

            </div>

            <div className="hero-visual">

  {/* Floating blood cells */}
  <div className="blood-cell cell-1">
    <div className="cell-nucleus"></div>
  </div>

  <div className="blood-cell cell-2">
    <div className="cell-nucleus"></div>
  </div>

  <div className="blood-cell cell-3">
    <div className="cell-nucleus"></div>
  </div>

  <div className="blood-cell cell-4">
    <div className="cell-nucleus"></div>
  </div>

  <div className="blood-cell cell-5">
    <div className="cell-nucleus"></div>
  </div>

  <div className="blood-cell cell-6">
    <div className="cell-nucleus"></div>
  </div>

  {/* Main scanning area */}
  <div className="scan-circle">

    <div className="scan-ring ring-one"></div>

    <div className="scan-ring ring-two"></div>

    <div className="scan-cross horizontal"></div>

    <div className="scan-cross vertical"></div>

    <div className="scan-core">

      <div className="core-cell">
        <div className="core-nucleus"></div>
      </div>

    </div>

  </div>

  <span className="hero-coordinate">
    224 × 224
  </span>

  <span className="scan-label">
    LIVE ANALYSIS
  </span>

</div>
          </div>

        </section>

        {/* ===================================================
            WORKFLOW
        ==================================================== */}

        <div className="workflow">

          <div className="workflow-item active">
            <span>01</span>
            Upload specimen
          </div>

          <div className="workflow-line"></div>

          <div
            className={`workflow-item ${
              result ? "active" : ""
            }`}
          >
            <span>02</span>
            AI classification
          </div>

          <div className="workflow-line"></div>

          <div
            className={`workflow-item ${
              result ? "active" : ""
            }`}
          >
            <span>03</span>
            Explain prediction
          </div>

        </div>

        {/* ===================================================
            UPLOAD
        ==================================================== */}

        <section className="panel upload-panel">

          <div className="panel-header">

            <div className="panel-heading">

              <div className="panel-index">
                01
              </div>

              <div>
                <span className="panel-kicker">
                  INPUT SPECIMEN
                </span>

                <h2>
                  Upload Cell Image
                </h2>

                <p>
                  Provide a microscopic blood-cell image
                  for model analysis.
                </p>
              </div>

            </div>

            <div className="format-note">
              JPG · JPEG · PNG
              <br />
              Max 10 MB
            </div>

          </div>

          <div
            className={`drop-zone ${
              dragActive ? "drag-active" : ""
            } ${
              preview ? "has-preview" : ""
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >

            {!preview ? (

              <div className="drop-content">

                <div className="microscope-icon">

                  <div className="scope-arm"></div>
                  <div className="scope-head"></div>
                  <div className="scope-stage"></div>
                  <div className="scope-base"></div>

                </div>

                <h3>
                  Drop your microscopy image here
                </h3>

                <p>
                  Drag and drop your file or browse
                  your computer
                </p>

                <label className="browse-button">

                  Browse Files

                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={handleFileChange}
                  />

                </label>

              </div>

            ) : (

              <div className="preview-layout">

                <div className="preview-image-container">

                  <img
                    src={preview}
                    alt="Selected microscopy specimen"
                  />

                  <div className="preview-badge">
                    Image ready
                  </div>

                </div>

                <div className="preview-details">

                  <span className="ready-label">
                    SPECIMEN READY
                  </span>

                  <h3>
                    Ready for analysis
                  </h3>

                  <p>
                    Your image has been loaded and is
                    ready to be processed by the
                    ResNet50 model.
                  </p>

                  <div className="file-meta">

                    <div>
                      <span>FILE</span>
                      <strong>
                        {selectedFile?.name}
                      </strong>
                    </div>

                    <div>
                      <span>SIZE</span>
                      <strong>
                        {(
                          selectedFile?.size /
                          1024
                        ).toFixed(1)} KB
                      </strong>
                    </div>

                  </div>

                  <label className="secondary-button">

                    Change image

                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/jpg"
                      onChange={handleFileChange}
                    />

                  </label>

                </div>

              </div>

            )}

          </div>

          <div className="upload-footer">

            <div className="privacy-note">
              <span className="shield-icon">
                ✓
              </span>

              <span>
                Image is processed locally by the
                application backend.
              </span>
            </div>

            <div className="upload-actions">

              {selectedFile && (

                <button
                  className="text-button"
                  onClick={resetAnalysis}
                  disabled={loading}
                >
                  Clear
                </button>

              )}

              <button
                className="primary-button"
                onClick={analyzeImage}
                disabled={!selectedFile || loading}
              >

                {loading ? (

                  <>
                    <span className="button-spinner"></span>
                    Analyzing specimen
                  </>

                ) : (

                  <>
                    Analyze specimen
                    <span>→</span>
                  </>

                )}

              </button>

            </div>

          </div>

          {error && (

            <div className="error-box">
              <span>!</span>
              {error}
            </div>

          )}

        </section>

        {/* ===================================================
            RESULTS
        ==================================================== */}

        {result && (

          <section
            className="results-section"
            id="analysis-results"
          >

            {/* =================================================
                RESULT HEADER
            ================================================== */}

            <div className="results-heading">

              <div>

                <div className="section-eyebrow">
                  ANALYSIS COMPLETE
                </div>

                <h2>
                  Classification Result
                </h2>

                <p>
                  The model has analyzed the uploaded
                  microscopy specimen.
                </p>

              </div>

              <button
                className="outline-button"
                onClick={resetAnalysis}
              >
                + New analysis
              </button>

            </div>

            {/* =================================================
                PREDICTION CARD
            ================================================== */}

            <div className="prediction-panel">

              <div className="prediction-main">

                <div className="prediction-status">
                  <span className="status-check">
                    ✓
                  </span>

                  MODEL PREDICTION
                </div>

                <h3>
                  {predictedInfo?.title ||
                    formatClassName(
                      result.prediction
                    )}
                </h3>

                <p>
                  {predictedInfo?.short ||
                    "Classification generated by the trained model."}
                </p>

                <div className="prediction-category">

                  <span>
                    Category
                  </span>

                  <strong>
                    {predictedInfo?.category ||
                      "Cell Classification"}
                  </strong>

                </div>

              </div>

              <div className="confidence-panel">

                <div className="confidence-top">

                  <span>
                    CONFIDENCE
                  </span>

                  <strong>
                    {confidencePercentage.toFixed(2)}%
                  </strong>

                </div>

                <div className="confidence-bar">

                  <div
                    style={{
                      width: `${confidencePercentage}%`,
                    }}
                  />

                </div>

                <div className="confidence-bottom">

                  <span>
                    Model certainty
                  </span>

                  <strong>
                    {getConfidenceLabel()}
                  </strong>

                </div>

              </div>

            </div>

            {/* =================================================
                EXPLAINABLE AI
            ================================================== */}

            <section className="xai-section">

              <div className="xai-heading">

                <div>

                  <div className="section-eyebrow">
                    EXPLAINABLE AI
                  </div>

                  <h2>
                    Where did the model look?
                  </h2>

                  <p>
                    Grad-CAM visualizes regions that
                    contributed to the model's selected
                    classification.
                  </p>

                </div>

                <div className="xai-method">
                  <span>METHOD</span>
                  Grad-CAM
                </div>

              </div>

              {/* =================================================
                  MAIN VISUALIZATION
              ================================================== */}

              <div className="xai-workspace">

                <div className="visual-main">

                  <div className="visual-toolbar">

                    <div className="view-tabs">

                      <button
                        className={
                          viewMode === "original"
                            ? "active"
                            : ""
                        }
                        onClick={() =>
                          setViewMode("original")
                        }
                      >
                        Original
                      </button>

                      <button
                        className={
                          viewMode === "heatmap"
                            ? "active"
                            : ""
                        }
                        onClick={() =>
                          setViewMode("heatmap")
                        }
                      >
                        Heatmap
                      </button>

                      <button
                        className={
                          viewMode === "overlay"
                            ? "active"
                            : ""
                        }
                        onClick={() =>
                          setViewMode("overlay")
                        }
                      >
                        Overlay
                      </button>

                    </div>

                    <span className="resolution">
                      224 × 224
                    </span>

                  </div>

                  <div className="main-image-frame">

                    {viewMode === "original" && (

                      <img
                        src={preview}
                        alt="Original microscopy image"
                      />

                    )}

                    {viewMode === "heatmap" && (

                      gradcamHeatmap ? (

                        <img
                          src={gradcamHeatmap}
                          alt="Grad-CAM heatmap"
                        />

                      ) : (

                        <div className="visual-error">
                          Heatmap unavailable
                        </div>

                      )

                    )}

                    {viewMode === "overlay" && (

                      gradcamOverlay ? (

                        <img
                          src={gradcamOverlay}
                          alt="Grad-CAM overlay"
                          style={{
                            opacity:
                              0.65 +
                              heatmapOpacity * 0.35,
                          }}
                        />

                      ) : (

                        <div className="visual-error">
                          Grad-CAM unavailable
                        </div>

                      )

                    )}

                    <div className="image-corner top-left"></div>
                    <div className="image-corner top-right"></div>
                    <div className="image-corner bottom-left"></div>
                    <div className="image-corner bottom-right"></div>

                  </div>

                  <div className="visual-caption">

                    <div>

                      <span className="caption-dot"></span>

                      <strong>
                        {viewMode === "original"
                          ? "Original specimen"
                          : viewMode === "heatmap"
                          ? "Activation intensity"
                          : "Grad-CAM overlay"}
                      </strong>

                    </div>

                    <span>
                      {viewMode === "heatmap"
                        ? "Blue → Red = increasing activation"
                        : "Regions highlighted by model activation"}
                    </span>

                  </div>

                </div>

                {/* =================================================
                    VISUAL SIDE PANEL
                ================================================== */}

                <aside className="visual-sidebar">

                  <div className="sidebar-block">

                    <span className="sidebar-label">
                      VISUALIZATION
                    </span>

                    <h3>
                      Grad-CAM
                    </h3>

                    <p>
                      Warmer colors represent stronger
                      model activation toward the selected
                      class.
                    </p>

                  </div>

                  <div className="legend-block">

                    <span className="sidebar-label">
                      ACTIVATION SCALE
                    </span>

                    <div className="gradient-vertical"></div>

                    <div className="gradient-labels">
                      <span>High</span>
                      <span>Low</span>
                    </div>

                  </div>

                  <div className="slider-block">

                    <div className="slider-title">

                      <span>
                        Overlay intensity
                      </span>

                      <strong>
                        {Math.round(
                          heatmapOpacity * 100
                        )}
                        %
                      </strong>

                    </div>

                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={heatmapOpacity}
                      onChange={(event) =>
                        setHeatmapOpacity(
                          Number(
                            event.target.value
                          )
                        )
                      }
                    />

                    <div className="slider-range">
                      <span>Subtle</span>
                      <span>Strong</span>
                    </div>

                  </div>

                  <div className="xai-note">

                    <div className="xai-note-icon">
                      i
                    </div>

                    <p>
                      Grad-CAM is an interpretability
                      visualization. It shows model
                      activation and should not be
                      interpreted as a clinical finding.
                    </p>

                  </div>

                </aside>

              </div>

              {/* =================================================
                  COMPARISON CARDS
              ================================================== */}

              <div className="comparison-grid">

                <div className="comparison-card">

                  <div className="comparison-header">

                    <span>A</span>

                    <div>
                      <strong>
                        Original
                      </strong>

                      <small>
                        Input specimen
                      </small>
                    </div>

                  </div>

                  <div className="comparison-image">

                    <img
                      src={preview}
                      alt="Original specimen"
                    />

                  </div>

                </div>

                <div className="comparison-card featured">

                  <div className="comparison-header">

                    <span>B</span>

                    <div>
                      <strong>
                        Grad-CAM
                      </strong>

                      <small>
                        Model attention overlay
                      </small>
                    </div>

                  </div>

                  <div className="comparison-image">

                    {gradcamOverlay ? (

                      <img
                        src={gradcamOverlay}
                        alt="Grad-CAM overlay"
                      />

                    ) : (

                      <div className="visual-error">
                        Unavailable
                      </div>

                    )}

                  </div>

                </div>

                <div className="comparison-card">

                  <div className="comparison-header">

                    <span>C</span>

                    <div>
                      <strong>
                        Heatmap
                      </strong>

                      <small>
                        Activation map
                      </small>
                    </div>

                  </div>

                  <div className="comparison-image">

                    {gradcamHeatmap ? (

                      <img
                        src={gradcamHeatmap}
                        alt="Activation heatmap"
                      />

                    ) : (

                      <div className="visual-error">
                        Unavailable
                      </div>

                    )}

                  </div>

                </div>

              </div>

            </section>

            {/* =================================================
                CELL INFORMATION
            ================================================== */}

            <section className="information-section">

              <div className="info-card cell-card">

                <div className="info-card-heading">

                  <div className="info-icon">
                    ◉
                  </div>

                  <div>

                    <span className="section-eyebrow">
                      CELL PROFILE
                    </span>

                    <h3>
                      {predictedInfo?.title ||
                        formatClassName(
                          result.prediction
                        )}
                    </h3>

                  </div>

                </div>

                <p>
                  {predictedInfo?.description ||
                    "The model identified this cell based on learned microscopic image features."}
                </p>

              </div>

              {/* =================================================
                  PROBABILITIES
              ================================================== */}

              <div className="info-card probabilities-card">

                <div className="info-card-heading">

                  <div className="info-icon">
                    ≡
                  </div>

                  <div>

                    <span className="section-eyebrow">
                      MODEL OUTPUT
                    </span>

                    <h3>
                      Class Probabilities
                    </h3>

                  </div>

                </div>

                <div className="probability-list">

                  {Object.entries(
                    result.probabilities || {}
                  )
                    .sort(
                      ([, a], [, b]) =>
                        b - a
                    )
                    .map(
                      ([name, probability]) => {

                        const percentage =
                          Math.min(
                            Math.max(
                              probability * 100,
                              0
                            ),
                            100
                          );

                        const predicted =
                          name ===
                          result.prediction;

                        return (

                          <div
                            className={`probability-item ${
                              predicted
                                ? "predicted"
                                : ""
                            }`}
                            key={name}
                          >

                            <div className="probability-top">

                              <span>

                                {predicted && (
                                  <b className="mini-check">
                                    ✓
                                  </b>
                                )}

                                {formatClassName(
                                  name
                                )}

                              </span>

                              <strong>
                                {percentage.toFixed(2)}%
                              </strong>

                            </div>

                            <div className="probability-track">

                              <div
                                style={{
                                  width: `${Math.max(
                                    percentage,
                                    percentage > 0
                                      ? 0.4
                                      : 0
                                  )}%`,
                                }}
                              />

                            </div>

                          </div>

                        );
                      }
                    )}

                </div>

              </div>

            </section>

            {/* =================================================
                DISCLAIMER
            ================================================== */}

            <div className="research-disclaimer">

              <div className="disclaimer-icon">
                !
              </div>

              <div>

                <strong>
                  Research & Educational Use
                </strong>

                <p>
                  SynthMicro is an AI-assisted research
                  and educational application. Model
                  predictions, confidence values and
                  Grad-CAM visualizations are not clinical
                  diagnoses and should not replace evaluation
                  by a qualified healthcare professional.
                </p>

              </div>

            </div>

          </section>

        )}

      </main>

      {/* =====================================================
          FOOTER
      ====================================================== */}

      <footer className="footer">

        <div className="footer-inner">

          <div>
            <strong>
              SynthMicro
            </strong>

            <span>
              Explainable AI for microscopic cell analysis
            </span>
          </div>

          <div className="footer-tech">
            ResNet50
            <span>•</span>
            Grad-CAM
            <span>•</span>
            FastAPI
          </div>

        </div>

      </footer>

    </div>
  );
}

export default App;