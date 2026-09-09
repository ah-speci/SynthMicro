import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const CLASS_INFO = {
  basophil: {
    title: "Basophil",
    category: "White Blood Cell",
  },
  erythroblast: {
    title: "Erythroblast",
    category: "Red Blood Cell Precursor",
  },
  monocyte: {
    title: "Monocyte",
    category: "White Blood Cell",
  },
  myeloblast: {
    title: "Myeloblast",
    category: "Immature Myeloid Cell",
  },
  seg_neutrophil: {
    title: "Segmented Neutrophil",
    category: "White Blood Cell",
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
  const [dragActive, setDragActive] = useState(false);
  const [viewMode, setViewMode] = useState("original");
  const [selectedXaiCell, setSelectedXaiCell] = useState(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const processFile = (file) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid blood-smear image.");
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setError("Image size must be less than 20 MB.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
    setViewMode("original");
    setSelectedXaiCell(null);
  };

  const handleFileChange = (event) => {
    processFile(event.target.files?.[0]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    processFile(event.dataTransfer.files?.[0]);
  };

  const analyzeImage = async () => {
    if (!selectedFile) {
      setError("Please select a whole blood-smear image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data?.error || `Backend error (${response.status})`
        );
      }

      const normalized = {
        ...data,
        originalUrl: data.original_url
          ? `${API_URL}${data.original_url}`
          : preview,
        overlayUrl: data.overlay_url
          ? `${API_URL}${data.overlay_url}`
          : null,
        reportUrl: data.report_url
          ? `${API_URL}${data.report_url}`
          : null,
        gradcamRecords: (data.gradcam_records || []).map((item) => ({
          ...item,
          image_url: `${API_URL}${item.image_url}`,
        })),
        shapRecords: (data.shap_records || []).map((item) => ({
          ...item,
          image_url: `${API_URL}${item.image_url}`,
        })),
      };

      setResult(normalized);
      if (normalized.gradcamRecords.length > 0) {
        setSelectedXaiCell(normalized.gradcamRecords[0].cell_id);
      }
      setViewMode("overlay");

      setTimeout(() => {
        document
          .getElementById("analysis-results")
          ?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      console.error(err);
      setError(
        err?.message?.includes("fetch")
          ? "Unable to connect to the FastAPI backend on port 8000."
          : err.message || "Unable to analyze the blood smear."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    if (preview) URL.revokeObjectURL(preview);
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    setViewMode("original");
    setSelectedXaiCell(null);
  };

  const topPrediction = useMemo(() => {
    if (!result?.probabilities) return null;
    return Object.entries(result.probabilities).sort(
      ([, a], [, b]) => b - a
    )[0];
  }, [result]);

  const topConfidence = topPrediction ? topPrediction[1] * 100 : 0;
  const selectedGradcam =
    result?.gradcamRecords?.find(
      (item) => item.cell_id === selectedXaiCell
    ) || result?.gradcamRecords?.[0];
  const selectedShap =
    result?.shapRecords?.find(
      (item) => item.cell_id === selectedXaiCell
    ) || result?.shapRecords?.[0];

  const screeningTone =
    result?.screening_level === "ELEVATED_SCREENING_INDICATOR"
      ? "elevated"
      : result?.screening_level === "SCREENING_FLAG"
      ? "flag"
      : "lower";

  return (
    <div className="app">
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
              <span className="brand-name">SynthMicro</span>
              <span className="brand-subtitle">
                Whole-Smear Cell Intelligence
              </span>
            </div>
          </div>

          <div className="topbar-right">
            <div className="model-pill">
              <span className="live-dot"></span>
              Cellpose + ResNet50 + XAI
            </div>
            <div className="model-name">Phase 7</div>
          </div>
        </div>
      </header>

      <main className="page">
        <section className="hero">
          <div className="hero-grid">
            <div className="hero-content">
              <div className="eyebrow">
                <span className="eyebrow-line"></span>
                WHOLE BLOOD-SMEAR ANALYSIS
              </div>
              <h1>
                From Smear to
                <span> Explainable Analysis</span>
              </h1>
              <p className="hero-description">
                Upload a whole blood smear. SynthMicro segments cells with
                Cellpose, classifies candidate cells with the 384×384 ResNet50
                model, generates Grad-CAM and SHAP explanations, and builds a
                downloadable PDF report.
              </p>
              <div className="hero-tags">
                <span><b>01</b> Cellpose</span>
                <span><b>02</b> ResNet50</span>
                <span><b>03</b> Grad-CAM + SHAP</span>
                <span><b>04</b> HF Report</span>
              </div>
            </div>

            <div className="hero-visual">
              <div className="blood-cell cell-1"><div className="cell-nucleus"></div></div>
              <div className="blood-cell cell-2"><div className="cell-nucleus"></div></div>
              <div className="blood-cell cell-3"><div className="cell-nucleus"></div></div>
              <div className="blood-cell cell-4"><div className="cell-nucleus"></div></div>
              <div className="blood-cell cell-5"><div className="cell-nucleus"></div></div>
              <div className="blood-cell cell-6"><div className="cell-nucleus"></div></div>
              <div className="scan-circle">
                <div className="scan-ring ring-one"></div>
                <div className="scan-ring ring-two"></div>
                <div className="scan-cross horizontal"></div>
                <div className="scan-cross vertical"></div>
                <div className="scan-core">
                  <div className="core-cell"><div className="core-nucleus"></div></div>
                </div>
              </div>
              <span className="hero-coordinate">384 × 384</span>
              <span className="scan-label">WHOLE SMEAR</span>
            </div>
          </div>
        </section>

        <div className="workflow">
          <div className="workflow-item active"><span>01</span> Upload smear</div>
          <div className="workflow-line"></div>
          <div className={`workflow-item ${result ? "active" : ""}`}><span>02</span> Segment & classify</div>
          <div className="workflow-line"></div>
          <div className={`workflow-item ${result ? "active" : ""}`}><span>03</span> Explain & report</div>
        </div>

        <section className="panel upload-panel">
          <div className="panel-header">
            <div className="panel-heading">
              <div className="panel-index">01</div>
              <div>
                <span className="panel-kicker">INPUT SPECIMEN</span>
                <h2>Upload Whole Blood Smear</h2>
                <p>
                  Give the backend the complete microscopy image rather than a
                  pre-cropped cell.
                </p>
              </div>
            </div>
            <div className="format-note">JPG · JPEG · PNG<br />Max 20 MB</div>
          </div>

          <div
            className={`drop-zone ${dragActive ? "drag-active" : ""} ${preview ? "has-preview" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragActive(true); }}
            onDragLeave={(event) => { event.preventDefault(); setDragActive(false); }}
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
                <h3>Drop your blood smear here</h3>
                <p>Cellpose will segment the complete specimen automatically.</p>
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
                  <img src={preview} alt="Uploaded blood smear" />
                  <div className="preview-badge">Smear ready</div>
                </div>
                <div className="preview-details">
                  <span className="ready-label">SPECIMEN READY</span>
                  <h3>Ready for whole-smear analysis</h3>
                  <p>
                    The backend will run Cellpose, classify candidate cells,
                    generate XAI evidence and build the PDF report.
                  </p>
                  <div className="file-meta">
                    <div><span>FILE</span><strong>{selectedFile?.name}</strong></div>
                    <div><span>SIZE</span><strong>{(selectedFile?.size / 1024).toFixed(1)} KB</strong></div>
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
              <span className="shield-icon">✓</span>
              <span>Research workflow: uploaded data is processed by the local FastAPI backend.</span>
            </div>
            <div className="upload-actions">
              {selectedFile && (
                <button className="text-button" onClick={resetAnalysis} disabled={loading}>Clear</button>
              )}
              <button className="primary-button" onClick={analyzeImage} disabled={!selectedFile || loading}>
                {loading ? <><span className="button-spinner"></span>Running Cellpose + XAI</> : <>Analyze whole smear <span>→</span></>}
              </button>
            </div>
          </div>

          {error && <div className="error-box"><span>!</span>{error}</div>}
        </section>

        {result && (
          <section className="results-section" id="analysis-results">
            <div className="results-heading">
              <div>
                <div className="section-eyebrow">ANALYSIS COMPLETE</div>
                <h2>Whole-Smear Screening Result</h2>
                <p>
                  {result.cellpose_objects} Cellpose objects were processed;
                  {" "}{result.candidate_cells} were retained as nucleated-cell candidates.
                </p>
              </div>
              <div className="upload-actions">
                {result.reportUrl && (
                  <a className="outline-button" href={result.reportUrl} target="_blank" rel="noreferrer">
                    Open PDF report ↗
                  </a>
                )}
                <button className="outline-button" onClick={resetAnalysis}>+ New analysis</button>
              </div>
            </div>

            <div className="prediction-panel">
              <div className="prediction-main">
                <div className="prediction-status"><span className="status-check">✓</span> MODEL SUMMARY</div>
                <h3>{result.screening_label}</h3>
                <p>
                  {result.myeloblast_like_cells} myeloblast-like candidates among {result.candidate_cells} nucleated-cell candidates.
                </p>
                <div className="prediction-category">
                  <span>Myeloblast-like proportion</span>
                  <strong>{result.myeloblast_like_proportion.toFixed(2)}%</strong>
                </div>
              </div>

              <div className={`confidence-panel screening-${screeningTone}`}>
                <div className="confidence-top"><span>SCREENING INDICATOR</span><strong>{result.myeloblast_like_proportion.toFixed(1)}%</strong></div>
                <div className="confidence-bar"><div style={{ width: `${Math.min(result.myeloblast_like_proportion, 100)}%` }} /></div>
                <div className="confidence-bottom">
                  <span>Candidate-cell burden</span>
                  <strong>{result.screening_level === "ELEVATED_SCREENING_INDICATOR" ? "Elevated" : result.myeloblast_like_cells ? "Flagged" : "Lower"}</strong>
                </div>
              </div>
            </div>

            <section className="information-section">
              <div className="info-card cell-card">
                <div className="info-card-heading">
                  <div className="info-icon">◉</div>
                  <div><span className="section-eyebrow">SEGMENTATION</span><h3>Cellpose Results</h3></div>
                </div>
                <p>
                  The whole smear was segmented first. Only candidates with a
                  purple/blue score ≥ {result ? 35 : 35} were used for the
                  notebook-style nucleated-cell screening summary.
                </p>
                <div className="file-meta">
                  <div><span>OBJECTS</span><strong>{result.cellpose_objects}</strong></div>
                  <div><span>CANDIDATES</span><strong>{result.candidate_cells}</strong></div>
                </div>
              </div>

              <div className="info-card probabilities-card">
                <div className="info-card-heading">
                  <div className="info-icon">≡</div>
                  <div><span className="section-eyebrow">CANDIDATE DISTRIBUTION</span><h3>Cell Classes</h3></div>
                </div>
                <div className="probability-list">
                  {Object.entries(result.class_distribution || {}).sort(([, a], [, b]) => b - a).map(([name, count]) => {
                    const percentage = result.candidate_cells ? (count / result.candidate_cells) * 100 : 0;
                    return (
                      <div className="probability-item" key={name}>
                        <div className="probability-top"><span>{formatClassName(name)}</span><strong>{count} · {percentage.toFixed(1)}%</strong></div>
                        <div className="probability-track"><div style={{ width: `${Math.max(percentage, count ? 0.5 : 0)}%` }} /></div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>

            <section className="xai-section">
              <div className="xai-heading">
                <div>
                  <div className="section-eyebrow">EXPLAINABLE AI</div>
                  <h2>Cellpose → Grad-CAM → SHAP</h2>
                  <p>Inspect the whole-smear classification map and the strongest myeloblast-like candidate explanations.</p>
                </div>
                <div className="xai-method"><span>VLM</span>{result.vlm_model}</div>
              </div>

              <div className="xai-workspace">
                <div className="visual-main">
                  <div className="visual-toolbar">
                    <div className="view-tabs">
                      <button className={viewMode === "original" ? "active" : ""} onClick={() => setViewMode("original")}>Original</button>
                      <button className={viewMode === "overlay" ? "active" : ""} onClick={() => setViewMode("overlay")}>Cell map</button>
                      <button className={viewMode === "gradcam" ? "active" : ""} onClick={() => setViewMode("gradcam")}>Grad-CAM</button>
                      <button className={viewMode === "shap" ? "active" : ""} onClick={() => setViewMode("shap")}>SHAP</button>
                    </div>
                    <span className="resolution">384 × 384 cells</span>
                  </div>

                  <div className="main-image-frame">
                    {viewMode === "original" && <img src={result.originalUrl || preview} alt="Original blood smear" />}
                    {viewMode === "overlay" && result.overlayUrl && <img src={result.overlayUrl} alt="Cell classification overlay" />}
                    {viewMode === "gradcam" && selectedGradcam && <img src={selectedGradcam.image_url} alt="Grad-CAM explanation" />}
                    {viewMode === "shap" && selectedShap && <img src={selectedShap.image_url} alt="SHAP explanation" />}
                    {viewMode !== "original" && viewMode !== "overlay" && !selectedGradcam && !selectedShap && (
                      <div className="visual-error">No XAI image was generated for this analysis.</div>
                    )}
                    {viewMode === "overlay" && !result.overlayUrl && <div className="visual-error">Classification overlay unavailable.</div>}
                    <div className="image-corner top-left"></div><div className="image-corner top-right"></div><div className="image-corner bottom-left"></div><div className="image-corner bottom-right"></div>
                  </div>

                  <div className="visual-caption">
                    <div><span className="caption-dot"></span><strong>{viewMode === "original" ? "Original smear" : viewMode === "overlay" ? "Cell classification map" : viewMode === "gradcam" ? "Grad-CAM evidence" : "SHAP attribution"}</strong></div>
                    <span>{viewMode === "overlay" ? "Candidate labels are generated after Cellpose segmentation" : "Model explanation — not a clinical finding"}</span>
                  </div>
                </div>

                <aside className="visual-sidebar">
                  <div className="sidebar-block">
                    <span className="sidebar-label">SELECTED XAI CELL</span>
                    <h3>{selectedGradcam ? `Cell ${selectedGradcam.cell_id}` : "None"}</h3>
                    <p>{selectedGradcam ? `${formatClassName(selectedGradcam.class)} · ${(selectedGradcam.confidence * 100).toFixed(1)}%` : "No myeloblast-like candidate was selected."}</p>
                  </div>

                  <div className="legend-block">
                    <span className="sidebar-label">TOP MYELOBLAST-LIKE CELLS</span>
                    <div style={{ display: "grid", gap: 8, marginTop: 10 }}>
                      {result.gradcamRecords.map((item) => (
                        <button
                          key={item.cell_id}
                          onClick={() => { setSelectedXaiCell(item.cell_id); setViewMode("gradcam"); }}
                          style={{
                            textAlign: "left",
                            border: item.cell_id === selectedXaiCell ? "2px solid #333" : "1px solid #ddd",
                            background: "white",
                            padding: "8px 10px",
                            borderRadius: 8,
                            cursor: "pointer",
                          }}
                        >
                          Cell {item.cell_id} — {formatClassName(item.class)} ({(item.confidence * 100).toFixed(1)}%)
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="xai-note">
                    <div className="xai-note-icon">i</div>
                    <p>
                      Grad-CAM and SHAP explain the classifier's behavior. They do
                      not prove AML, leukemia, cancer, or a specific biological structure.
                    </p>
                  </div>
                </aside>
              </div>
            </section>

            <section className="information-section">
              <div className="info-card cell-card">
                <div className="info-card-heading">
                  <div className="info-icon">AI</div>
                  <div><span className="section-eyebrow">REPORT GENERATION</span><h3>Hugging Face VLM</h3></div>
                </div>
                <p>
                  The PDF report uses the configured Hugging Face vision-language
                  model to produce conservative explanations for selected cell
                  evidence. If HF inference is unavailable, the report falls back
                  to deterministic model-output explanations.
                </p>
                <strong>{result.vlm_enabled ? "VLM explanation enabled" : "VLM token not configured — fallback explanations used"}</strong>
              </div>

              <div className="info-card probabilities-card">
                <div className="info-card-heading">
                  <div className="info-icon">PDF</div>
                  <div><span className="section-eyebrow">FINAL REPORT</span><h3>Download Analysis</h3></div>
                </div>
                <p>Includes segmentation summary, candidate table, classification overlay, Grad-CAM, SHAP and limitations.</p>
                {result.reportUrl && (
                  <a className="primary-button" href={result.reportUrl} target="_blank" rel="noreferrer" style={{ display: "inline-flex", textDecoration: "none" }}>
                    Open PDF report →
                  </a>
                )}
              </div>
            </section>

            <div className="research-disclaimer">
              <div className="disclaimer-icon">!</div>
              <div>
                <strong>Research / screening support only</strong>
                <p>
                  This pipeline is not a clinical cancer detector. A high
                  myeloblast-like proportion is a screening indicator for further
                  review, not a diagnosis. The model is a five-class closed-set
                  classifier and does not contain an explicit RBC/unknown class.
                  Clinical interpretation must be performed by qualified professionals.
                </p>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <div className="footer-inner">
          <div><strong>SynthMicro</strong><span>Explainable whole-smear blood-cell analysis</span></div>
          <div className="footer-tech">Cellpose <span>•</span> ResNet50 <span>•</span> Grad-CAM <span>•</span> SHAP <span>•</span> Hugging Face</div>
        </div>
      </footer>
    </div>
  );
}

export default App;
