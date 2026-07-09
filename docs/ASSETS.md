# Assets Guide

## Overview

The `assets/` directory contains all visual resources used throughout the project documentation, dashboard, and GitHub repository.

These assets help explain the system architecture, workflow, dashboard, and overall project structure.

---

# Folder Structure

```
assets/
│
├── branding/
│   ├── app_logo.png
│   ├── banner.png
│   ├── cover_image.png
│   └── favicon.png
│
├── diagrams/
│   ├── architecture.png
│   ├── audit_pipeline.png
│   ├── workflow_graph.png
│   ├── langgraph_flow.png
│   ├── preprocessing_pipeline.png
│   └── system_design.png
│
├── screenshots/
│   ├── ai_chatbot.png
│   ├── api_docs.png
│   ├── audit_report.png
│   ├── audit_score.png
│   ├── baseline_models.png
│   ├── class_imbalance.png
│   ├── data_quality.png
│   ├── dataset_upload.png
│   ├── explainability_shap.png
│   ├── feature_importance.png
│   ├── leakage_risks.png
│   ├── metric_recommendation.png
│   ├── mlflow_tracking.png
│   ├── profiler.png
│   ├── streamlit_home.png
│   ├── swagger_ui.png
│   ├── test_suite.png
│   └── ci_pipeline.png
│
└── demo/
    └── demo.gif
```

---

# Branding

The branding folder contains images used throughout the repository.

| Asset | Purpose |
|--------|----------|
| app_logo.png | Application logo |
| banner.png | README banner |
| cover_image.png | Social preview image |
| favicon.png | Browser icon |

---

# Architecture Diagrams

These diagrams explain the overall design of the application.

| Diagram | Description |
|----------|-------------|
| architecture.png | High-level system architecture |
| audit_pipeline.png | End-to-end audit pipeline |
| workflow_graph.png | Complete audit workflow |
| langgraph_flow.png | LangGraph execution flow |
| preprocessing_pipeline.png | Data preprocessing pipeline |
| system_design.png | Overall component interaction |

---

# Dashboard Screenshots

The screenshots folder contains images captured from the running application.

Examples include

- Home dashboard
- Dataset upload
- Data quality report
- Leakage detection
- Baseline model results
- SHAP explainability
- MLflow tracking
- Generated audit report

These images are referenced throughout the README and documentation.

---

# Demo

The demo folder contains a short screen recording of the application.

Recommended duration

- 20–60 seconds

Suggested flow

1. Launch Streamlit
2. Upload dataset
3. Select target column
4. Run audit
5. Review dashboard
6. Download report

---

# Image Guidelines

Recommended format

- PNG for screenshots
- GIF for demo
- SVG where appropriate for diagrams

Recommended resolution

- 1920 × 1080
- 2560 × 1440

Keep screenshots clean and avoid including unnecessary desktop elements.

---

# Naming Convention

Use lowercase names with underscores.

Examples

```
streamlit_home.png
```

```
mlflow_tracking.png
```

```
feature_importance.png
```

Avoid names such as

```
Screenshot (1).png
```

or

```
image_final_new.png
```

---

# Updating Assets

When updating an image

- Keep the same filename whenever possible.
- Replace only if the new version improves clarity.
- Ensure the README still references the correct path.

---

# Summary

The assets directory provides all visual resources required for the documentation and GitHub repository. Keeping images organized and consistently named makes the project easier to maintain and improves the overall presentation.