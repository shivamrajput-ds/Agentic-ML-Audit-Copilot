# Assets Guide

## Overview

The `assets/` directory contains all visual resources used throughout **Agentic ML Audit Copilot**.

These assets are referenced in the README, project documentation, GitHub repository, Docker Hub, and YouTube demonstration to illustrate the application's architecture, workflow, user interface, and deployment.

---

# Directory Structure

```text
assets/
│
├── branding/
│   ├── app_logo.png
│   ├── cover_image.png
│   ├── youtube_banner.png
│   ├── youtube_profile.png
│   └── youtube_thumbnail.png
│
├── demo/
│   ├── demo_gif.gif
│   ├── youtube_demo.png
│   └── youtube_demo.txt
│
├── diagrams/
│   ├── architecture.png
│   ├── audit_pipeline.png
│   ├── deployment_architecture.png
│   ├── langgraph_flow.png
│   ├── preprocessing_pipeline.png
│   ├── repository_structure.png
│   ├── system_design.png
│   └── workflow_graph.png
│
└── screenshots/
    ├── api_docs.png
    ├── audit_report.png
    ├── baseline_models.png
    ├── ci_passed.png
    ├── data_quality.png
    ├── dataset_upload.png
    ├── docker_hub.png
    ├── explainability_shap.png
    ├── feature_importance.png
    ├── leakage_risks.png
    ├── metric_recommendation.png
    ├── mlflow_tracking.png
    ├── streamlit_home.png
    ├── swagger_ui.png
    └── test_suite.png
```

---

# Branding Assets

The branding folder contains project identity resources.

| Asset | Purpose |
|--------|----------|
| `app_logo.png` | Application logo |
| `cover_image.png` | GitHub social preview |
| `youtube_banner.png` | YouTube channel banner |
| `youtube_profile.png` | YouTube profile picture |
| `youtube_thumbnail.png` | Project video thumbnail |

---

# Architecture Diagrams

These diagrams explain the overall system design.

| Diagram | Description |
|----------|-------------|
| `architecture.png` | High-level architecture |
| `audit_pipeline.png` | End-to-end audit pipeline |
| `deployment_architecture.png` | Deployment overview |
| `langgraph_flow.png` | LangGraph execution flow |
| `preprocessing_pipeline.png` | Data preprocessing workflow |
| `repository_structure.png` | Repository organization |
| `system_design.png` | System components |
| `workflow_graph.png` | Audit workflow graph |

---

# Application Screenshots

The `screenshots/` directory contains images captured from the running application.

Current screenshots include:

- Streamlit home page
- Dataset upload
- Data quality audit
- Leakage analysis
- Metric recommendation
- Baseline model comparison
- SHAP explainability
- Feature importance
- MLflow tracking
- Audit report
- FastAPI Swagger UI
- GitHub Actions CI
- Docker Hub image
- Test suite results

These screenshots are referenced throughout the README and documentation.

---

# Demo Assets

The `demo/` directory contains project demonstration resources.

| File | Purpose |
|------|----------|
| `demo_gif.gif` | Short application walkthrough |
| `youtube_demo.png` | YouTube video preview |
| `youtube_demo.txt` | YouTube demo URL |

Example:

```text
https://youtu.be/kFzNam74QBc
```

---

# Image Guidelines

Recommended formats:

- PNG for screenshots
- GIF for demonstrations
- SVG (optional) for diagrams

Recommended resolution:

- 1920 × 1080
- 2560 × 1440

Recommendations:

- Keep screenshots clean.
- Crop unnecessary desktop elements.
- Maintain consistent resolution.
- Use clear labels where appropriate.

---

# Naming Convention

Use lowercase filenames with underscores.

Examples:

```text
streamlit_home.png
mlflow_tracking.png
feature_importance.png
deployment_architecture.png
```

Avoid names such as:

```text
Screenshot (1).png
final_latest.png
image_new.png
```

---

# Updating Assets

When replacing an image:

- Keep the same filename whenever possible.
- Replace only if the new version improves quality.
- Verify that all documentation still references the correct path.
- Remove outdated or duplicate assets.

---

# Best Practices

- Keep diagrams synchronized with the latest architecture.
- Update screenshots after major UI changes.
- Compress images without noticeable quality loss.
- Store only repository-related assets.
- Keep demo GIFs under reasonable file sizes.

---

# Summary

The `assets/` directory centralizes all visual resources used throughout the project.

Maintaining a consistent naming convention and organized directory structure improves documentation quality, repository maintainability, and the overall presentation of **Agentic ML Audit Copilot**.