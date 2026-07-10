# Assets Guide

## Overview

The `assets/` directory contains the visual resources used in **Agentic ML Audit Copilot**.

These assets are used in the README, documentation, GitHub repository preview, project demo, and portfolio presentation.

The goal of this folder is simple:

- Keep project visuals organized
- Avoid duplicate screenshots
- Keep documentation paths consistent
- Make the repository look clean and professional

---

## Final Directory Structure

Use this structure:

```text
assets/
├── architecture/
│   ├── 01_system_architecture.png
│   ├── 02_hitl_workflow.png
│   └── 03_fastapi_workflow.png
│
├── branding/
│   └── repo_banner.png
│
├── demo/
│   ├── demo_git.gif
│   └── demo_script.md
│
└── screenshots/
    ├── 01_streamlit_home.png
    ├── 02_human_review_gate.png
    ├── 03_executive_dashboard.png
    └── 04_fastapi_docs.png
```

Do not add extra image folders unless the project really needs them.

---

## Branding Assets

| File | Purpose |
| --- | --- |
| `assets/branding/repo_banner.png` | Main GitHub README banner |

This image should appear at the top of `README.md`.

Example:

```md
<p align="center">
  <img src="assets/branding/repo_banner.png" alt="Agentic ML Audit Copilot Banner">
</p>
```

---

## Architecture Assets

| File | Purpose |
| --- | --- |
| `assets/architecture/01_system_architecture.png` | Full system architecture |
| `assets/architecture/02_hitl_workflow.png` | Human-in-the-loop review workflow |
| `assets/architecture/03_fastapi_workflow.png` | FastAPI and HITL API workflow |

These images explain the project without requiring the reader to inspect the code first.

Recommended README usage:

```md
## System Architecture

![System Architecture](assets/architecture/01_system_architecture.png)

## Human-in-the-Loop Workflow

![Human Review Workflow](assets/architecture/02_hitl_workflow.png)

## FastAPI Workflow

![FastAPI Workflow](assets/architecture/03_fastapi_workflow.png)
```

---

## Screenshot Assets

| File | Purpose |
| --- | --- |
| `assets/screenshots/01_streamlit_home.png` | Streamlit home and dataset upload view |
| `assets/screenshots/02_human_review_gate.png` | Human Review Gate with risk cards and reviewer decisions |
| `assets/screenshots/03_executive_dashboard.png` | Executive dashboard with KPIs and audit status |
| `assets/screenshots/04_fastapi_docs.png` | FastAPI Swagger documentation |

These screenshots should be captured from the running application.

Recommended README usage:

```md
## Dashboard Preview

![Streamlit Home](assets/screenshots/01_streamlit_home.png)

![Human Review Gate](assets/screenshots/02_human_review_gate.png)

![Executive Dashboard](assets/screenshots/03_executive_dashboard.png)

![FastAPI Docs](assets/screenshots/04_fastapi_docs.png)
```

---

## Demo Assets

| File | Purpose |
| --- | --- |
| `assets/demo/demo_git.gif` | Short demo GIF for README |
| `assets/demo/demo_script.md` | Short demo notes or YouTube walkthrough script |

Example README usage:

```md
## Demo Preview

<p align="center">
  <img src="assets/demo/demo_git.gif" width="100%" alt="Project Demo">
</p>
```

---

## Screenshot Capture Checklist

Before taking screenshots, run the app locally:

```bash
uv run streamlit run app/streamlit_app.py --server.port 8501
```

Run FastAPI in a second terminal:

```bash
uv run uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

Capture these views:

1. Streamlit home page after app loads
2. Human Review Gate tab after running an audit with risks
3. Executive dashboard after audit results are available
4. FastAPI Swagger UI at `http://127.0.0.1:8000/docs`

---

## Image Guidelines

Recommended format:

- PNG for banners, diagrams, and screenshots
- GIF for short demo previews
- Markdown for demo notes or scripts

Recommended aspect ratios:

| Asset Type | Recommended Ratio |
| --- | --- |
| Repository banner | 16:9 |
| Architecture diagrams | 16:9 |
| Dashboard screenshots | 16:9 or full browser width |
| Demo GIF | 16:9 |

Best practices:

- Keep screenshots clean.
- Avoid desktop clutter.
- Avoid browser bookmarks or private information.
- Use high-resolution captures.
- Keep labels readable.
- Prefer fewer strong visuals over many weak screenshots.
- Replace old screenshots after major UI changes.

---

## Naming Convention

Use numbered, lowercase, descriptive filenames.

Good examples:

```text
01_system_architecture.png
02_hitl_workflow.png
03_fastapi_workflow.png
01_streamlit_home.png
02_human_review_gate.png
03_executive_dashboard.png
04_fastapi_docs.png
repo_banner.png
demo_git.gif
demo_script.md
```

Avoid:

```text
Screenshot (1).png
final_latest.png
new_image.png
copy2.png
banner_final_final.png
```

---

## Updating Assets

When updating an asset:

1. Keep the same filename if the purpose is unchanged.
2. Replace the old file instead of adding duplicates.
3. Check README image paths after replacing.
4. Check documentation image paths after replacing.
5. Run `git status` to confirm only intended files changed.

Useful command:

```bash
git status
```

Add assets:

```bash
git add assets/
```

---

## Paths Used in Documentation

From `README.md`, use paths like:

```text
assets/branding/repo_banner.png
assets/architecture/01_system_architecture.png
assets/screenshots/03_executive_dashboard.png
```

From files inside `docs/`, use paths like:

```text
../assets/architecture/01_system_architecture.png
../assets/architecture/02_hitl_workflow.png
../assets/architecture/03_fastapi_workflow.png
../assets/screenshots/04_fastapi_docs.png
```

This path difference is important because documentation files are inside the `docs/` folder.

---

## What Not to Add

Do not add too many screenshots.

Avoid adding:

- Multiple similar dashboard screenshots
- Outdated UI screenshots
- Random generated images
- Temporary images
- Local system screenshots with private paths
- Large files that slow down the repository

The final asset set should stay small and strong.

---

## Final Asset Set

The recommended final visual set is:

```text
repo_banner.png
01_system_architecture.png
02_hitl_workflow.png
03_fastapi_workflow.png
01_streamlit_home.png
02_human_review_gate.png
03_executive_dashboard.png
04_fastapi_docs.png
demo_git.gif
```

This is enough for a clean, recruiter-friendly GitHub repository.

---

## Summary

The `assets/` directory should stay simple, organized, and consistent.

A small set of high-quality visuals is better than many repeated or outdated images. Keep the folder clean, keep filenames stable, and make sure all README and documentation image paths match the actual files.
