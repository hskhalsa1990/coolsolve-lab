# CoolSolve Lab — Python, Data Science, Computer Vision, 2D & 3D Diagrams

Companion toolkit for **[CoolSolve](https://github.com/)**, a C++ parser, structural analyzer and equation solver for an EES-compatible language (`.eescode`) that uses CoolProp for thermodynamic properties.
This repository drives the real CoolSolve binary from Python, then analyses the results with data-science, machine-learning and computer-vision tools and draws **2D and 3D diagrams, charts, graphs and comparison tables**.

> ⚠️ **Demo data notice:** the images below were generated from a **synthetic DEMO dataset** (a Carnot-style refrigeration trend with noise), because the CoolSolve binary was not available when they were made. Every image carries a `[DEMO DATA]` label. Run the notebook with a built CoolSolve binary to regenerate them from real solver output.

## Contents (flat layout — no sub-folders)

| File | Purpose |
|---|---|
| `CoolSolve_Notebook.ipynb` | Full notebook: build/run CoolSolve, REST API, parametric sweep, data cleaning, analysis, ML, DL, computer vision, Flask, Bootstrap/Tailwind report, R/C/C++/Go/Bash, plus Part III (all 2D/3D charts) |
| `coolsolve_data.py` | Loads `coolsolve_sweep_clean.csv` (or a labelled DEMO dataset) |
| `coolsolve_charts.py` | 3D charts, 3D comparison tables, all standard 2D charts |
| `coolsolve_datascience.py` | Cleaning, statistics, ML (Linear / Random Forest), KMeans, PCA |
| `coolsolve_vision.py` | OpenCV: Canny edges, high-value region detection, image height-map |
| `run_all.py` | Runs everything and regenerates all PNG diagrams |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
pip install -r requirements.txt
python run_all.py          # writes all diagrams as PNG files next to the scripts
```

To use real data: run sections 1–10 of the notebook (this creates `coolsolve_sweep_clean.csv`), then run `python run_all.py`.

## 3D diagrams

| | |
|---|---|
| ![surface](3d_surface.png) | ![wireframe](3d_wireframe.png) |
| ![contour](3d_contour.png) | ![projections](3d_projections.png) |
| ![scatter](3d_scatter.png) | ![trisurf](3d_trisurf.png) |
| ![lines](3d_lines.png) | ![bar](3d_bar.png) |
| ![stem](3d_stem.png) | ![quiver](3d_quiver.png) |

## Comparison tables as 3D diagrams

![comparison bars](3d_comparison_bars.png)
![comparison metrics](3d_comparison_metrics.png)
![table](table_comparison.png)

## 2D charts and graphs

Line, bar, grouped bar, stacked bar, stacked area, scatter, histogram, box, violin, pie, donut, heatmap, filled contour, hexbin, error bars, step, stem, range band, polar, radar and correlation matrix.

![2D 1](2d_gallery_1.png)
![2D 2](2d_gallery_2.png)
![2D 3](2d_gallery_3.png)

## Data science modules

Cleaning (pandas), descriptive statistics, regression surrogate models (scikit-learn), clustering and PCA.

![ML](ml_results.png)
![ML surface 3D](3d_ml_surface.png)
![Clusters 3D](3d_clusters.png)
![PCA](pca.png)

## Computer vision (OpenCV)

The result heatmap is treated as an image: Canny edge detection, colour-mask segmentation of the highest-value region, contour boxes, and a 3D height map from pixel intensity.

![vision](vision_result.png)
![height map](3d_vision_heightmap.png)

## Notes

- The notebook's optional sections (TensorFlow, Flask, R, C, C++, Go) skip themselves when the tool is not installed.
- CoolSolve itself is a separate project (MIT licensed); this repo only wraps and visualises its output.
