"""Computer vision (OpenCV) on the heatmap image: edges, high-value regions, contours, 3-panel diagram."""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coolsolve_data import load, grid

D, X, Y, Z, DEMO = load()
_, xx, yy, zz = grid(D, X, Y, Z)


def make_heatmap(path="vision_heatmap.png"):
    f, a = plt.subplots(figsize=(6, 5)); a.imshow(zz, origin="lower", cmap="viridis", aspect="auto"); a.axis("off")
    f.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0); plt.close(f); return path


def main():
    img = cv2.imread(make_heatmap()); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); mask = cv2.inRange(hsv, (20, 120, 150), (40, 255, 255))   # yellow = highest values
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); out = img.copy()
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        x, y, w, h = cv2.boundingRect(c); cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite("vision_edges.png", edges)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title("Source heatmap")
    ax[1].imshow(edges, cmap="gray"); ax[1].set_title("Canny edges")
    ax[2].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); ax[2].set_title(f"High-value regions ({len(cnts)} found)")
    for a in ax: a.axis("off")
    fig.suptitle("Computer vision on the result heatmap" + (" [DEMO DATA]" if DEMO else "")); fig.tight_layout(); fig.savefig("vision_result.png", dpi=120); plt.close(fig)
    # 3D from image intensity (height map)
    g = cv2.resize(gray, (80, 60)).astype(float); gx, gy = np.meshgrid(np.arange(80), np.arange(60))
    f = plt.figure(figsize=(8, 6)); a = f.add_subplot(projection="3d"); a.plot_surface(gx, gy, g, cmap="gray"); a.set_title("3D height map from image intensity")
    f.tight_layout(); f.savefig("3d_vision_heightmap.png", dpi=120); plt.close(f); print("vision done")


if __name__ == "__main__":
    main()
