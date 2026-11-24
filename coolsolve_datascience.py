"""Data cleaning, analysis and machine learning on the sweep. Produces 2D + 3D diagrams."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coolsolve_data import load

D, X, Y, Z, DEMO = load()
TAG = " [DEMO DATA]" if DEMO else ""


def clean(d):
    d = d.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    d = d.loc[:, d.nunique() > 1].drop_duplicates().reset_index(drop=True)
    return d


def main():
    d = clean(D); print("clean shape:", d.shape); print(d.describe().T.round(3))
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error
    xtr, xte, ytr, yte = train_test_split(d[[X, Y]].values, d[Z].values, test_size=.25, random_state=0)
    models = {"LinearRegression": LinearRegression(), "RandomForest": RandomForestRegressor(300, random_state=0)}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5)); res = []
    for (n, m), a in zip(models.items(), ax):
        m.fit(xtr, ytr); p = m.predict(xte); res.append((n, r2_score(yte, p), mean_absolute_error(yte, p)))
        a.scatter(yte, p); lo, hi = yte.min(), yte.max(); a.plot([lo, hi], [lo, hi], "r--"); a.set_title(f"{n} R2={res[-1][1]:.3f}"); a.set_xlabel("actual"); a.set_ylabel("predicted")
    rf = models["RandomForest"]; ax[2].bar([X, Y], rf.feature_importances_, color="tab:green"); ax[2].set_title("RandomForest feature importance")
    fig.suptitle("Machine learning surrogate" + TAG); fig.tight_layout(); fig.savefig("ml_results.png", dpi=120); plt.close(fig)
    print(pd.DataFrame(res, columns=["model", "R2", "MAE"]).round(4))
    # 3D: true points vs RandomForest prediction surface
    gx, gy = np.meshgrid(np.linspace(d[X].min(), d[X].max(), 30), np.linspace(d[Y].min(), d[Y].max(), 30))
    gz = rf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    f = plt.figure(figsize=(8, 6)); a = f.add_subplot(projection="3d"); a.plot_surface(gx, gy, gz, cmap="viridis", alpha=.7); a.scatter(d[X], d[Y], d[Z], c="r", s=15)
    a.set_xlabel(X); a.set_ylabel(Y); a.set_zlabel(Z); a.set_title("3D: data (red) vs ML surface" + TAG); f.tight_layout(); f.savefig("3d_ml_surface.png", dpi=120); plt.close(f)
    # Clusters (KMeans) in 3D
    from sklearn.cluster import KMeans
    k = KMeans(3, n_init=10, random_state=0).fit_predict((d[[X, Y, Z]] - d[[X, Y, Z]].mean()) / d[[X, Y, Z]].std())
    f = plt.figure(figsize=(8, 6)); a = f.add_subplot(projection="3d"); a.scatter(d[X], d[Y], d[Z], c=k, cmap="Set1", s=40)
    a.set_xlabel(X); a.set_ylabel(Y); a.set_zlabel(Z); a.set_title("3D KMeans clusters" + TAG); f.tight_layout(); f.savefig("3d_clusters.png", dpi=120); plt.close(f)
    # PCA 2D
    from sklearn.decomposition import PCA
    num = d.select_dtypes("number"); pc = PCA(2).fit_transform((num - num.mean()) / (num.std() + 1e-12))
    f, a = plt.subplots(figsize=(6, 5)); s = a.scatter(pc[:, 0], pc[:, 1], c=d[Z], cmap="viridis"); f.colorbar(s, label=Z); a.set_title("PCA (2 components)" + TAG)
    f.tight_layout(); f.savefig("pca.png", dpi=120); plt.close(f)


if __name__ == "__main__":
    main()
