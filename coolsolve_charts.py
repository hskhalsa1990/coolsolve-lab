"""All 3D + 2D charts and comparison tables. Run: python coolsolve_charts.py  (PNG files land in this folder)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from coolsolve_data import load, grid

D, X, Y, Z, DEMO = load()
P, xx, yy, zz = grid(D, X, Y, Z)
others = [c for c in D.columns if c not in (X, Y)][:4]
TAG = "  [DEMO DATA]" if DEMO else ""
mid = P.iloc[len(P) // 2]


def save(fig, name):
    fig.suptitle(fig._suptitle.get_text() + TAG if fig._suptitle else TAG.strip(), fontsize=9, y=.995) if DEMO else None
    fig.tight_layout(); fig.savefig(f"{name}.png", dpi=120); plt.close(fig); print("saved", name)


def ax3(fig=None, size=(8, 6)):
    fig = plt.figure(figsize=size); return fig, fig.add_subplot(projection="3d")


def labels(ax, title):
    ax.set_xlabel(X); ax.set_ylabel(Y); ax.set_zlabel(Z); ax.set_title(title)


def three_d():
    f, a = ax3(); s = a.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="k", linewidth=.3); labels(a, "3D surface"); f.colorbar(s, shrink=.6); save(f, "3d_surface")
    f, a = ax3(); a.plot_wireframe(xx, yy, zz); labels(a, "3D wireframe"); save(f, "3d_wireframe")
    f, a = ax3(); a.contour3D(xx, yy, zz, 25, cmap="plasma"); labels(a, "3D contour"); save(f, "3d_contour")
    f, a = ax3(); a.plot_surface(xx, yy, zz, cmap="coolwarm", alpha=.8)
    a.contourf(xx, yy, zz, zdir="z", offset=zz.min(), cmap="coolwarm")
    a.contourf(xx, yy, zz, zdir="x", offset=xx.min(), cmap="coolwarm", alpha=.5)
    a.contourf(xx, yy, zz, zdir="y", offset=yy.max(), cmap="coolwarm", alpha=.5); labels(a, "3D surface with projections"); save(f, "3d_projections")
    f, a = ax3(); sc = a.scatter(D[X], D[Y], D[Z], c=D[Z], cmap="turbo", s=40); labels(a, "3D scatter"); f.colorbar(sc, shrink=.6); save(f, "3d_scatter")
    f, a = ax3(); t = a.plot_trisurf(D[X], D[Y], D[Z], cmap="magma"); labels(a, "3D triangulated surface"); f.colorbar(t, shrink=.6); save(f, "3d_trisurf")
    f, a = ax3()
    for yv, row in P.iterrows(): a.plot(P.columns, [yv] * len(row), row.values, marker="o", label=f"{Y}={yv:g}")
    a.legend(fontsize=7); labels(a, "3D multi-line graph"); save(f, "3d_lines")
    f, a = ax3(); xs, ys = np.meshgrid(np.arange(zz.shape[1]), np.arange(zz.shape[0])); xs, ys, h = xs.ravel(), ys.ravel(), zz.ravel()
    a.bar3d(xs, ys, np.zeros_like(h), .7, .7, h, color=cm.viridis((h - h.min()) / (np.ptp(h) + 1e-12)))
    a.set_xticks(np.arange(zz.shape[1]) + .35); a.set_xticklabels([f"{v:g}" for v in P.columns], fontsize=7)
    a.set_yticks(np.arange(zz.shape[0]) + .35); a.set_yticklabels([f"{v:g}" for v in P.index], fontsize=7); labels(a, "3D bar chart"); save(f, "3d_bar")
    f, a = ax3(); a.stem(D[X][:40], D[Y][:40], D[Z][:40]); labels(a, "3D stem"); save(f, "3d_stem")
    f, a = ax3(); gy, gx = np.gradient(zz, P.index.values, P.columns.values)
    a.quiver(xx, yy, zz, gx, gy, np.zeros_like(zz), length=3, normalize=True, color="tab:red"); a.plot_surface(xx, yy, zz, alpha=.3); labels(a, "3D gradient arrows"); save(f, "3d_quiver")


def comparison():
    f, a = ax3((9, 6))
    for k, (yv, row) in enumerate(P.iterrows()): a.bar(np.arange(len(row)), row.values, zs=k, zdir="y", width=.6, alpha=.85, color=cm.tab10(k % 10))
    a.set_xticks(range(len(P.columns))); a.set_xticklabels([f"{v:g}" for v in P.columns], fontsize=7)
    a.set_yticks(range(len(P.index))); a.set_yticklabels([f"{v:g}" for v in P.index], fontsize=7); labels(a, "Comparison table as 3D grouped bars"); save(f, "3d_comparison_bars")
    m = D[[X, Y] + others[:3]].head(6).reset_index(drop=True); v = m[others[:3]]; n = (v - v.min()) / (v.max() - v.min() + 1e-12)
    f, a = ax3((9, 6))
    for j, c in enumerate(n.columns): a.bar(np.arange(len(n)), n[c].values, zs=j, zdir="y", width=.6, alpha=.85, color=cm.Set2(j))
    a.set_yticks(range(len(n.columns))); a.set_yticklabels(n.columns, fontsize=7)
    a.set_xticks(range(len(n))); a.set_xticklabels([f"{p:g}/{q:g}" for p, q in zip(m[X], m[Y])], fontsize=7, rotation=20)
    a.set_zlabel("normalised"); a.set_title("Multi-metric comparison (3D)"); save(f, "3d_comparison_metrics")
    comp = P.round(3); f, a = plt.subplots(figsize=(9, 2 + .35 * len(comp))); a.axis("off")
    t = a.table(cellText=comp.values, rowLabels=[f"{v:g}" for v in comp.index], colLabels=[f"{v:g}" for v in comp.columns],
                cellColours=cm.YlGnBu((comp.values - np.nanmin(comp.values)) / (np.ptp(comp.values) + 1e-12)), loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); a.set_title(f"{Z}: rows={Y}, cols={X}"); save(f, "table_comparison")


def two_d():
    f, ax = plt.subplots(3, 3, figsize=(15, 12)); a = ax.ravel(); cols = [f"{v:g}" for v in P.columns]
    for yv, row in P.iterrows(): a[0].plot(P.columns, row.values, marker="o", label=f"{yv:g}")
    a[0].set_title("Line"); a[0].legend(fontsize=6)
    a[1].bar(cols, mid.values, color="tab:orange"); a[1].set_title("Bar")
    w = .8 / len(P)
    for i, (yv, row) in enumerate(P.iterrows()): a[2].bar(np.arange(len(row)) + i * w, row.values, w)
    a[2].set_title("Grouped bar"); bot = np.zeros(len(cols))
    for yv, row in P.iterrows(): a[3].bar(cols, row.values, bottom=bot); bot += row.values
    a[3].set_title("Stacked bar"); a[4].stackplot(P.columns, P.values); a[4].set_title("Stacked area")
    a[5].scatter(D[X], D[Z], c=D[Y], cmap="viridis"); a[5].set_title("Scatter")
    a[6].hist(D[Z], bins=12, color="tab:green", edgecolor="k"); a[6].set_title("Histogram")
    a[7].boxplot([r.values for _, r in P.iterrows()]); a[7].set_title("Box plot")
    a[8].violinplot([r.values for _, r in P.iterrows()]); a[8].set_title("Violin plot"); f.suptitle("2D charts (1/3)"); save(f, "2d_gallery_1")
    f, ax = plt.subplots(3, 3, figsize=(15, 12)); a = ax.ravel()
    a[0].pie(mid.values, labels=cols, autopct="%1.0f%%"); a[0].set_title("Pie")
    a[1].pie(mid.values, labels=cols, wedgeprops=dict(width=.4)); a[1].set_title("Donut")
    im = a[2].imshow(zz, origin="lower", aspect="auto", cmap="inferno", extent=[xx.min(), xx.max(), yy.min(), yy.max()]); f.colorbar(im, ax=a[2]); a[2].set_title("Heatmap")
    c = a[3].contourf(xx, yy, zz, 15, cmap="viridis"); f.colorbar(c, ax=a[3]); a[3].set_title("Filled contour")
    a[4].hexbin(D[X], D[Y], C=D[Z], gridsize=8, cmap="plasma"); a[4].set_title("Hexbin")
    a[5].errorbar(P.columns, P.mean(axis=0), yerr=P.std(axis=0), fmt="o-", capsize=4); a[5].set_title("Error bars")
    a[6].step(P.columns, mid.values, where="mid"); a[6].set_title("Step"); a[7].stem(P.columns, mid.values); a[7].set_title("Stem")
    a[8].fill_between(P.columns, P.min(axis=0), P.max(axis=0), alpha=.3); a[8].plot(P.columns, P.mean(axis=0)); a[8].set_title("Range band")
    f.suptitle("2D charts (2/3)"); save(f, "2d_gallery_2")
    f = plt.figure(figsize=(15, 5)); a = f.add_subplot(131, projection="polar"); th = np.linspace(0, 2 * np.pi, len(mid), endpoint=False)
    a.plot(np.r_[th, th[0]], np.r_[mid.values, mid.values[0]]); a.fill(th, mid.values, alpha=.3); a.set_title("Polar")
    a = f.add_subplot(132, projection="polar"); cc = others[:4] or [Z]; r = D[cc].mean(); r = (r - D[cc].min()) / (D[cc].max() - D[cc].min() + 1e-12)
    th = np.linspace(0, 2 * np.pi, len(cc), endpoint=False); a.plot(np.r_[th, th[0]], np.r_[r, r.iloc[0]]); a.fill(th, r, alpha=.3); a.set_xticks(th); a.set_xticklabels(cc, fontsize=7); a.set_title("Radar")
    a = f.add_subplot(133); cm_ = D.corr().iloc[:8, :8]; im = a.imshow(cm_, cmap="coolwarm", vmin=-1, vmax=1)
    a.set_xticks(range(len(cm_))); a.set_xticklabels(cm_.columns, rotation=60, fontsize=6); a.set_yticks(range(len(cm_))); a.set_yticklabels(cm_.index, fontsize=6); f.colorbar(im, ax=a); a.set_title("Correlation")
    f.suptitle("2D charts (3/3)"); save(f, "2d_gallery_3")


if __name__ == "__main__":
    three_d(); comparison(); two_d()
