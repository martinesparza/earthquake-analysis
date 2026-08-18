"""PCA + KMeans clustering on rotated (rc/vt/ml) post-perturbation kinematics, across sessions.

Mirrors the "## 5. Are there discrete patterns of activity?" backbone in
notebooks/behaviour/kinematics_001.ipynb: cluster trials directly on their body-relative
kinematics (no synergy-fitting step, unlike the network-drive kinematic-motifs pipeline this was
inspired by) and, critically, validate any apparent clustering against a structure-destroying
permutation null before trusting it -- a naive silhouette score is high even on unstructured data
because KMeans always finds *some* partition.

Pipeline, per session:
    1. common_utils.load_session(session)                    -- load, preprocess, drop unperturbed
    2. dt.add_bhv(td, bhv_fields=["pos_keypoints"])           -- stack keypoints into `bhv`
    3. pyal.restrict_to_interval(idx_sol_on, -200..200 bins)  -- +/-2s around perturbation onset
    4. kin.rotate_bhv_td(td)                                  -- rotate into body frame (rc/vt/ml)
    5. flatten POST window (time x keypoint-dim) -> one row per trial, StandardScaler, PCA(20)
    6. KMeans k=2..8, silhouette score per k -> k_best = argmax
    7. N_NULL=50 surrogates: permute each PC independently across trials (destroys joint/cluster
       structure, keeps each PC's own marginal distribution), silhouette at k_best on each
    8. p = fraction of null silhouettes >= observed silhouette at k_best

Saves one .npz per session to `data/<session>.npz`:
    k_range          (n_k,)            values of k swept (2..8, or fewer if capped by n_trials)
    sil_scores        (n_k,)            silhouette score per k
    k_best            int               k with the highest silhouette
    obs_sil           float             silhouette score at k_best
    null_sil          (N_NULL,)         permutation-null silhouette scores at k_best
    null95            float             95th percentile of null_sil
    p_value           float             fraction of null_sil >= obs_sil
    cluster_labels    (n_trials,)       KMeans labels at k_best, for later re-plotting
    X_pca             (n_trials, k_pca) PCA-reduced features used for clustering
    explained_var     (k_pca,)          PCA explained_variance_ratio_
    sol_dir           (n_trials,)       values_Sol_direction, for later re-plotting
    n_trials          int
    session           str

Usage:
    uv run python across-sessions/clustering/compute_clustering.py
    uv run python across-sessions/clustering/compute_clustering.py --sessions M061_2025_03_06_14_00
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pyaldata as pyal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))  # across-sessions/, for common_utils
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for tools
import common_utils as cu  # noqa: E402
import tools.dataTools as dt  # noqa: E402
import tools.kinematics as kin  # noqa: E402

FS = 100
POST = (0.0, 1.5)  # response window (s, rel. onset) -- clustering runs on this window
REL_START, REL_END = -200, 200  # bins around idx_sol_on -> +/-2s epoch
N_PCA = 20
N_NULL = 50
SEED = 0

OUT_DIR = Path(__file__).resolve().parent / "data"


def run(session, data_dir=cu.DATA_DIR, std=cu.STD):
    print(f"\n{'='*72}\n### {session}", flush=True)

    td = cu.load_session(session, data_dir=data_dir, std=std, run_pca=False)
    td = dt.add_bhv(td, bhv_fields=["pos_keypoints"])
    td = pyal.restrict_to_interval(
        td, start_point_name="idx_sol_on", rel_start=REL_START, rel_end=REL_END
    )
    if len(td) < 10:
        print(f"  !! only {len(td)} trials left, skipping (need >= 10)")
        return

    onset = int(td.idx_sol_on.values[0])
    post_sl = slice(onset + int(POST[0] * FS), onset + int(POST[1] * FS))

    td, R, yaw = kin.rotate_bhv_td(td, verbose=False)
    X_ = np.stack(td.bhv_rot.values[:])

    X_clust = X_[:, post_sl, :]
    n_trials, n_time, n_features = X_clust.shape
    X_flat = X_clust.reshape(n_trials, -1)
    X_scaled = StandardScaler().fit_transform(X_flat)

    k_pca = min(N_PCA, n_trials - 1, X_flat.shape[1])
    pca = PCA(n_components=k_pca, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  {n_trials} trials, {X_flat.shape[1]} raw features -> {k_pca} PCs "
          f"({pca.explained_variance_ratio_.sum():.1%} var)")

    k_range = [k for k in range(2, 9) if k < n_trials]
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = km.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, labels))
    sil_scores = np.array(sil_scores)

    k_best = k_range[int(np.argmax(sil_scores))]
    km = KMeans(n_clusters=k_best, n_init=10, random_state=SEED)
    cluster_labels = km.fit_predict(X_pca)
    obs_sil = sil_scores.max()

    rng = np.random.default_rng(SEED)
    null_sil = np.full(N_NULL, np.nan)
    for i in range(N_NULL):
        X_shuff = np.column_stack([rng.permutation(X_pca[:, j]) for j in range(X_pca.shape[1])])
        km_null = KMeans(n_clusters=k_best, n_init=10, random_state=SEED)
        labels_null = km_null.fit_predict(X_shuff)
        null_sil[i] = silhouette_score(X_shuff, labels_null)

    null95 = np.percentile(null_sil, 95)
    p_value = (np.sum(null_sil >= obs_sil) + 1) / (N_NULL + 1)
    print(f"  k_best={k_best}: observed silhouette={obs_sil:.3f}, null95={null95:.3f}, "
          f"p={p_value:.3f}{'  ** SIGNIFICANT **' if p_value < 0.05 else ''}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{session}.npz"
    np.savez_compressed(
        out,
        k_range=np.array(k_range),
        sil_scores=sil_scores,
        k_best=k_best,
        obs_sil=obs_sil,
        null_sil=null_sil,
        null95=null95,
        p_value=p_value,
        cluster_labels=cluster_labels,
        X_pca=X_pca,
        explained_var=pca.explained_variance_ratio_,
        sol_dir=td.values_Sol_direction.values.astype(float),
        n_trials=n_trials,
        session=session,
    )
    print(f"  -> {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", default=cu.ALL_SESSIONS, metavar="SESSION")
    args = parser.parse_args()

    for session in args.sessions:
        try:
            run(session)
        except Exception as e:
            print(f"  !! FAILED {session}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
