"""Indices classiques d'interaction, inspirés de Long et al. (2014) et du package R wildlifeDI.

Ce sont des réimplémentations simplifiées (le package wildlifeDI n'est pas utilisé ici) :
- `corr` : corrélation de Pearson entre les positions des deux animaux (moyenne sur x et y) ;
- `prox` : proportion de fixes simultanés où la distance est inférieure à `delta` (indice Prox) ;
- `cs`   : coefficient de sociabilité de Kenward, (d_attendue - d_observée)/(d_attendue + d_observée),
           où d_attendue est la distance moyenne entre toutes les paires de positions.

Le test est un test de permutation unilatéral : on mélange aléatoirement les instants de l'animal 2
(hypothèse nulle : les deux animaux sont indépendants).
"""
import numpy as np


def _stats(X1, X2, delta):
    """X1, X2 : (..., n, 2). Retourne corr, prox, cs (tableaux de forme (...,))."""
    d = np.linalg.norm(X1 - X2, axis=-1)                       # (..., n)
    prox = (d < delta).mean(axis=-1)
    d_obs = d.mean(axis=-1)
    # distance moyenne entre toutes les paires (i, j) : invariante par permutation
    diff = X1[..., :, None, :] - X2[..., None, :, :]
    d_exp = np.linalg.norm(diff, axis=-1).mean(axis=(-1, -2))
    cs = (d_exp - d_obs) / (d_exp + d_obs)
    c = []
    for k in range(2):
        a = X1[..., k] - X1[..., k].mean(axis=-1, keepdims=True)
        b = X2[..., k] - X2[..., k].mean(axis=-1, keepdims=True)
        c.append((a * b).sum(-1) / np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1)))
    corr = 0.5 * (c[0] + c[1])
    return corr, prox, cs


def _corr(X1, X2p):
    """Corrélation de Pearson moyenne (x et y) entre X1 (n, 2) et chaque permutation X2p (B, n, 2)."""
    c = []
    for k in range(2):
        a = X1[:, k] - X1[:, k].mean()
        b = X2p[:, :, k] - X2p[:, :, k].mean(axis=1, keepdims=True)
        c.append((b @ a) / np.sqrt((a ** 2).sum() * (b ** 2).sum(axis=1)))
    return 0.5 * (c[0] + c[1])


def classical_tests(X1, X2, delta=0.3, n_perm=199, rng=None):
    """Test de permutation pour chaque indice. X1, X2 : (n, 2). Retourne indices et p-valeurs."""
    rng = np.random.default_rng(rng)
    n = X1.shape[0]
    obs = _stats(X1, X2, delta)
    perms = np.stack([rng.permutation(n) for _ in range(n_perm)])  # (B, n)
    X2p = X2[perms]                                                # (B, n, 2)
    # d_exp est identique pour toutes les permutations ; on évite de le recalculer B fois
    d = np.linalg.norm(X1[None] - X2p, axis=-1)
    prox_null = (d < delta).mean(axis=-1)
    diff = X1[:, None, :] - X2[None, :, :]
    d_exp = np.linalg.norm(diff, axis=-1).mean()
    d_null = d.mean(axis=-1)
    cs_null = (d_exp - d_null) / (d_exp + d_null)
    corr_null = _corr(X1, X2p)
    out = {}
    for name, o, nl in zip(("corr", "prox", "cs"), obs, (corr_null, prox_null, cs_null)):
        out[name] = float(o)
        out["p_" + name] = float((1 + np.sum(nl >= o)) / (n_perm + 1))
    return out
