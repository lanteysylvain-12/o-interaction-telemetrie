"""Simulation de deux animaux dont les positions suivent des processus d'Ornstein-Uhlenbeck 2D.

Modèle (pour chaque coordonnée, les deux coordonnées étant indépendantes) :

    dX1 = theta (m(t) - X1) dt + gamma (X2 - X1) dt + sigma dW1
    dX2 = theta (m(t) - X2) dt + gamma (X1 - X2) dt + sigma dW2

- theta : rappel vers l'attracteur commun m(t) (point de nourrissage, rythme nycthéméral)
- gamma : intensité de l'interaction (attraction mutuelle) ; gamma = 0 <=> animaux indépendants
- sigma : intensité du bruit

La simulation utilise un schéma d'Euler-Maruyama sur un pas fin, puis un sous-échantillonnage
au pas des fixes GPS. Elle est donc indépendante des formules exactes utilisées pour l'inférence.
"""
import numpy as np

OMEGA = 2.0 * np.pi / 24.0  # pulsation du rythme nycthéméral (temps en heures)


def daily_mean(t, feeding_point=(0.0, 0.0), amplitude=0.0):
    """Attracteur commun m(t), de forme (len(t), 2).

    amplitude = 0 : attracteur constant (point de nourrissage).
    amplitude > 0 : l'attracteur décrit un cercle de rayon `amplitude` (km) en 24 h
    autour du point de nourrissage (rythme nycthéméral commun aux deux animaux).
    """
    t = np.asarray(t, dtype=float)
    mx = feeding_point[0] + amplitude * np.cos(OMEGA * t)
    my = feeding_point[1] + amplitude * np.sin(OMEGA * t)
    return np.stack([mx, my], axis=-1)


def simulate_pair(n_rep, n_steps, dt=0.5, theta=0.2, gamma=0.0, sigma=0.2,
                  amplitude=0.0, feeding_point=(0.0, 0.0), substeps=10,
                  burn_in_hours=48.0, rng=None):
    """Simule `n_rep` réplicats de deux trajectoires 2D observées toutes les `dt` heures.

    Retourne (t, X) avec t de forme (n_steps,) et X de forme (n_rep, n_steps, 2 animaux, 2 coordonnées).
    """
    rng = np.random.default_rng(rng)
    h = dt / substeps
    n_burn = int(round(burn_in_hours / h))
    n_total = n_burn + (n_steps - 1) * substeps
    t_fine = (np.arange(n_total + 1) - n_burn) * h               # le temps 0 = fin du burn-in
    m_fine = daily_mean(t_fine, feeding_point, amplitude)          # (n_total+1, 2)

    X = np.empty((n_rep, 2, 2))                                     # (rep, animal, coord)
    X[:] = m_fine[0]
    X += rng.normal(0.0, sigma / np.sqrt(2 * theta), size=X.shape)  # départ ~ loi stationnaire
    out = np.empty((n_rep, n_steps, 2, 2))
    sq = sigma * np.sqrt(h)
    k_out = 0
    for k in range(n_total + 1):
        if k >= n_burn and (k - n_burn) % substeps == 0:
            out[:, k_out] = X
            k_out += 1
        if k == n_total:
            break
        drift = theta * (m_fine[k] - X)
        drift[:, 0] += gamma * (X[:, 1] - X[:, 0])
        drift[:, 1] += gamma * (X[:, 0] - X[:, 1])
        X = X + drift * h + sq * rng.normal(size=X.shape)
    t = np.arange(n_steps) * dt
    return t, out
