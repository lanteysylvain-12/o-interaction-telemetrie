"""Inférence exacte par maximum de vraisemblance du modèle d'interaction.

Idée clé : on passe aux coordonnées orthogonales
    u = (X1 + X2) / sqrt(2)   (mouvement commun)      -> OU de vitesse de rappel theta
    v = (X1 - X2) / sqrt(2)   (écart entre animaux)   -> OU de vitesse de rappel theta + 2 gamma
Ces deux processus sont indépendants (bruit isotrope), et l'attracteur commun m(t) n'agit que sur u.
Le paramètre gamma se lit donc dans la différence de vitesses de rappel entre u et v.

Discrétisation exacte au pas dt (avec m(t) affine par morceaux entre deux fixes pour u) :
    v_{k+1} = a_v v_k + eps_v,       a_v = exp(-(theta + 2 gamma) dt)
    u_{k+1} = a_u u_k + sqrt(2)[m_{k+1} - a_u m_k - (m_{k+1}-m_k)(1-a_u)/(theta dt)] + eps_u,  a_u = exp(-theta dt)
avec Var(eps_u) = sigma^2 (1-a_u^2)/(2 theta) et Var(eps_v) = sigma^2 (1-a_v^2)/(2 (theta+2 gamma)).

Les paramètres de moyenne (point de nourrissage, amplitude du rythme) entrent linéairement : ils sont
« profilés » par moindres carrés, et sigma est profilé analytiquement. Il reste à optimiser (theta, gamma).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from .simulate import OMEGA


def design(t, harmonics):
    """Matrice de régression de la moyenne : constante (+ cos/sin de période 24 h si harmonics=1)."""
    cols = [np.ones_like(t)]
    if harmonics >= 1:
        cols += [np.cos(OMEGA * t), np.sin(OMEGA * t)]
    return np.stack(cols, axis=1)


def _uv(X):
    """X : (n, 2 animaux, 2 coordonnées) -> U, V de forme (n, 2 coordonnées)."""
    return (X[:, 0] + X[:, 1]) / np.sqrt(2), (X[:, 0] - X[:, 1]) / np.sqrt(2)


def profile_nll(theta, gamma, U, V, Phi, dt):
    """Log-vraisemblance négative profilée (moyenne par MCO, sigma analytique)."""
    a_u, a_v = np.exp(-theta * dt), np.exp(-(theta + 2 * gamma) * dt)
    W = Phi[1:] - a_u * Phi[:-1] - (Phi[1:] - Phi[:-1]) * (1 - a_u) / (theta * dt)
    Yu = U[1:] - a_u * U[:-1]
    coef, *_ = np.linalg.lstsq(np.sqrt(2) * W, Yu, rcond=None)
    r_u = Yu - np.sqrt(2) * W @ coef
    r_v = V[1:] - a_v * V[:-1]
    q_u = (1 - a_u ** 2) / (2 * theta)
    q_v = (1 - a_v ** 2) / (2 * (theta + 2 * gamma))
    N = r_u.size
    sigma2 = ((r_u ** 2).sum() / q_u + (r_v ** 2).sum() / q_v) / (2 * N)
    nll = N * np.log(2 * np.pi * sigma2) + 0.5 * N * (np.log(q_u) + np.log(q_v)) + N
    return nll, sigma2, coef


def fit_h0(X, dt, harmonics=0):
    """Ajuste sous H0 (gamma = 0, animaux indépendants). X : (n, 2, 2)."""
    n = X.shape[0]
    t = np.arange(n) * dt
    Phi, (U, V) = design(t, harmonics), _uv(X)
    f = lambda p: profile_nll(np.exp(p[0]), 0.0, U, V, Phi, dt)[0]
    best = min((minimize(f, [np.log(s)], method="Nelder-Mead") for s in (0.05, 0.2, 1.0)),
               key=lambda r: r.fun)
    theta = float(np.exp(best.x[0]))
    nll, s2, coef = profile_nll(theta, 0.0, U, V, Phi, dt)
    return dict(theta=theta, gamma=0.0, sigma=float(np.sqrt(s2)), nll=float(nll), coef=coef)


def fit_h1(X, dt, harmonics=0):
    """Ajuste sous H1 (gamma >= 0). X : (n, 2, 2)."""
    n = X.shape[0]
    t = np.arange(n) * dt
    Phi, (U, V) = design(t, harmonics), _uv(X)
    h0 = fit_h0(X, dt, harmonics)
    f = lambda p: profile_nll(p[0], p[1], U, V, Phi, dt)[0]
    bounds = [(1e-3, 20.0), (0.0, 20.0)]
    cands = [minimize(f, [h0["theta"], g0], method="L-BFGS-B", bounds=bounds) for g0 in (1e-3, 0.05, 0.3)]
    best = min(cands, key=lambda r: r.fun)
    theta, gamma = best.x
    nll, s2, coef = profile_nll(theta, gamma, U, V, Phi, dt)
    if nll > h0["nll"]:                      # H1 contient H0 : on ne peut pas faire pire
        return dict(h0)
    return dict(theta=float(theta), gamma=float(gamma), sigma=float(np.sqrt(s2)), nll=float(nll), coef=coef)


def lrt_interaction(X, dt, harmonics=0):
    """Test du rapport de vraisemblance de H0 : gamma = 0 contre H1 : gamma > 0.

    gamma = 0 est au bord de l'espace des paramètres : sous H0 la statistique suit asymptotiquement
    le mélange 0.5 * chi2(0) + 0.5 * chi2(1) (Chernoff, 1954), d'où la p-valeur 0.5 * P(chi2(1) > LR).
    """
    h0, h1 = fit_h0(X, dt, harmonics), fit_h1(X, dt, harmonics)
    lr = max(0.0, 2.0 * (h0["nll"] - h1["nll"]))
    p = 0.5 * chi2.sf(lr, 1) if lr > 0 else 1.0
    return dict(lr=lr, p=float(p), theta=h1["theta"], gamma=h1["gamma"], sigma=h1["sigma"])
