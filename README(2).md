# Inférer des interactions entre animaux avec des processus d'Ornstein–Uhlenbeck

Projet personnel (Python) : détecter une **interaction** entre deux animaux à partir de leurs trajectoires GPS,
lorsque les deux animaux partagent un **point d'attraction commun** (par exemple un lieu de nourrissage) et un
**rythme nycthéméral** commun. Les indices classiques de la littérature écologique sont comparés à une approche
par **vraisemblance** fondée sur le processus d'Ornstein–Uhlenbeck (OU), un processus gaussien.

> Toutes les données sont **simulées**. Ce projet ne prétend pas reproduire une étude sur données réelles ;
> c'est une étude de simulation sur la robustesse de méthodes d'inférence.

## Résultat principal

Quand les animaux sont **indépendants** mais partagent un attracteur commun, les tests classiques par permutation
détectent à tort une interaction, de façon massive dès qu'un rythme de 24 h est présent. Le test du rapport de
vraisemblance OU qui modélise ce rythme garde un niveau proche de 5 %.

![Taux de rejet](results/fig_taux_de_rejet.png)

Taux de rejet de H0 (« pas d'interaction ») au seuil de 5 %, sur 200 réplicats par scénario
(erreur de Monte-Carlo ≤ 3,5 points) :

| Méthode | A. Indép., attracteur constant | B. Indép., rythme 24 h | C. Interaction, attracteur constant | D. Interaction, rythme 24 h |
|---|---|---|---|---|
| Corrélation (permutation) | 34,5 % | 100 % | 93,5 % | 100 % |
| Prox (permutation) | 19,5 % | 100 % | 59,5 % | 100 % |
| Cs de Kenward (permutation) | 32,5 % | 100 % | 91,5 % | 100 % |
| OU – LRT, moyenne constante | 4,5 % | 100 % | 60,5 % | 100 % |
| **OU – LRT, moyenne + rythme 24 h** | **4,0 %** | **5,0 %** | 51,5 % | 60,0 % |

Lecture : les colonnes A et B mesurent le **niveau** (fausses détections, à comparer à 5 %), les colonnes C et D
la **puissance**. Les puissances élevées des méthodes classiques ne sont pas comparables : ces méthodes rejettent
aussi H0 quand elle est vraie, donc leur rejet n'est pas informatif.

Estimation de l'intensité d'interaction γ avec le modèle complet (200 réplicats) :

| Scénario | γ vrai | γ̂ moyen | écart-type |
|---|---|---|---|
| A et B (indépendants) | 0 | 0,006 / 0,005 | 0,009 / 0,011 |
| C et D (interaction) | 0,04 | 0,035 / 0,037 | 0,019 / 0,020 |

L'estimateur est légèrement biaisé vers le bas pour γ ; θ et σ sont bien retrouvés (θ̂ ≈ 0,20–0,21 pour θ = 0,2 ;
σ̂ ≈ 0,20 pour σ = 0,2).

## Modèle

Pour deux animaux $i = 1, 2$ et chaque coordonnée (x, y), on simule

$$dX_1 = \theta\,(m(t) - X_1)\,dt + \gamma\,(X_2 - X_1)\,dt + \sigma\,dW_1,\qquad
dX_2 = \theta\,(m(t) - X_2)\,dt + \gamma\,(X_1 - X_2)\,dt + \sigma\,dW_2 .$$

- $\theta$ : rappel vers l'attracteur commun $m(t)$ ;
- $\gamma \ge 0$ : attraction mutuelle ; **$\gamma = 0$ signifie animaux indépendants** ;
- $\sigma$ : intensité du bruit ;
- $m(t)$ : constant (point de nourrissage) ou périodique de période 24 h (rythme nycthéméral).

La simulation utilise un schéma d'Euler–Maruyama au pas de 3 minutes, sous-échantillonné toutes les 30 minutes
(14 jours, 672 fixes). Elle est donc indépendante des formules exactes de l'inférence.

## Inférence par vraisemblance

Le point clé est un changement de variables orthogonal :

$$u = \frac{X_1 + X_2}{\sqrt 2}\ \text{(mouvement commun)},\qquad v = \frac{X_1 - X_2}{\sqrt 2}\ \text{(écart entre animaux)}.$$

Avec un bruit isotrope, $u$ et $v$ sont **deux OU indépendants** : $u$ a pour vitesse de rappel $\theta$ et subit
l'attracteur commun $m(t)$, tandis que $v$ a pour vitesse de rappel $\theta + 2\gamma$ et aucun forçage. Une
interaction se traduit donc par un écart entre ces deux vitesses, que l'attracteur commun ne peut pas produire.

La discrétisation exacte au pas $\Delta$ donne deux autorégressions gaussiennes, avec
$a_u = e^{-\theta\Delta}$ et $a_v = e^{-(\theta + 2\gamma)\Delta}$ :

$$v_{k+1} = a_v\, v_k + \varepsilon^v_k,\qquad
u_{k+1} = a_u\, u_k + \sqrt 2\Big[m_{k+1} - a_u m_k - (m_{k+1}-m_k)\tfrac{1-a_u}{\theta\Delta}\Big] + \varepsilon^u_k .$$

Les variances des innovations valent $\sigma^2(1-a_u^2)/(2\theta)$ et $\sigma^2(1-a_v^2)/(2(\theta+2\gamma))$.
La moyenne $m(t)$ (constante, ou constante + cos/sin de période 24 h) entre linéairement : ses coefficients sont
profilés par moindres carrés et $\sigma$ analytiquement. Il reste à maximiser en $(\theta, \gamma)$.

**Test.** On teste $H_0: \gamma = 0$ contre $H_1: \gamma > 0$ par rapport de vraisemblance. Comme $\gamma = 0$ est au
bord de l'espace des paramètres, la statistique suit asymptotiquement le mélange
$\tfrac12\chi^2_0 + \tfrac12\chi^2_1$ (Chernoff, 1954), d'où la p-valeur $\tfrac12\,\mathbb P(\chi^2_1 > \mathrm{LR})$.

## Méthodes classiques comparées

Réimplémentations simplifiées, **inspirées de Long et al. (2014) et du package R `wildlifeDI`** (le package n'est
pas utilisé ici). Chaque indice est testé par permutation des instants de l'animal 2 (199 permutations, test
unilatéral) :

- **Corrélation** de Pearson entre les positions, moyennée sur x et y ;
- **Prox** : proportion de fixes simultanés à moins de 300 m ;
- **Cs** (coefficient de sociabilité de Kenward) : $(d_{\text{att}} - d_{\text{obs}})/(d_{\text{att}} + d_{\text{obs}})$.

Ces tests de permutation supposent des observations échangeables. Or des trajectoires autocorrélées ne le sont
pas, et un rythme commun crée une dépendance supplémentaire : c'est l'origine des fausses détections observées.

## Structure du dépôt

```
ouint/simulate.py     simulation de deux trajectoires OU 2D (attracteur constant ou rythmé)
ouint/classical.py    Corrélation, Prox, Cs et tests de permutation
ouint/inference.py    vraisemblance exacte, estimation (theta, gamma, sigma), test LRT
scripts/run_experiment.py   étude de Monte-Carlo, tableaux et figures
tests/test_basic.py   récupération des paramètres et calibration sous H0
results/              résultats (CSV) et figures de l'expérience décrite ci-dessus
```

## Reproduire

```bash
pip install -r requirements.txt
python tests/test_basic.py                          # tests rapides
python scripts/run_experiment.py --n-rep 200 --seed 2026   # ~1,5 min
```

Paramètres de l'expérience : $\theta = 0{,}2\ \mathrm{h}^{-1}$, $\sigma = 0{,}2\ \mathrm{km}\cdot\mathrm{h}^{-1/2}$,
$\gamma = 0{,}04\ \mathrm{h}^{-1}$ (scénarios C et D), rayon du rythme 0,8 km, fixe toutes les 30 min pendant 14 jours.

## Limites

- Données simulées uniquement ; aucune validation sur télémétrie réelle.
- Le modèle d'estimation est correctement spécifié (OU, bruit isotrope, mêmes θ et σ pour les deux animaux), et la
  période du rythme (24 h) est connue. Un rythme de forme différente ou des paramètres propres à chaque animal
  seraient à étudier.
- Seule une attraction ($\gamma \ge 0$) est modélisée ; la répulsion et les interactions asymétriques ne le sont pas.
- La puissance du test OU est modérée (≈ 50–60 % pour γ = 0,04) : détecter une interaction faible demande des
  séries plus longues.
- La loi de la statistique du rapport de vraisemblance est asymptotique ; un bootstrap paramétrique serait plus
  rigoureux à petit nombre de fixes.
- Pistes : confinement spatial, observations manquantes ou bruitées, plus de deux animaux, processus gaussiens plus
  généraux (noyau de Matérn), comparaison avec le package `wildlifeDI` sur les mêmes données.

## Références

- Long, J. A., Nelson, T. A., Webb, S. L., Gee, K. L. (2014). A critical examination of indices of dynamic
  interaction for wildlife telemetry studies. *Journal of Animal Ecology*, 83(5), 1216–1233.
- Roberts, S., Osborne, M., Ebden, M., Reece, S., Gibson, N., Aigrain, S. (2013). Gaussian processes for time-series
  modelling. *Phil. Trans. R. Soc. A*, 371(1984), 20110550.
- Chernoff, H. (1954). On the distribution of the likelihood ratio. *Annals of Mathematical Statistics*, 25(3), 573–578.

## Auteur

Combete Lantey – Master 2 Mathématiques des données, Institut Galilée, Université Sorbonne Paris Nord.
