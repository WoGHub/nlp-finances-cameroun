"""
conformite_snd30.py
===================
Phase 4 — Analyse Statistique de Conformité SND30
Loi de Finances 2024-2025 — Cameroun

Questions :
  - Est-ce que le pilier qui occupe le plus de programmes reçoit aussi
    le plus de crédits ? (alignement discours/budget)
  - Cet alignement est-il statistiquement significatif ?

Méthodes :
  1. Fréquences thématiques vs montants CP (tableaux croisés)
  2. Corrélation de Spearman (fréquence vs CP par pilier)
  3. Test du Chi-2 (distribution observée vs distribution uniforme)
  4. Clustering K-Means sur embeddings
  5. Clustering HDBSCAN sur embeddings

Produit :
  reports/conformite_snd30.html
  reports/conformite_snd30_rapport.txt
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR   = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CLASSIF = PROCESSED_DIR / "classification_snd30_v2.json"

PILIERS = [
    "Transformation Structurelle",
    "Capital Humain",
    "Gouvernance",
    "Développement Régional",
]

COULEURS = {
    "Transformation Structurelle": "#F4A261",
    "Capital Humain":              "#2DC653",
    "Gouvernance":                 "#457B9D",
    "Développement Régional":      "#9B5DE5",
}

# Dépenses incompressibles à isoler (codes programmes)
CODES_INCOMPRESSIBLES = {
    "199",  # Dette publique extérieure
    "203",  # Dette publique intérieure
    "200",  # Pensions
    "201",  # Dépenses communes de fonctionnement
    "202",  # Subventions et contributions
    "195",  # Interventions en investissement
    "196",  # Réhabilitation entreprises
    "197",  # Reports de crédits
    "198",  # Participations
}

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────

def charger_resultats() -> tuple[list, list]:
    """
    Retourne (tous_programmes, programmes_discretionnaires).
    Les discrétionnaires excluent les dépenses incompressibles.
    """
    with open(SOURCE_CLASSIF, encoding="utf-8") as f:
        data = json.load(f)

    tous = data
    discret = [r for r in tous if r.get("code", "") not in CODES_INCOMPRESSIBLES]

    print(f"  Total programmes     : {len(tous)}")
    print(f"  Discrétionnaires     : {len(discret)}")
    print(f"  Incompressibles      : {len(tous) - len(discret)}")
    return tous, discret


# ─────────────────────────────────────────────────────────────────
# 1. FRÉQUENCES ET MONTANTS PAR PILIER
# ─────────────────────────────────────────────────────────────────

def calculer_stats_piliers(programmes: list) -> dict:
    """
    Pour chaque pilier : nombre de programmes, CP total, CP moyen.
    Retourne aussi les vecteurs fréquence et CP pour les tests.
    """
    stats_dict = {p: {"n": 0, "cp": 0.0} for p in PILIERS}

    for r in programmes:
        pilier = r.get("pilier")
        cp     = r.get("montant_cp") or 0.0
        if pilier in stats_dict:
            stats_dict[pilier]["n"]  += 1
            stats_dict[pilier]["cp"] += cp

    total_n  = sum(v["n"]  for v in stats_dict.values())
    total_cp = sum(v["cp"] for v in stats_dict.values())

    for pilier in PILIERS:
        n  = stats_dict[pilier]["n"]
        cp = stats_dict[pilier]["cp"]
        stats_dict[pilier]["freq_pct"] = round(100 * n  / total_n,  2) if total_n  else 0
        stats_dict[pilier]["cp_pct"]   = round(100 * cp / total_cp, 2) if total_cp else 0
        stats_dict[pilier]["cp_moy"]   = round(cp / n, 0) if n else 0
        stats_dict[pilier]["ecart"]    = round(
            stats_dict[pilier]["cp_pct"] - stats_dict[pilier]["freq_pct"], 2
        )

    return stats_dict, total_n, total_cp


# ─────────────────────────────────────────────────────────────────
# 2. CORRÉLATION DE SPEARMAN
# ─────────────────────────────────────────────────────────────────

def test_spearman(stats_dict: dict) -> dict:
    """
    Corrélation de Spearman entre fréquence (% programmes) et CP (% budget).
    H0 : pas de corrélation entre discours et budget.
    """
    freq = [stats_dict[p]["freq_pct"] for p in PILIERS]
    cp   = [stats_dict[p]["cp_pct"]   for p in PILIERS]

    rho, pval = stats.spearmanr(freq, cp)

    return {
        "rho":         round(rho, 4),
        "pval":        round(pval, 4),
        "significatif": pval < 0.05,
        "freq":        freq,
        "cp":          cp,
        "interpretation": (
            "Corrélation positive forte → alignement discours/budget"
            if rho > 0.7 else
            "Corrélation modérée → alignement partiel"
            if rho > 0.3 else
            "Pas de corrélation → désalignement discours/budget"
        )
    }


# ─────────────────────────────────────────────────────────────────
# 3. TEST DU CHI-2
# ─────────────────────────────────────────────────────────────────

def test_chi2(stats_dict: dict, total_n: int) -> dict:
    """
    Test Chi-2 : la distribution observée des programmes par pilier
    diffère-t-elle significativement d'une distribution uniforme ?
    H0 : distribution uniforme entre les 4 piliers.
    """
    observes  = [stats_dict[p]["n"] for p in PILIERS]
    attendus  = [total_n / len(PILIERS)] * len(PILIERS)

    chi2, pval = stats.chisquare(observes, attendus)

    # Chi-2 sur CP
    total_cp  = sum(stats_dict[p]["cp"] for p in PILIERS)
    obs_cp    = [stats_dict[p]["cp"] for p in PILIERS]
    att_cp    = [total_cp / len(PILIERS)] * len(PILIERS)
    chi2_cp, pval_cp = stats.chisquare(obs_cp, att_cp)

    return {
        "chi2_n":     round(chi2, 4),
        "pval_n":     round(pval, 4),
        "chi2_cp":    round(chi2_cp, 4),
        "pval_cp":    round(pval_cp, 4),
        "sig_n":      pval < 0.05,
        "sig_cp":     pval_cp < 0.05,
        "observes":   observes,
        "attendus":   [round(a, 1) for a in attendus],
    }


# ─────────────────────────────────────────────────────────────────
# 4. CLUSTERING K-MEANS
# ─────────────────────────────────────────────────────────────────

def clustering_kmeans(programmes: list, n_clusters: int = 4) -> dict:
    """
    K-Means sur les vecteurs [freq_norm, cp_norm, score_dominant, ecart_confiance]
    pour chaque programme.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = np.array([
        [
            r.get("score_dominant", 0),
            r.get("ecart_confiance", 0),
            r.get("montant_cp", 0) or 0,
        ]
        for r in programmes
    ])

    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Inertie par k pour courbe du coude
    inerties = []
    for k in range(2, 8):
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_k.fit(X_scaled)
        inerties.append(km_k.inertia_)

    # Caractériser chaque cluster
    clusters = {}
    for k in range(n_clusters):
        indices = [i for i, l in enumerate(labels) if l == k]
        progs_k = [programmes[i] for i in indices]
        piliers_k = Counter(r["pilier"] for r in progs_k)
        cp_moy_k  = np.mean([r.get("montant_cp", 0) or 0 for r in progs_k])
        score_moy = np.mean([r.get("score_dominant", 0) for r in progs_k])
        clusters[k] = {
            "n":           len(progs_k),
            "piliers":     dict(piliers_k.most_common()),
            "cp_moyen":    round(cp_moy_k, 0),
            "score_moyen": round(score_moy, 4),
            "pilier_dom":  piliers_k.most_common(1)[0][0] if piliers_k else "?",
        }

    return {
        "labels":    labels.tolist(),
        "clusters":  clusters,
        "inerties":  inerties,
        "X_scaled":  X_scaled,
    }


# ─────────────────────────────────────────────────────────────────
# 5. CLUSTERING HDBSCAN
# ─────────────────────────────────────────────────────────────────

def clustering_hdbscan(programmes: list) -> dict:
    """
    HDBSCAN sur les mêmes features — détecte automatiquement le nombre
    de clusters et identifie les outliers (label = -1).
    """
    try:
        import hdbscan
    except ImportError:
        print("  [WARN] hdbscan non installé → pip install hdbscan")
        return None

    from sklearn.preprocessing import StandardScaler

    X = np.array([
        [
            r.get("score_dominant", 0),
            r.get("ecart_confiance", 0),
            r.get("montant_cp", 0) or 0,
        ]
        for r in programmes
    ])
    X_scaled = StandardScaler().fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()

    clusters = {}
    for k in sorted(set(labels)):
        indices = [i for i, l in enumerate(labels) if l == k]
        progs_k = [programmes[i] for i in indices]
        piliers_k = Counter(r["pilier"] for r in progs_k)
        cp_moy_k  = np.mean([r.get("montant_cp", 0) or 0 for r in progs_k])
        clusters[int(k)] = {
            "n":        len(progs_k),
            "piliers":  dict(piliers_k.most_common()),
            "cp_moyen": round(cp_moy_k, 0),
            "label":    "Outliers" if k == -1 else f"Cluster {k}",
        }

    return {
        "labels":     labels.tolist(),
        "n_clusters": n_clusters,
        "n_outliers": int(n_outliers),
        "clusters":   clusters,
        "X_scaled":   X_scaled,
    }


# ─────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────

def plot_freq_vs_cp(stats_dict: dict, titre: str) -> go.Figure:
    """Barres groupées : fréquence % vs CP % par pilier."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Fréquence (% programmes)",
        x=PILIERS,
        y=[stats_dict[p]["freq_pct"] for p in PILIERS],
        marker_color=[COULEURS[p] for p in PILIERS],
        opacity=0.9,
        text=[f"{stats_dict[p]['freq_pct']:.1f}%" for p in PILIERS],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        name="CP (% budget)",
        x=PILIERS,
        y=[stats_dict[p]["cp_pct"] for p in PILIERS],
        marker_color=[COULEURS[p] for p in PILIERS],
        opacity=0.45,
        marker_pattern_shape="/",
        text=[f"{stats_dict[p]['cp_pct']:.1f}%" for p in PILIERS],
        textposition="outside",
    ))

    fig.update_layout(
        title=titre,
        barmode="group",
        yaxis_title="Pourcentage (%)",
        xaxis=dict(tickangle=-15),
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", family="Georgia"),
        legend=dict(bgcolor="rgba(255,255,255,0.07)"),
        height=460,
    )
    return fig


def plot_ecart_alignement(stats_dict: dict, titre: str) -> go.Figure:
    """Barres horizontales : écart CP% - Freq% (sur/sous-financement)."""
    ecarts  = [stats_dict[p]["ecart"] for p in PILIERS]
    couleurs_ecart = ["#2DC653" if e >= 0 else "#E63946" for e in ecarts]

    fig = go.Figure(go.Bar(
        x=ecarts,
        y=PILIERS,
        orientation="h",
        marker_color=couleurs_ecart,
        text=[f"{e:+.1f}%" for e in ecarts],
        textposition="outside",
    ))

    fig.add_vline(x=0, line_color="white", line_width=1)

    fig.update_layout(
        title=titre,
        xaxis_title="Écart = CP% − Fréquence% (positif = sur-financé, négatif = sous-financé)",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", family="Georgia"),
        height=360,
        margin=dict(l=220),
    )
    return fig


def plot_spearman(res_sp: dict) -> go.Figure:
    """Scatter fréquence vs CP avec droite de régression."""
    freq = res_sp["freq"]
    cp   = res_sp["cp"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freq, y=cp,
        mode="markers+text",
        text=PILIERS,
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            color=[COULEURS[p] for p in PILIERS],
            size=16,
            line=dict(width=1, color="white"),
        ),
        hovertemplate="<b>%{text}</b><br>Fréquence : %{x:.1f}%<br>CP : %{y:.1f}%<extra></extra>",
    ))

    # Droite de régression
    m, b = np.polyfit(freq, cp, 1)
    x_line = np.linspace(min(freq) - 2, max(freq) + 2, 50)
    fig.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b,
        mode="lines",
        line=dict(dash="dash", color="rgba(255,255,255,0.4)", width=1),
        name="Régression",
        showlegend=False,
    ))

    fig.add_shape(type="line", x0=0, x1=max(freq)+5, y0=0, y1=max(freq)+5,
                  line=dict(dash="dot", color="rgba(255,255,255,0.2)", width=1))

    fig.update_layout(
        title=f"Corrélation Spearman : Fréquence vs CP — ρ={res_sp['rho']:.3f}  p={res_sp['pval']:.3f}",
        xaxis_title="Fréquence (% programmes)",
        yaxis_title="CP (% budget)",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", family="Georgia"),
        height=450,
        annotations=[dict(
            x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"ρ = {res_sp['rho']:.3f}<br>p = {res_sp['pval']:.3f}<br>"
                 f"{'✅ Significatif' if res_sp['significatif'] else '❌ Non significatif'}<br>"
                 f"{res_sp['interpretation']}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.08)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(size=11),
            align="left",
        )]
    )
    return fig


def plot_kmeans(km_res: dict, programmes: list) -> go.Figure:
    """Scatter des programmes colorés par cluster K-Means."""
    labels   = km_res["labels"]
    X_scaled = km_res["X_scaled"]

    PALETTE_CLUSTERS = ["#E63946", "#F4A261", "#2DC653", "#457B9D",
                        "#9B5DE5", "#00BBF9", "#FEE440"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Clusters K-Means (score vs écart)", "Courbe du coude"],
    )

    # Scatter clusters
    for k in range(max(labels) + 1):
        indices = [i for i, l in enumerate(labels) if l == k]
        info = km_res["clusters"][k]
        fig.add_trace(go.Scatter(
            x=[X_scaled[i, 0] for i in indices],
            y=[X_scaled[i, 1] for i in indices],
            mode="markers",
            name=f"Cluster {k} ({info['pilier_dom'][:20]})",
            marker=dict(color=PALETTE_CLUSTERS[k % len(PALETTE_CLUSTERS)], size=8, opacity=0.8),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pilier : %{customdata[1]}<extra></extra>"
            ),
            customdata=[[programmes[i]["libelle"][:40], programmes[i]["pilier"]]
                        for i in indices],
        ), row=1, col=1)

    # Courbe du coude
    fig.add_trace(go.Scatter(
        x=list(range(2, 2 + len(km_res["inerties"]))),
        y=km_res["inerties"],
        mode="lines+markers",
        line=dict(color="#F4A261"),
        marker=dict(size=8),
        name="Inertie",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title="Clustering K-Means des programmes budgétaires",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", family="Georgia"),
        height=460,
        legend=dict(bgcolor="rgba(255,255,255,0.05)", font=dict(size=9)),
    )
    fig.update_xaxes(title_text="Score (normalisé)", row=1, col=1)
    fig.update_yaxes(title_text="Écart confiance (normalisé)", row=1, col=1)
    fig.update_xaxes(title_text="Nombre de clusters k", row=1, col=2)
    fig.update_yaxes(title_text="Inertie", row=1, col=2)
    return fig


def plot_hdbscan(hdb_res: dict, programmes: list) -> go.Figure:
    """Scatter des programmes colorés par cluster HDBSCAN."""
    labels   = hdb_res["labels"]
    X_scaled = hdb_res["X_scaled"]

    PALETTE = ["#E63946", "#F4A261", "#2DC653", "#457B9D",
               "#9B5DE5", "#00BBF9", "#FEE440", "#FB5607"]

    fig = go.Figure()

    unique_labels = sorted(set(labels))
    for k in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == k]
        nom = "Outliers" if k == -1 else f"Cluster {k}"
        couleur = "rgba(150,150,150,0.3)" if k == -1 else PALETTE[k % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=[X_scaled[i, 0] for i in indices],
            y=[X_scaled[i, 1] for i in indices],
            mode="markers",
            name=nom,
            marker=dict(color=couleur, size=7 if k == -1 else 9, opacity=0.8),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pilier : %{customdata[1]}<extra></extra>"
            ),
            customdata=[[programmes[i]["libelle"][:40], programmes[i]["pilier"]]
                        for i in indices],
        ))

    fig.update_layout(
        title=(f"Clustering HDBSCAN — {hdb_res['n_clusters']} clusters détectés "
               f"({hdb_res['n_outliers']} outliers)"),
        xaxis_title="Score dominant (normalisé)",
        yaxis_title="Écart confiance (normalisé)",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", family="Georgia"),
        legend=dict(bgcolor="rgba(255,255,255,0.05)"),
        height=480,
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# RAPPORT TEXTE
# ─────────────────────────────────────────────────────────────────

def generer_rapport(stats_tot, stats_dis, res_sp_tot, res_sp_dis,
                    res_chi2_tot, res_chi2_dis, km_res, hdb_res) -> str:
    L = []
    L.append("=" * 70)
    L.append("  RAPPORT — ANALYSE STATISTIQUE DE CONFORMITÉ SND30")
    L.append("  Loi de Finances 2024-2025 — Cameroun")
    L.append("=" * 70)

    for label, sd, rs, rc in [
        ("BUDGET TOTAL (182 programmes)", stats_tot, res_sp_tot, res_chi2_tot),
        ("BUDGET DISCRÉTIONNAIRE (hors incompressibles)", stats_dis, res_sp_dis, res_chi2_dis),
    ]:
        L.append(f"\n{'─'*70}")
        L.append(f"  {label}")
        L.append(f"{'─'*70}")

        L.append(f"\n  Fréquence vs CP par pilier :")
        L.append(f"  {'Pilier':<35} {'Freq%':>7} {'CP%':>7} {'Écart':>7}")
        L.append(f"  {'-'*56}")
        for p in PILIERS:
            L.append(f"  {p:<35} {sd[p]['freq_pct']:>7.1f} {sd[p]['cp_pct']:>7.1f} "
                     f"{sd[p]['ecart']:>+7.1f}")

        L.append(f"\n  Corrélation de Spearman :")
        L.append(f"    ρ = {rs['rho']:.4f}   p-value = {rs['pval']:.4f}")
        L.append(f"    {'✅ Significatif (p < 0.05)' if rs['significatif'] else '❌ Non significatif (p ≥ 0.05)'}")
        L.append(f"    → {rs['interpretation']}")

        L.append(f"\n  Test Chi-2 (distribution programmes) :")
        L.append(f"    χ² = {rc['chi2_n']:.4f}   p-value = {rc['pval_n']:.4f}")
        L.append(f"    {'✅ Distribution non uniforme' if rc['sig_n'] else '❌ Distribution non significativement différente de l uniforme'}")
        L.append(f"\n  Test Chi-2 (distribution CP) :")
        L.append(f"    χ² = {rc['chi2_cp']:.4f}   p-value = {rc['pval_cp']:.4f}")
        L.append(f"    {'✅ Distribution CP non uniforme' if rc['sig_cp'] else '❌ Distribution CP non significativement différente de l uniforme'}")

    if km_res:
        L.append(f"\n{'─'*70}")
        L.append(f"  CLUSTERING K-MEANS (k=4)")
        L.append(f"{'─'*70}")
        for k, info in km_res["clusters"].items():
            L.append(f"  Cluster {k} ({info['n']} programmes) — dominant : {info['pilier_dom']}")
            L.append(f"    CP moyen : {info['cp_moyen']/1e6:,.0f} M FCFA  |  Score moyen : {info['score_moyen']:.3f}")

    if hdb_res:
        L.append(f"\n{'─'*70}")
        L.append(f"  CLUSTERING HDBSCAN")
        L.append(f"{'─'*70}")
        L.append(f"  {hdb_res['n_clusters']} clusters détectés  |  {hdb_res['n_outliers']} outliers")
        for k, info in hdb_res["clusters"].items():
            L.append(f"  {info['label']} ({info['n']} programmes) — CP moyen : {info['cp_moyen']/1e6:,.0f} M FCFA")

    L.append(f"\n{'='*70}\n")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  ANALYSE STATISTIQUE DE CONFORMITÉ SND30 — Phase 4")
    print("="*60)

    print("\n[1/5] Chargement des données...")
    tous, discret = charger_resultats()

    print("\n[2/5] Statistiques fréquences / CP...")
    stats_tot, n_tot, cp_tot = calculer_stats_piliers(tous)
    stats_dis, n_dis, cp_dis = calculer_stats_piliers(discret)

    print("\n[3/5] Tests statistiques (Spearman + Chi-2)...")
    res_sp_tot  = test_spearman(stats_tot)
    res_sp_dis  = test_spearman(stats_dis)
    res_chi2_tot = test_chi2(stats_tot, n_tot)
    res_chi2_dis = test_chi2(stats_dis, n_dis)

    print(f"  Spearman total        : ρ={res_sp_tot['rho']}  p={res_sp_tot['pval']}")
    print(f"  Spearman discrét.     : ρ={res_sp_dis['rho']}  p={res_sp_dis['pval']}")

    print("\n[4/5] Clustering K-Means...")
    km_res  = clustering_kmeans(discret, n_clusters=4)

    print("       Clustering HDBSCAN...")
    hdb_res = clustering_hdbscan(discret)

    print("\n[5/5] Génération des visualisations...")
    figs = [
        plot_freq_vs_cp(stats_tot, "Fréquence vs CP — Budget total"),
        plot_ecart_alignement(stats_tot, "Écart alignement discours/budget — Total"),
        plot_freq_vs_cp(stats_dis, "Fréquence vs CP — Budget discrétionnaire"),
        plot_ecart_alignement(stats_dis, "Écart alignement discours/budget — Discrétionnaire"),
        plot_spearman(res_sp_dis),
        plot_kmeans(km_res, discret),
    ]
    if hdb_res:
        figs.append(plot_hdbscan(hdb_res, discret))

    html_parts = [
        "<h1 style='font-family:Georgia;color:#F0EAD6;background:#0F0F1A;"
        "padding:20px 30px;margin:0'>"
        "Analyse de Conformité SND30 — Loi de Finances 2024-2025</h1>"
    ]
    for i, fig in enumerate(figs):
        html_parts.append(to_html(
            fig,
            include_plotlyjs="cdn" if i == 0 else False,
            full_html=False,
        ))

    html_final = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{background:#0F0F1A;margin:0;padding:10px}</style>"
        "</head><body>" + "".join(html_parts) + "</body></html>"
    )

    sortie_html = REPORTS_DIR / "conformite_snd30.html"
    with open(sortie_html, "w", encoding="utf-8") as f:
        f.write(html_final)
    print(f"  HTML : {sortie_html.name}")

    rapport = generer_rapport(stats_tot, stats_dis, res_sp_tot, res_sp_dis,
                              res_chi2_tot, res_chi2_dis, km_res, hdb_res)
    sortie_txt = REPORTS_DIR / "conformite_snd30_rapport.txt"
    with open(sortie_txt, "w", encoding="utf-8") as f:
        f.write(rapport)
    print(f"  Rapport : {sortie_txt.name}")
    print("\n" + rapport)


if __name__ == "__main__":
    main()