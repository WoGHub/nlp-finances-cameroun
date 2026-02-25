"""
semantic_audit.py
=================
Audit sémantique des Lois de Finances 2023-2024 vs 2024-2025.

Produit :
  - data/processed/audit_correspondances.json   → tableau complet des correspondances
  - reports/audit_semantique_camembert.html      → visualisation interactive Plotly
  - reports/audit_semantique_minilm.html         → visualisation interactive Plotly
  - reports/rapport_audit.txt                    → rapport texte synthétique

Analyse :
  1. Correspondances article-à-article (best match bidirectionnel)
  2. Classification : Stable / Modifié / Nouveau / Supprimé
  3. Distribution des similarités
  4. Heatmap inter-titres
  5. Visualisation UMAP des deux lois superposées
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR   = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Seuils de classification
SEUIL_STABLE   = 0.90   # article quasi identique
SEUIL_MODIFIE  = 0.70   # article modifié (contenu partiellement changé)
# < SEUIL_MODIFIE → article très différent / nouveau / supprimé

MODELES = ["camembert", "minilm"]

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────

def charger_embeddings(nom_modele: str) -> dict:
    chemin = PROCESSED_DIR / f"embeddings_{nom_modele}.npz"
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}\n→ Lancez compute_embeddings.py")
    data = np.load(chemin, allow_pickle=True)
    return {
        "emb_2023":    data["embeddings_2023"],
        "emb_2024":    data["embeddings_2024"],
        "ids_2023":    data["ids_2023"].tolist(),
        "ids_2024":    data["ids_2024"].tolist(),
        "textes_2023": data["textes_2023"].tolist(),
        "textes_2024": data["textes_2024"].tolist(),
    }


def charger_articles(annee: str) -> list:
    chemin = PROCESSED_DIR / f"loi_{annee.replace('-', '_')}_articles.json"
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────
# CORRESPONDANCES
# ─────────────────────────────────────────────────────────────────

def calculer_correspondances(data: dict) -> list:
    """
    Pour chaque article de 2023, trouve son meilleur correspondant en 2024
    et le classe selon le seuil de similarité.
    """
    emb_2023 = data["emb_2023"]
    emb_2024 = data["emb_2024"]

    # Matrice N1 x N2
    sim_matrix = emb_2023 @ emb_2024.T

    correspondances = []
    for i, ref_2023 in enumerate(data["ids_2023"]):
        scores = sim_matrix[i]
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= SEUIL_STABLE:
            statut = "STABLE"
        elif best_score >= SEUIL_MODIFIE:
            statut = "MODIFIÉ"
        else:
            statut = "TRÈS MODIFIÉ / NOUVEAU"

        correspondances.append({
            "ref_2023":        ref_2023,
            "texte_2023":      data["textes_2023"][i][:200],
            "ref_2024":        data["ids_2024"][best_idx],
            "texte_2024":      data["textes_2024"][best_idx][:200],
            "score":           round(best_score, 4),
            "statut":          statut,
            "idx_2023":        i,
            "idx_2024":        best_idx,
        })

    # Détecter les articles 2024 sans correspondance forte (nouveaux)
    idx_2024_couverts = {c["idx_2024"] for c in correspondances if c["score"] >= SEUIL_MODIFIE}
    nouveaux = []
    for j, ref_2024 in enumerate(data["ids_2024"]):
        if j not in idx_2024_couverts:
            nouveaux.append({
                "ref_2023":   None,
                "texte_2023": None,
                "ref_2024":   ref_2024,
                "texte_2024": data["textes_2024"][j][:200],
                "score":      0.0,
                "statut":     "NOUVEAU",
                "idx_2023":   None,
                "idx_2024":   j,
            })

    return correspondances + nouveaux


# ─────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────

COULEURS_STATUT = {
    "STABLE":                "#2DC653",
    "MODIFIÉ":               "#F4A261",
    "TRÈS MODIFIÉ / NOUVEAU": "#E63946",
    "NOUVEAU":               "#9B5DE5",
}


def plot_distribution(correspondances: list, nom_modele: str) -> go.Figure:
    """Distribution des scores de similarité avec zones colorées."""
    scores = [c["score"] for c in correspondances if c["score"] > 0]
    statuts = [c["statut"] for c in correspondances if c["score"] > 0]

    fig = go.Figure()

    # Histogramme global
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=40,
        name="Distribution",
        marker_color="#457B9D",
        opacity=0.8,
    ))

    # Zones de seuil
    fig.add_vrect(x0=SEUIL_STABLE, x1=1.0,
                  fillcolor="#2DC653", opacity=0.1,
                  annotation_text="Stable", annotation_position="top left")
    fig.add_vrect(x0=SEUIL_MODIFIE, x1=SEUIL_STABLE,
                  fillcolor="#F4A261", opacity=0.1,
                  annotation_text="Modifié", annotation_position="top left")
    fig.add_vrect(x0=0, x1=SEUIL_MODIFIE,
                  fillcolor="#E63946", opacity=0.1,
                  annotation_text="Très modifié", annotation_position="top left")

    fig.update_layout(
        title=f"Distribution des similarités cosinus — {nom_modele.upper()}",
        xaxis_title="Score de similarité cosinus",
        yaxis_title="Nombre d'articles",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6"),
        height=400,
    )
    return fig


def plot_scatter_correspondances(correspondances: list, nom_modele: str) -> go.Figure:
    """Scatter plot : chaque point = un article 2023, coloré par statut."""
    from collections import defaultdict
    par_statut = defaultdict(list)
    for c in correspondances:
        if c["score"] > 0:
            par_statut[c["statut"]].append(c)

    fig = go.Figure()
    for statut, items in par_statut.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(items))),
            y=[c["score"] for c in items],
            mode="markers",
            name=statut,
            marker=dict(
                color=COULEURS_STATUT.get(statut, "#888"),
                size=8,
                opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "→ %{customdata[1]}<br>"
                "Score : %{y:.4f}<extra></extra>"
            ),
            customdata=[[c["ref_2023"], c["ref_2024"]] for c in items],
        ))

    fig.add_hline(y=SEUIL_STABLE,  line_dash="dash", line_color="#2DC653",
                  annotation_text=f"Seuil stable ({SEUIL_STABLE})")
    fig.add_hline(y=SEUIL_MODIFIE, line_dash="dash", line_color="#F4A261",
                  annotation_text=f"Seuil modifié ({SEUIL_MODIFIE})")

    fig.update_layout(
        title=f"Scores de correspondance par article — {nom_modele.upper()}",
        xaxis_title="Articles (indexés)",
        yaxis_title="Similarité cosinus",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6"),
        height=450,
        legend=dict(bgcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def plot_heatmap_titres(data: dict, articles_2023: list, articles_2024: list,
                        nom_modele: str) -> go.Figure:
    """Heatmap de similarité moyenne entre titres des deux lois."""
    emb_2023 = data["emb_2023"]
    emb_2024 = data["emb_2024"]

    # Récupérer les titres
    titres_2023 = [a.get("titre") or "Sans titre" for a in articles_2023]
    titres_2024 = [a.get("titre") or "Sans titre" for a in articles_2024]

    unique_titres_2023 = list(dict.fromkeys(titres_2023))[:12]  # max 12
    unique_titres_2024 = list(dict.fromkeys(titres_2024))[:12]

    # Embeddings moyens par titre
    def emb_moyen_titre(embeddings, titres, titre_cible):
        indices = [i for i, t in enumerate(titres) if t == titre_cible]
        if not indices:
            return np.zeros(embeddings.shape[1])
        return embeddings[indices].mean(axis=0)

    emb_moy_2023 = np.array([emb_moyen_titre(emb_2023, titres_2023, t)
                              for t in unique_titres_2023])
    emb_moy_2024 = np.array([emb_moyen_titre(emb_2024, titres_2024, t)
                              for t in unique_titres_2024])

    sim_matrix = cosine_similarity(emb_moy_2023, emb_moy_2024)

    # Raccourcir les labels
    def raccourcir(label, n=40):
        return label[:n] + "..." if len(label) > n else label

    labels_2023 = [raccourcir(t) for t in unique_titres_2023]
    labels_2024 = [raccourcir(t) for t in unique_titres_2024]

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=labels_2024,
        y=labels_2023,
        colorscale="Viridis",
        zmin=0.5, zmax=1.0,
        text=np.round(sim_matrix, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hoverongaps=False,
    ))

    fig.update_layout(
        title=f"Similarité inter-titres — {nom_modele.upper()} (2023-2024 vs 2024-2025)",
        xaxis_title="Titres Loi 2024-2025",
        yaxis_title="Titres Loi 2023-2024",
        xaxis=dict(tickangle=-40),
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", size=10),
        height=520,
    )
    return fig


def plot_umap(data: dict, articles_2023: list, articles_2024: list,
              nom_modele: str) -> go.Figure:
    """UMAP superposé des deux lois coloré par titre."""
    try:
        import umap
    except ImportError:
        print("  [WARN] umap-learn non installé, UMAP ignoré.")
        return None

    from sklearn.preprocessing import StandardScaler

    emb_2023 = data["emb_2023"]
    emb_2024 = data["emb_2024"]

    X_combined = np.vstack([emb_2023, emb_2024])
    X_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     random_state=42).fit_transform(
        StandardScaler().fit_transform(X_combined)
    )

    n1 = len(emb_2023)
    X1_2d = X_2d[:n1]
    X2_2d = X_2d[n1:]

    titres_2023 = [a.get("titre") or "Sans titre" for a in articles_2023]
    titres_2024 = [a.get("titre") or "Sans titre" for a in articles_2024]

    PALETTE = ["#E63946","#457B9D","#2DC653","#F4A261","#9B5DE5",
               "#00BBF9","#F15BB5","#FEE440","#00F5D4","#FB5607",
               "#8338EC","#3A86FF"]
    tous_titres = sorted(set(titres_2023 + titres_2024))
    cmap = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(tous_titres)}

    fig = go.Figure()

    for titre in tous_titres:
        # Points loi 2023 (cercles)
        idx1 = [i for i, t in enumerate(titres_2023) if t == titre]
        if idx1:
            fig.add_trace(go.Scatter(
                x=X1_2d[idx1, 0], y=X1_2d[idx1, 1],
                mode="markers",
                name=f"2023 – {titre[:30]}",
                legendgroup=titre,
                showlegend=True,
                marker=dict(color=cmap[titre], size=7, symbol="circle", opacity=0.75),
                hovertemplate=f"<b>2023-2024</b><br>{titre}<extra></extra>",
            ))
        # Points loi 2024 (carrés)
        idx2 = [i for i, t in enumerate(titres_2024) if t == titre]
        if idx2:
            fig.add_trace(go.Scatter(
                x=X2_2d[idx2, 0], y=X2_2d[idx2, 1],
                mode="markers",
                name=f"2024 – {titre[:30]}",
                legendgroup=titre,
                showlegend=False,
                marker=dict(color=cmap[titre], size=7, symbol="square", opacity=0.75),
                hovertemplate=f"<b>2024-2025</b><br>{titre}<extra></extra>",
            ))

    fig.update_layout(
        title=f"UMAP superposé — {nom_modele.upper()} (○ = 2023-2024 | □ = 2024-2025)",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6"),
        legend=dict(bgcolor="rgba(255,255,255,0.05)", font=dict(size=9)),
        height=600,
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# RAPPORT TEXTE
# ─────────────────────────────────────────────────────────────────

def generer_rapport(correspondances_par_modele: dict) -> str:
    lignes = []
    lignes.append("=" * 70)
    lignes.append("  RAPPORT D'AUDIT SÉMANTIQUE")
    lignes.append("  Loi de Finances 2023-2024 vs 2024-2025 — Cameroun")
    lignes.append("=" * 70)

    for nom_modele, correspondances in correspondances_par_modele.items():
        lignes.append(f"\n{'─'*70}")
        lignes.append(f"  MODÈLE : {nom_modele.upper()}")
        lignes.append(f"{'─'*70}")

        scores = [c["score"] for c in correspondances if c["score"] > 0]
        n_total   = len([c for c in correspondances if c["ref_2023"]])
        n_stable  = sum(1 for c in correspondances if c["statut"] == "STABLE")
        n_modif   = sum(1 for c in correspondances if c["statut"] == "MODIFIÉ")
        n_tres    = sum(1 for c in correspondances if c["statut"] == "TRÈS MODIFIÉ / NOUVEAU")
        n_nouveau = sum(1 for c in correspondances if c["statut"] == "NOUVEAU")

        lignes.append(f"\n  Articles analysés (2023-2024)  : {n_total}")
        lignes.append(f"  Similarité moyenne             : {np.mean(scores):.4f}")
        lignes.append(f"  Similarité médiane             : {np.median(scores):.4f}")
        lignes.append(f"\n  Répartition par statut :")
        lignes.append(f"    ✅ STABLE          (≥ {SEUIL_STABLE}) : {n_stable:3d}  ({100*n_stable/n_total:.1f}%)")
        lignes.append(f"    ⚠️  MODIFIÉ         ({SEUIL_MODIFIE}–{SEUIL_STABLE}) : {n_modif:3d}  ({100*n_modif/n_total:.1f}%)")
        lignes.append(f"    ❌ TRÈS MODIFIÉ    (< {SEUIL_MODIFIE}) : {n_tres:3d}  ({100*n_tres/n_total:.1f}%)")
        lignes.append(f"    🆕 NOUVEAU (2024)               : {n_nouveau:3d}")

        # Top 10 articles les plus modifiés
        tres_modifies = sorted(
            [c for c in correspondances if c["statut"] in ("TRÈS MODIFIÉ / NOUVEAU",)],
            key=lambda x: x["score"]
        )[:10]

        if tres_modifies:
            lignes.append(f"\n  Top articles les plus modifiés/nouveaux :")
            for c in tres_modifies:
                lignes.append(f"    {c['score']:.4f}  {(c['ref_2023'] or 'N/A')[:40]}")
                lignes.append(f"           → {c['ref_2024'][:40]}")

    lignes.append(f"\n{'='*70}\n")
    return "\n".join(lignes)


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  AUDIT SÉMANTIQUE")
    print("="*60)

    articles_2023 = charger_articles("2023-2024")
    articles_2024 = charger_articles("2024-2025")

    correspondances_par_modele = {}
    toutes_figures = {}

    for nom_modele in MODELES:
        print(f"\n[{nom_modele.upper()}] Chargement des embeddings...")
        data = charger_embeddings(nom_modele)

        print(f"[{nom_modele.upper()}] Calcul des correspondances...")
        correspondances = calculer_correspondances(data)
        correspondances_par_modele[nom_modele] = correspondances

        # Stats rapides
        scores = [c["score"] for c in correspondances if c["score"] > 0]
        n_stable = sum(1 for c in correspondances if c["statut"] == "STABLE")
        n_modif  = sum(1 for c in correspondances if c["statut"] == "MODIFIÉ")
        n_tres   = sum(1 for c in correspondances if c["statut"] == "TRÈS MODIFIÉ / NOUVEAU")
        n_nouv   = sum(1 for c in correspondances if c["statut"] == "NOUVEAU")
        print(f"  Stable: {n_stable} | Modifié: {n_modif} | Très modifié: {n_tres} | Nouveau: {n_nouv}")

        # Figures
        print(f"[{nom_modele.upper()}] Génération des visualisations...")
        figs = {}
        figs["distribution"] = plot_distribution(correspondances, nom_modele)
        figs["scatter"]      = plot_scatter_correspondances(correspondances, nom_modele)
        figs["heatmap"]      = plot_heatmap_titres(data, articles_2023, articles_2024, nom_modele)

        umap_fig = plot_umap(data, articles_2023, articles_2024, nom_modele)
        if umap_fig:
            figs["umap"] = umap_fig

        toutes_figures[nom_modele] = figs

        # Assemblage HTML
        from plotly.io import to_html
        html_parts = [f"<h1 style='font-family:Georgia;color:#F0EAD6;background:#0F0F1A;padding:20px'>"
                      f"Audit Sémantique — {nom_modele.upper()}</h1>"]
        for nom_fig, fig in figs.items():
            html_parts.append(to_html(fig, include_plotlyjs="cdn" if nom_fig == list(figs.keys())[0] else False,
                                      full_html=False))

        html_final = f"""<!DOCTYPE html>
<html>
<head><meta charset='utf-8'>
<style>body{{background:#0F0F1A;margin:20px}}</style>
</head>
<body>{''.join(html_parts)}</body>
</html>"""

        sortie_html = REPORTS_DIR / f"audit_semantique_{nom_modele}.html"
        with open(sortie_html, "w", encoding="utf-8") as f:
            f.write(html_final)
        print(f"  Rapport HTML : {sortie_html.name}")

    # Sauvegarde JSON des correspondances
    sortie_json = PROCESSED_DIR / "audit_correspondances.json"
    with open(sortie_json, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in correspondances_par_modele.items()}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n  Correspondances JSON : {sortie_json.name}")

    # Rapport texte
    rapport = generer_rapport(correspondances_par_modele)
    sortie_rapport = REPORTS_DIR / "rapport_audit.txt"
    with open(sortie_rapport, "w", encoding="utf-8") as f:
        f.write(rapport)
    print(f"  Rapport texte : {sortie_rapport.name}")
    print("\n" + rapport)


if __name__ == "__main__":
    main()