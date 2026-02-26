"""
zero_shot_snd30_v2.py
=====================
Classification zero-shot des programmes budgétaires de la Loi de Finances
2024-2025 selon les 4 piliers de la SND30.

Source : loi2024_2025_ligne_budgetaire.json (structure Gemini)
         → 59 chapitres × N programmes = 182 lignes propres

Approche : similarité cosinus entre embeddings MiniLM des programmes
           (libellé + objectif + indicateur) et centroïdes SND30.

Produit :
  data/processed/classification_snd30_v2.json
  reports/classification_snd30_v2.html
  reports/classification_snd30_rapport_v2.txt
"""

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR   = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Chercher le fichier source dans plusieurs emplacements possibles
CANDIDATS_SOURCE = [
    BASE_DIR / "data" / "raw"       / "loi2024_2025_ligne_budgetaire.json",
    BASE_DIR / "data" / "processed" / "loi2024_2025_ligne_budgetaire.json",
    BASE_DIR / "loi2024_2025_ligne_budgetaire.json",
]

MODELE_ID  = "paraphrase-multilingual-MiniLM-L12-v2"
SEUIL_CONF = 0.25   # score minimum pour classification fiable
SEUIL_ECART = 0.02  # écart minimum entre 1er et 2e pilier

# ─────────────────────────────────────────────────────────────────
# PILIERS SND30 — DESCRIPTEURS ENRICHIS
# ─────────────────────────────────────────────────────────────────

PILIERS_SND30 = {
    "Transformation Structurelle": [
        "Développement industriel manufacturier transformation locale matières premières",
        "Infrastructures de transport routier ferroviaire portuaire construction routes",
        "Compétitivité entreprises innovation technologique numérique",
        "Agriculture élevage pêche mines énergie secteurs productifs",
        "Électrification accès énergie ménages industries eau potable",
        "Développement secteur privé climate affaires exportations commerce",
        "Travaux publics BIP investissement programme infrastructure",
        "Réseaux télécommunications postes numérique",
    ],
    "Capital Humain": [
        "Éducation de base enseignement primaire secondaire formation professionnelle",
        "Santé publique couverture sanitaire soins prévention maladie nutrition",
        "Emploi insertion professionnelle jeunes formation compétences",
        "Protection sociale femme famille enfance vulnérabilités",
        "Enseignement supérieur université recherche innovation académique",
        "Sport culture épanouissement population arts",
        "Alphabétisation scolarisation apprentissage",
        "Travail décent sécurité sociale droits travailleurs",
    ],
    "Gouvernance": [
        "Modernisation administration publique réforme État services publics",
        "Lutte contre corruption transparence finances publiques audit contrôle",
        "Gestion budget fiscalité recettes dépenses trésor",
        "Justice institutions droits de l'homme État de droit",
        "Défense nationale sécurité intérieure police gendarmerie",
        "Relations extérieures diplomatie coopération internationale",
        "Elections démocratie institutions constitutionnelles parlement sénat",
        "Marchés publics passation contrats",
    ],
    "Développement Régional": [
        "Décentralisation collectivités territoriales développement local CTD",
        "Aménagement territoire équilibre régional disparités",
        "Habitat urbain logement assainissement villes",
        "Environnement forêts biodiversité changement climatique protection nature",
        "Reconstruction paix cohésion sociale Boko Haram Nord-Ouest Sud-Ouest CNDDR",
        "Développement rural zones défavorisées infrastructures rurales",
        "Cadastre domaines foncier patrimoine État",
        "Tourisme loisirs promotion régions",
    ],
}

COULEURS_PILIERS = {
    "Transformation Structurelle": "#F4A261",
    "Capital Humain":              "#2DC653",
    "Gouvernance":                 "#457B9D",
    "Développement Régional":      "#9B5DE5",
}

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT ET APLATISSAGE
# ─────────────────────────────────────────────────────────────────

def trouver_source() -> Path:
    for chemin in CANDIDATS_SOURCE:
        if chemin.exists():
            return chemin
    raise FileNotFoundError(
        f"Fichier source introuvable. Placez 'loi2024_2025_ligne_budgetaire.json' dans:\n"
        + "\n".join(f"  {c}" for c in CANDIDATS_SOURCE)
    )


def charger_et_aplatir() -> list:
    """
    Charge le JSON hiérarchique et extrait les programmes (niveau feuille).
    Chaque programme hérite du contexte de son chapitre.
    """
    source = trouver_source()
    print(f"  Source : {source}")

    with open(source, encoding="utf-8") as f:
        data = json.load(f)

    chapitres = data["credits_du_budget_general"]
    programmes = []

    for chap in chapitres:
        for prog in chap.get("programmes", []):
            # Montants : convertir les chaînes "1 234 567" en float
            def parse_montant(s):
                if not s:
                    return None
                v = re.sub(r"[\s\u00a0]", "", str(s))
                v = re.sub(r"[^\d.]", "", v)
                return float(v) if v else None

            cp = parse_montant(prog.get("cp"))
            ae = parse_montant(prog.get("ae"))

            # Texte enrichi pour l'embedding
            parties = []
            if prog.get("libelle"):
                parties.append(prog["libelle"])
            if prog.get("objectif"):
                parties.append(prog["objectif"])
            if prog.get("indicateur"):
                parties.append(prog["indicateur"])

            texte_enrichi = " | ".join(parties)

            programmes.append({
                "n":                 prog.get("n", ""),
                "code":              prog.get("code", ""),
                "libelle":           prog.get("libelle", ""),
                "objectif":          prog.get("objectif", ""),
                "indicateur":        prog.get("indicateur", ""),
                "texte":             texte_enrichi,
                "montant_cp":        cp,
                "montant_ae":        ae,
                "chapitre":          chap.get("chapitre", ""),
                "libelle_chapitre":  chap.get("libelle_chapitre", ""),
            })

    return programmes


# ─────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────

def classifier(programmes: list, model: SentenceTransformer) -> list:
    print("  Encodage des descripteurs SND30...")
    noms_piliers = list(PILIERS_SND30.keys())

    # Centroïde de chaque pilier = moyenne normalisée des descripteurs
    centroïdes = []
    for pilier in noms_piliers:
        embs = model.encode(PILIERS_SND30[pilier], normalize_embeddings=True)
        centroïde = embs.mean(axis=0)
        centroïde = centroïde / np.linalg.norm(centroïde)
        centroïdes.append(centroïde)
    matrice_piliers = np.array(centroïdes)  # (4, dim)

    print(f"  Encodage de {len(programmes)} programmes...")
    textes = [p["texte"] for p in programmes]
    emb_progs = model.encode(
        textes,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("  Classification...")
    resultats = []
    for i, prog in enumerate(programmes):
        scores = (emb_progs[i] @ matrice_piliers.T).tolist()
        scores_dict = {p: round(float(s), 4) for p, s in zip(noms_piliers, scores)}

        pilier_dominant = max(scores_dict, key=scores_dict.get)
        score_dominant  = scores_dict[pilier_dominant]
        scores_tries    = sorted(scores_dict.values(), reverse=True)
        ecart           = round(scores_tries[0] - scores_tries[1], 4)

        resultats.append({
            **prog,
            "scores":          scores_dict,
            "pilier":          pilier_dominant,
            "score_dominant":  round(score_dominant, 4),
            "ecart_confiance": ecart,
            "fiable":          score_dominant >= SEUIL_CONF and ecart >= SEUIL_ECART,
        })

    return resultats


# ─────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────

def plot_repartition(resultats: list) -> go.Figure:
    """Répartition CP par pilier (camembert + barres)."""
    # Agréger CP par pilier
    cp_par_pilier = {p: 0.0 for p in PILIERS_SND30}
    n_par_pilier  = Counter()
    for r in resultats:
        if r["pilier"] and r["montant_cp"]:
            cp_par_pilier[r["pilier"]] += r["montant_cp"]
            n_par_pilier[r["pilier"]] += 1

    piliers = list(cp_par_pilier.keys())
    valeurs_cp = [cp_par_pilier[p] / 1e6 for p in piliers]  # en milliards
    valeurs_n  = [n_par_pilier[p] for p in piliers]
    couleurs   = [COULEURS_PILIERS[p] for p in piliers]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=[
            "Répartition CP par pilier (milliards FCFA)",
            "Nombre de programmes par pilier",
        ],
    )

    fig.add_trace(go.Pie(
        labels=[p.replace(" ", "<br>") for p in piliers],
        values=valeurs_cp,
        marker=dict(colors=couleurs),
        textinfo="label+percent",
        textfont=dict(size=11),
        hole=0.4,
        hovertemplate="<b>%{label}</b><br>%{value:.1f} Mrd FCFA<br>%{percent}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=[p.replace(" ", "<br>") for p in piliers],
        y=valeurs_n,
        marker_color=couleurs,
        text=valeurs_n,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} programmes<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title="Répartition des crédits budgétaires par pilier SND30 — Loi 2024-2025",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", family="Georgia"),
        showlegend=False,
        height=480,
    )
    return fig


def plot_treemap(resultats: list) -> go.Figure:
    """Treemap : montants CP par chapitre, coloré par pilier."""
    labels, parents, values, colors, texts = [], [], [], [], []

    # Racine
    labels.append("Budget 2024-2025")
    parents.append("")
    values.append(0)
    colors.append("#1a1a2e")
    texts.append("")

    # Piliers
    for pilier in PILIERS_SND30:
        labels.append(pilier)
        parents.append("Budget 2024-2025")
        total = sum(r["montant_cp"] or 0 for r in resultats if r["pilier"] == pilier)
        values.append(total / 1e6)
        colors.append(COULEURS_PILIERS[pilier])
        texts.append(f"{total/1e9:.1f} Mrd")

    # Programmes
    for r in resultats:
        if not r["montant_cp"]:
            continue
        label = f"{r['code']} – {r['libelle'][:35]}"
        labels.append(label)
        parents.append(r["pilier"])
        values.append(r["montant_cp"] / 1e6)
        colors.append(COULEURS_PILIERS.get(r["pilier"], "#888"))
        texts.append(f"{r['montant_cp']/1e6:.0f} M")

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        text=texts,
        marker=dict(colors=colors, line=dict(width=1, color="#0F0F1A")),
        textinfo="label+text",
        hovertemplate="<b>%{label}</b><br>CP : %{value:.0f} M FCFA<extra></extra>",
        maxdepth=2,
    ))

    fig.update_layout(
        title="Treemap des crédits CP par programme et pilier SND30",
        paper_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", family="Georgia"),
        height=600,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return fig


def plot_top_programmes(resultats: list) -> go.Figure:
    """Top 20 programmes par CP, colorés par pilier."""
    top = sorted([r for r in resultats if r["montant_cp"]], 
                 key=lambda r: r["montant_cp"], reverse=True)[:20]

    labels  = [f"{r['code']} {r['libelle'][:40]}..." for r in top]
    valeurs = [r["montant_cp"] / 1e6 for r in top]
    couleurs = [COULEURS_PILIERS.get(r["pilier"], "#888") for r in top]
    hover   = [f"<b>{r['libelle']}</b><br>Pilier : {r['pilier']}<br>"
               f"Score : {r['score_dominant']:.3f}<br>"
               f"CP : {r['montant_cp']/1e6:.0f} M FCFA" for r in top]

    fig = go.Figure(go.Bar(
        x=valeurs,
        y=labels,
        orientation="h",
        marker_color=couleurs,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        text=[f"{v:.0f} M" for v in valeurs],
        textposition="outside",
        textfont=dict(size=9),
    ))

    fig.update_layout(
        title="Top 20 programmes par CP (milliers FCFA) — colorés par pilier SND30",
        xaxis_title="CP (millions FCFA)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", family="Georgia"),
        height=600,
        margin=dict(l=300),
    )

    # Légende manuelle
    for pilier, couleur in COULEURS_PILIERS.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name=pilier,
            marker_color=couleur,
            showlegend=True,
        ))

    return fig


def plot_scores_confiance(resultats: list) -> go.Figure:
    """Scatter : score dominant vs montant CP, coloré par pilier."""
    fig = go.Figure()

    for pilier in PILIERS_SND30:
        items = [r for r in resultats if r["pilier"] == pilier and r["montant_cp"]]
        if not items:
            continue
        fig.add_trace(go.Scatter(
            x=[r["score_dominant"] for r in items],
            y=[r["montant_cp"] / 1e6 for r in items],
            mode="markers",
            name=pilier,
            marker=dict(
                color=COULEURS_PILIERS[pilier],
                size=[max(6, min(30, r["montant_cp"] / 5e6)) for r in items],
                opacity=0.8,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Score : %{x:.3f}<br>"
                "CP : %{y:.0f} M FCFA<extra></extra>"
            ),
            customdata=[[r["libelle"]] for r in items],
        ))

    fig.add_vline(x=SEUIL_CONF, line_dash="dash", line_color="#E63946",
                  annotation_text=f"Seuil confiance ({SEUIL_CONF})")

    fig.update_layout(
        title="Score de classification vs CP — (taille = montant CP)",
        xaxis_title="Score de similarité SND30",
        yaxis_title="CP (millions FCFA)",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#F0EAD6", family="Georgia"),
        legend=dict(bgcolor="rgba(255,255,255,0.05)"),
        height=500,
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# RAPPORT TEXTE
# ─────────────────────────────────────────────────────────────────

def generer_rapport(resultats: list) -> str:
    lignes = []
    lignes.append("=" * 70)
    lignes.append("  RAPPORT CLASSIFICATION ZERO-SHOT SND30")
    lignes.append("  Loi de Finances 2024-2025 — Cameroun")
    lignes.append("=" * 70)

    total_cp = sum(r["montant_cp"] for r in resultats if r["montant_cp"])
    fiables  = [r for r in resultats if r["fiable"]]

    lignes.append(f"\n  Programmes classifiés   : {len(resultats)}")
    lignes.append(f"  Dont fiables            : {len(fiables)} ({100*len(fiables)/len(resultats):.1f}%)")
    lignes.append(f"  Total CP                : {total_cp/1e6:,.0f} milliards FCFA")

    # CP par pilier
    lignes.append(f"\n  Répartition CP par pilier SND30 :")
    cp_par_pilier = {p: 0.0 for p in PILIERS_SND30}
    n_par_pilier  = Counter()
    for r in resultats:
        if r["montant_cp"]:
            cp_par_pilier[r["pilier"]] += r["montant_cp"]
            n_par_pilier[r["pilier"]] += 1

    for pilier in sorted(cp_par_pilier, key=cp_par_pilier.get, reverse=True):
        cp  = cp_par_pilier[pilier]
        n   = n_par_pilier[pilier]
        pct = 100 * cp / total_cp if total_cp else 0
        lignes.append(f"    {pilier:<35} : {cp/1e6:>10,.0f} Mrd  ({pct:.1f}%)  [{n} programmes]")

    # Top 5 par pilier
    lignes.append(f"\n  Top 5 programmes CP par pilier :")
    for pilier in PILIERS_SND30:
        lignes.append(f"\n  ── {pilier} ──")
        top = sorted([r for r in resultats if r["pilier"] == pilier and r["montant_cp"]],
                     key=lambda r: r["montant_cp"], reverse=True)[:5]
        for r in top:
            lignes.append(f"    {r['montant_cp']/1e6:>10,.0f} M  [{r['code']}] {r['libelle'][:50]}")
            lignes.append(f"               Score={r['score_dominant']:.3f}  Écart={r['ecart_confiance']:.3f}")

    lignes.append(f"\n{'='*70}\n")
    return "\n".join(lignes)


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  CLASSIFICATION ZERO-SHOT SND30 — v2")
    print("="*60)

    print("\n[1/3] Chargement et aplatissage des programmes...")
    programmes = charger_et_aplatir()
    print(f"  {len(programmes)} programmes chargés")

    print("\n[2/3] Chargement du modèle MiniLM...")
    model = SentenceTransformer(MODELE_ID)

    print("\n[3/3] Classification SND30...")
    resultats = classifier(programmes, model)

    # Stats rapides
    cp_par_pilier = Counter()
    for r in resultats:
        if r["montant_cp"]:
            cp_par_pilier[r["pilier"]] += r["montant_cp"]

    print("\n  Répartition CP par pilier :")
    total_cp = sum(cp_par_pilier.values())
    for pilier, cp in sorted(cp_par_pilier.items(), key=lambda x: -x[1]):
        print(f"    {pilier:<35} : {cp/1e9:>6.1f} Mrd  ({100*cp/total_cp:.1f}%)")

    # Sauvegarde JSON
    sortie_json = PROCESSED_DIR / "classification_snd30_v2.json"
    with open(sortie_json, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON : {sortie_json.name}")

    # Visualisations
    print("  Génération des visualisations...")
    figs = [
        ("Répartition",    plot_repartition(resultats)),
        ("Treemap",        plot_treemap(resultats)),
        ("Top programmes", plot_top_programmes(resultats)),
        ("Scores",         plot_scores_confiance(resultats)),
    ]

    html_parts = [
        "<h1 style='font-family:Georgia;color:#F0EAD6;background:#0F0F1A;"
        "padding:20px 30px;margin:0'>Classification SND30 — Loi de Finances 2024-2025</h1>"
    ]
    for i, (_, fig) in enumerate(figs):
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

    sortie_html = REPORTS_DIR / "classification_snd30_v2.html"
    with open(sortie_html, "w", encoding="utf-8") as f:
        f.write(html_final)
    print(f"  HTML : {sortie_html.name}")

    # Rapport texte
    rapport = generer_rapport(resultats)
    sortie_rapport = REPORTS_DIR / "classification_snd30_rapport_v2.txt"
    with open(sortie_rapport, "w", encoding="utf-8") as f:
        f.write(rapport)
    print(f"  Rapport : {sortie_rapport.name}")
    print("\n" + rapport)

    print(f"\n{'='*60}")
    print("  Prochaine étape : src/statistics/conformite.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()