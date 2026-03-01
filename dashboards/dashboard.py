"""
dashboard.py
============
Phase 5 — Dashboard Streamlit
Tableau de bord interactif : NLP & Lois de Finances du Cameroun

Sections :
  1. Glissement sémantique 2023-2024 → 2024-2025
  2. Répartition des dépenses par pilier SND30
  3. Zones de divergence discours / budget

Lancement :
  streamlit run src/dashboard/dashboard.py
"""

import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION GÉNÉRALE
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NLP · Finances Cameroun",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Couleurs
COULEURS_PILIERS = {
    "Transformation Structurelle": "#F4A261",
    "Capital Humain":              "#2DC653",
    "Gouvernance":                 "#457B9D",
    "Développement Régional":      "#9B5DE5",
}
PILIERS = list(COULEURS_PILIERS.keys())

BG_COLOR    = "#E6E6ED"
CARD_COLOR  = "#1A1A2E"
TEXT_COLOR  = "#352B0B"
ACCENT      = "#E63946"

# CSS personnalisé
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
    .block-container {{ padding: 2rem 3rem; }}
    h1, h2, h3 {{ color: {TEXT_COLOR}; font-family: Georgia, serif; }}
    .metric-card {{
        background: {CARD_COLOR};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid {ACCENT};
        margin-bottom: 0.5rem;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {TEXT_COLOR};
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.2rem;
    }}
    .pilier-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 2px;
    }}
    .section-header {{
        background: linear-gradient(90deg, {ACCENT}22, transparent);
        border-left: 4px solid {ACCENT};
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-family: Georgia, serif;
        font-size: 1.2rem;
        color: {TEXT_COLOR};
    }}
    div[data-testid="stMetricValue"] {{ color: {TEXT_COLOR} !important; }}
    .stTabs [data-baseweb="tab"] {{ color: #aaa; }}
    .stTabs [aria-selected="true"] {{ color: {TEXT_COLOR} !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def chemin(rel):
    return BASE_DIR / rel

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES (mis en cache)
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def charger_audit():
    p = chemin("data/processed/audit_correspondances.json")
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def charger_classification():
    p = chemin("data/processed/classification_snd30_v2.json")
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def charger_articles(annee):
    p = chemin(f"data/processed/loi_{annee.replace('-','_')}_articles.json")
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)

# Codes incompressibles
CODES_INCOMPRESSIBLES = {
    "199","203","200","201","202","195","196","197","198"
}

def parse_montant(s):
    if not s:
        return None
    import re
    v = re.sub(r"[\s\u00a0]", "", str(s))
    v = re.sub(r"[^\d.]", "", v)
    return float(v) if v else None

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:1rem 0'>
        <div style='font-size:2.5rem'>🇨🇲</div>
        <div style='font-family:Georgia;font-size:1.1rem;color:{TEXT_COLOR};
                    font-weight:bold;margin-top:0.5rem'>
            NLP · Finances Cameroun
        </div>
        <div style='font-size:0.75rem;color:#888;margin-top:0.3rem'>
            Loi de Finances 2023–2025
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    section = st.radio(
        "Navigation",
        [
            "> Vue d'ensemble",
            "> Glissement Sémantique",
            "> Classification SND30",
            "> Conformité Discours/Budget",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem;color:#555;text-align:center'>"
        "ISE3 · ISSEA Yaoundé<br>NLP · Lois de Finances</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────
# SECTION 0 : VUE D'ENSEMBLE
# ─────────────────────────────────────────────────────────────────

if section == "🏠 Vue d'ensemble":
    st.markdown(
        "<h1 style='font-family:Georgia;margin-bottom:0.2rem'>"
        "Analyse NLP des Lois de Finances du Cameroun</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#aaa;margin-bottom:2rem'>"
        "Audit sémantique · Classification SND30 · Conformité budgétaire</p>",
        unsafe_allow_html=True,
    )

    # Métriques globales
    audit       = charger_audit()
    classif     = charger_classification()
    arts_2023   = charger_articles("2023-2024")
    arts_2024   = charger_articles("2024-2025")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(arts_2023)}</div>
            <div class='metric-label'>Articles — Loi 2023-2024</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card' style='border-left-color:#F4A261'>
            <div class='metric-value'>{len(arts_2024)}</div>
            <div class='metric-label'>Articles — Loi 2024-2025</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        n_progs = len(classif) if classif else 0
        st.markdown(f"""
        <div class='metric-card' style='border-left-color:#2DC653'>
            <div class='metric-value'>{n_progs}</div>
            <div class='metric-label'>Programmes budgétaires classifiés</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        total_cp = sum(
            parse_montant(r.get("montant_cp")) or 0
            for r in (classif or [])
        )
        st.markdown(f"""
        <div class='metric-card' style='border-left-color:#9B5DE5'>
            <div class='metric-value'>{total_cp/1e9:,.0f} Mrd</div>
            <div class='metric-label'>Total CP 2024-2025 (FCFA)</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Pipeline
    st.markdown("<div class='section-header'>🔄 Pipeline du projet</div>",
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        phases = [
            ("✅", "Phase 1", "Extraction & Segmentation", "OCR + pdfplumber + camelot"),
            ("✅", "Phase 2", "Embeddings & Audit Sémantique", "CamemBERT + MiniLM"),
            ("✅", "Phase 3", "Classification Zero-Shot SND30", "Similarité cosinus"),
            ("✅", "Phase 4", "Conformité Statistique", "Spearman · Chi-2 · K-Means · HDBSCAN"),
            ("✅", "Phase 5", "Dashboard Streamlit", "Visualisation interactive"),
        ]
        for icon, num, titre, detail in phases:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:12px;
                        padding:0.7rem;margin:0.3rem 0;
                        background:{CARD_COLOR};border-radius:8px'>
                <span style='font-size:1.2rem'>{icon}</span>
                <div>
                    <div style='font-size:0.75rem;color:#888'>{num}</div>
                    <div style='font-weight:bold;color:{TEXT_COLOR}'>{titre}</div>
                    <div style='font-size:0.78rem;color:#aaa'>{detail}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        if classif:
            cp_par_pilier = {p: 0.0 for p in PILIERS}
            for r in classif:
                cp = parse_montant(r.get("montant_cp")) or 0
                if r.get("pilier") in cp_par_pilier:
                    cp_par_pilier[r["pilier"]] += cp

            fig = go.Figure(go.Pie(
                labels=list(cp_par_pilier.keys()),
                values=[v/1e9 for v in cp_par_pilier.values()],
                marker=dict(colors=[COULEURS_PILIERS[p] for p in PILIERS]),
                hole=0.5,
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                title="Répartition CP par pilier SND30",
                paper_bgcolor=BG_COLOR,
                plot_bgcolor=BG_COLOR,
                font=dict(color=TEXT_COLOR, family="Georgia"),
                showlegend=False,
                height=380,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# SECTION 1 : GLISSEMENT SÉMANTIQUE
# ─────────────────────────────────────────────────────────────────

elif section == "📊 Glissement Sémantique":
    st.markdown(
        "<h1 style='font-family:Georgia'>Audit Sémantique</h1>"
        "<p style='color:#aaa'>Évolution du discours législatif entre 2023-2024 et 2024-2025</p>",
        unsafe_allow_html=True,
    )

    audit = charger_audit()
    if not audit:
        st.error("Fichier audit_correspondances.json introuvable.")
        st.stop()

    modele = st.selectbox("Modèle d'embedding", ["camembert", "minilm"])
    corresp = audit.get(modele, [])

    if not corresp:
        st.warning(f"Aucune donnée pour le modèle {modele}.")
        st.stop()

    scores = [c["score"] for c in corresp if c.get("score", 0) > 0]

    # KPIs
    seuil_stable  = 0.90
    seuil_modifie = 0.70

    n_total   = len([c for c in corresp if c.get("ref_2023")])
    n_stable  = sum(1 for c in corresp if c.get("statut") == "STABLE")
    n_modifie = sum(1 for c in corresp if c.get("statut") == "MODIFIÉ")
    n_tres    = sum(1 for c in corresp if c.get("statut") == "TRÈS MODIFIÉ / NOUVEAU")
    n_nouveau = sum(1 for c in corresp if c.get("statut") == "NOUVEAU")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label, couleur in [
        (c1, f"{np.mean(scores):.3f}", "Sim. moyenne", TEXT_COLOR),
        (c2, n_stable,  "Stables ✅",         "#2DC653"),
        (c3, n_modifie, "Modifiés ⚠️",         "#F4A261"),
        (c4, n_tres,    "Très modifiés ❌",    "#E63946"),
        (c5, n_nouveau, "Nouveaux 🆕",         "#9B5DE5"),
    ]:
        col.markdown(f"""
        <div class='metric-card' style='border-left-color:{couleur}'>
            <div class='metric-value' style='color:{couleur}'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📈 Distribution", "🔥 Articles modifiés", "🔍 Explorateur"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores, nbinsx=40,
            marker_color="#457B9D", opacity=0.85, name="Distribution",
        ))
        fig.add_vrect(x0=seuil_stable, x1=1.0,
                      fillcolor="#2DC653", opacity=0.08,
                      annotation_text="Stable", annotation_position="top left")
        fig.add_vrect(x0=seuil_modifie, x1=seuil_stable,
                      fillcolor="#F4A261", opacity=0.08,
                      annotation_text="Modifié", annotation_position="top left")
        fig.add_vrect(x0=0, x1=seuil_modifie,
                      fillcolor="#E63946", opacity=0.08,
                      annotation_text="Très modifié", annotation_position="top left")
        fig.add_vline(x=np.mean(scores), line_dash="dash", line_color="white",
                      annotation_text=f"μ={np.mean(scores):.3f}")
        fig.update_layout(
            title=f"Distribution des similarités cosinus — {modele.upper()}",
            xaxis_title="Score de similarité",
            yaxis_title="Nombre d'articles",
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            font=dict(color=TEXT_COLOR, family="Georgia"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        tres_modifies = sorted(
            [c for c in corresp if c.get("statut") in
             ("TRÈS MODIFIÉ / NOUVEAU", "NOUVEAU") and c.get("score", 0) > 0],
            key=lambda x: x["score"]
        )

        seuil_filtre = st.slider(
            "Seuil de similarité maximum", 0.0, 1.0, 0.70, 0.01
        )
        filtres = [c for c in tres_modifies if c["score"] <= seuil_filtre]

        st.markdown(f"**{len(filtres)} articles** avec score ≤ {seuil_filtre}")

        fig2 = go.Figure(go.Bar(
            x=[c["score"] for c in filtres[:20]],
            y=[f"{(c.get('ref_2023') or 'NOUVEAU')[:30]}" for c in filtres[:20]],
            orientation="h",
            marker_color=ACCENT,
            text=[f"{c['score']:.3f}" for c in filtres[:20]],
            textposition="outside",
        ))
        fig2.update_layout(
            title="Top 20 articles les plus modifiés",
            xaxis_title="Score de similarité",
            yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            font=dict(color=TEXT_COLOR, family="Georgia"),
            height=500, margin=dict(l=230),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("#### Rechercher un article")
        query = st.text_input("Filtrer par référence", placeholder="ex: Article VINGT")

        filtres_exp = [
            c for c in corresp
            if query.lower() in str(c.get("ref_2023", "")).lower()
            or query.lower() in str(c.get("ref_2024", "")).lower()
        ] if query else corresp[:50]

        for c in filtres_exp[:30]:
            score  = c.get("score", 0)
            statut = c.get("statut", "?")
            couleur_statut = {
                "STABLE": "#2DC653",
                "MODIFIÉ": "#F4A261",
                "TRÈS MODIFIÉ / NOUVEAU": ACCENT,
                "NOUVEAU": "#9B5DE5",
            }.get(statut, "#888")

            with st.expander(
                f"[{score:.3f}] {c.get('ref_2023', 'N/A')[:50]} → {c.get('ref_2024','N/A')[:50]}"
            ):
                col_g, col_d = st.columns(2)
                with col_g:
                    st.markdown("**Loi 2023-2024**")
                    st.caption(c.get("texte_2023", "")[:300])
                with col_d:
                    st.markdown("**Loi 2024-2025**")
                    st.caption(c.get("texte_2024", "")[:300])
                st.markdown(
                    f"<span style='background:{couleur_statut};color:#000;"
                    f"padding:2px 10px;border-radius:20px;font-size:0.8rem'>"
                    f"{statut}</span>  Score : {score:.4f}",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────
# SECTION 2 : CLASSIFICATION SND30
# ─────────────────────────────────────────────────────────────────

elif section == "🎯 Classification SND30":
    st.markdown(
        "<h1 style='font-family:Georgia'>Classification SND30</h1>"
        "<p style='color:#aaa'>Répartition des 182 programmes budgétaires par pilier SND30</p>",
        unsafe_allow_html=True,
    )

    classif = charger_classification()
    if not classif:
        st.error("Fichier classification_snd30_v2.json introuvable.")
        st.stop()

    # KPIs
    cp_par_pilier = {p: 0.0 for p in PILIERS}
    n_par_pilier  = Counter()
    for r in classif:
        cp = parse_montant(r.get("montant_cp")) or 0
        if r.get("pilier") in cp_par_pilier:
            cp_par_pilier[r["pilier"]] += cp
            n_par_pilier[r["pilier"]] += 1

    total_cp = sum(cp_par_pilier.values())

    cols = st.columns(4)
    for i, p in enumerate(PILIERS):
        with cols[i]:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color:{COULEURS_PILIERS[p]}'>
                <div class='metric-value' style='color:{COULEURS_PILIERS[p]}'>
                    {cp_par_pilier[p]/1e9:.1f} Mrd
                </div>
                <div class='metric-label'>{p}<br>
                    {n_par_pilier[p]} programmes · {100*cp_par_pilier[p]/total_cp:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Vue globale", "🗺️ Treemap", "🔍 Programmes"])

    with tab1:
        col_g, col_d = st.columns(2)

        with col_g:
            fig = go.Figure(go.Pie(
                labels=PILIERS,
                values=[cp_par_pilier[p]/1e9 for p in PILIERS],
                marker=dict(colors=[COULEURS_PILIERS[p] for p in PILIERS]),
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                title="CP par pilier (Mrd FCFA)",
                paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
                font=dict(color=TEXT_COLOR, family="Georgia"),
                showlegend=False, height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Programmes (%)",
                x=PILIERS,
                y=[100*n_par_pilier[p]/len(classif) for p in PILIERS],
                marker_color=[COULEURS_PILIERS[p] for p in PILIERS],
                opacity=0.9,
            ))
            fig2.add_trace(go.Bar(
                name="CP (%)",
                x=PILIERS,
                y=[100*cp_par_pilier[p]/total_cp for p in PILIERS],
                marker_color=[COULEURS_PILIERS[p] for p in PILIERS],
                opacity=0.45,
                marker_pattern_shape="/",
            ))
            fig2.update_layout(
                barmode="group",
                title="Fréquence vs CP par pilier",
                paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
                font=dict(color=TEXT_COLOR, family="Georgia"),
                legend=dict(bgcolor="rgba(255,255,255,0.05)"),
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        labels, parents, values, colors = [], [], [], []
        labels.append("Budget 2024-2025")
        parents.append(""); values.append(0); colors.append(BG_COLOR)

        for p in PILIERS:
            labels.append(p); parents.append("Budget 2024-2025")
            values.append(cp_par_pilier[p]/1e6)
            colors.append(COULEURS_PILIERS[p])

        for r in classif:
            cp = parse_montant(r.get("montant_cp")) or 0
            if not cp: continue
            label = f"{r.get('code','')} · {r.get('libelle','')[:35]}"
            labels.append(label); parents.append(r.get("pilier",""))
            values.append(cp/1e6)
            colors.append(COULEURS_PILIERS.get(r.get("pilier",""), "#888"))

        fig3 = go.Figure(go.Treemap(
            labels=labels, parents=parents, values=values,
            marker=dict(colors=colors, line=dict(width=1, color=BG_COLOR)),
            textinfo="label",
            hovertemplate="<b>%{label}</b><br>CP : %{value:.0f} M FCFA<extra></extra>",
            maxdepth=2,
        ))
        fig3.update_layout(
            paper_bgcolor=BG_COLOR, font=dict(color=TEXT_COLOR, family="Georgia"),
            height=550, margin=dict(t=10, l=5, r=5, b=5),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        col1, col2 = st.columns([1, 1])
        with col1:
            pilier_filtre = st.selectbox("Filtrer par pilier", ["Tous"] + PILIERS)
        with col2:
            tri = st.selectbox("Trier par", ["CP décroissant", "Score décroissant"])

        filtres = classif if pilier_filtre == "Tous" else [
            r for r in classif if r.get("pilier") == pilier_filtre
        ]
        if tri == "CP décroissant":
            filtres = sorted(filtres, key=lambda r: parse_montant(r.get("montant_cp")) or 0, reverse=True)
        else:
            filtres = sorted(filtres, key=lambda r: r.get("score_dominant", 0), reverse=True)

        for r in filtres[:40]:
            cp    = parse_montant(r.get("montant_cp")) or 0
            pilier = r.get("pilier", "?")
            couleur = COULEURS_PILIERS.get(pilier, "#888")
            score  = r.get("score_dominant", 0)

            with st.expander(
                f"[{r.get('code','')}] {r.get('libelle','')[:60]} — {cp/1e6:,.0f} M FCFA"
            ):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"**Objectif :** {r.get('objectif','N/A')}")
                    st.markdown(f"**Indicateur :** {r.get('indicateur','N/A')}")
                    st.markdown(f"**Chapitre :** {r.get('libelle_chapitre','N/A')}")
                with c2:
                    st.markdown(
                        f"<span class='pilier-badge' style='background:{couleur};color:#000'>"
                        f"{pilier}</span>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Score", f"{score:.3f}")
                    st.metric("Écart confiance", f"{r.get('ecart_confiance',0):.3f}")
                    st.metric("CP", f"{cp/1e6:,.0f} M FCFA")

# ─────────────────────────────────────────────────────────────────
# SECTION 3 : CONFORMITÉ DISCOURS / BUDGET
# ─────────────────────────────────────────────────────────────────

elif section == "⚖️ Conformité Discours/Budget":
    st.markdown(
        "<h1 style='font-family:Georgia'>Conformité Discours / Budget</h1>"
        "<p style='color:#aaa'>"
        "Alignement entre la fréquence thématique et les crédits alloués</p>",
        unsafe_allow_html=True,
    )

    classif = charger_classification()
    if not classif:
        st.error("Fichier classification_snd30_v2.json introuvable.")
        st.stop()

    mode = st.radio(
        "Périmètre d'analyse",
        ["Budget total (182 programmes)", "Budget discrétionnaire (hors incompressibles)"],
        horizontal=True,
    )

    progs = classif if "total" in mode else [
        r for r in classif if r.get("code","") not in CODES_INCOMPRESSIBLES
    ]

    # Calcul stats
    cp_par_pilier = {p: 0.0 for p in PILIERS}
    n_par_pilier  = Counter()
    for r in progs:
        cp = parse_montant(r.get("montant_cp")) or 0
        if r.get("pilier") in cp_par_pilier:
            cp_par_pilier[r["pilier"]] += cp
            n_par_pilier[r["pilier"]] += 1

    total_n  = len(progs)
    total_cp = sum(cp_par_pilier.values())

    freq_pct = {p: 100*n_par_pilier[p]/total_n    for p in PILIERS}
    cp_pct   = {p: 100*cp_par_pilier[p]/total_cp  for p in PILIERS}
    ecarts   = {p: cp_pct[p] - freq_pct[p]        for p in PILIERS}

    # Spearman
    from scipy import stats as scipy_stats
    freq_v = [freq_pct[p] for p in PILIERS]
    cp_v   = [cp_pct[p]   for p in PILIERS]
    rho, pval = scipy_stats.spearmanr(freq_v, cp_v)

    # KPI Spearman
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{rho:.3f}</div>
            <div class='metric-label'>ρ de Spearman</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        sig = "✅ Significatif" if pval < 0.05 else "❌ Non significatif"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{pval:.3f}</div>
            <div class='metric-label'>p-value · {sig}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        max_ecart = max(abs(ecarts[p]) for p in PILIERS)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{max_ecart:.1f}%</div>
            <div class='metric-label'>Écart max discours/budget</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Écarts", "🔬 Spearman", "📋 Tableau détaillé"])

    with tab1:
        col_g, col_d = st.columns(2)

        with col_g:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Fréquence (% programmes)",
                x=PILIERS, y=list(freq_pct.values()),
                marker_color=[COULEURS_PILIERS[p] for p in PILIERS], opacity=0.9,
                text=[f"{v:.1f}%" for v in freq_pct.values()], textposition="outside",
            ))
            fig.add_trace(go.Bar(
                name="CP (% budget)",
                x=PILIERS, y=list(cp_pct.values()),
                marker_color=[COULEURS_PILIERS[p] for p in PILIERS], opacity=0.4,
                marker_pattern_shape="/",
                text=[f"{v:.1f}%" for v in cp_pct.values()], textposition="outside",
            ))
            fig.update_layout(
                barmode="group", title="Fréquence vs CP par pilier",
                paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
                font=dict(color=TEXT_COLOR, family="Georgia"),
                legend=dict(bgcolor="rgba(255,255,255,0.05)"), height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            ecart_vals = [ecarts[p] for p in PILIERS]
            couleurs_e = ["#2DC653" if e >= 0 else ACCENT for e in ecart_vals]
            fig2 = go.Figure(go.Bar(
                x=ecart_vals, y=PILIERS, orientation="h",
                marker_color=couleurs_e,
                text=[f"{e:+.1f}%" for e in ecart_vals], textposition="outside",
            ))
            fig2.add_vline(x=0, line_color="white", line_width=1)
            fig2.update_layout(
                title="Écart CP% − Fréquence%<br>(+ = sur-financé · − = sous-financé)",
                paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
                font=dict(color=TEXT_COLOR, family="Georgia"),
                height=420, margin=dict(l=200),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Interprétation
        st.markdown("<div class='section-header'>💡 Interprétation</div>",
                    unsafe_allow_html=True)
        for p in sorted(PILIERS, key=lambda x: abs(ecarts[x]), reverse=True):
            e = ecarts[p]
            if e > 5:
                msg = f"**sur-financé** par rapport à son poids discursif (+{e:.1f}%)"
                icone = "🟢"
            elif e < -5:
                msg = f"**sous-financé** par rapport à son poids discursif ({e:.1f}%)"
                icone = "🔴"
            else:
                msg = f"alignement relatif (écart = {e:+.1f}%)"
                icone = "🟡"
            st.markdown(
                f"{icone} **{p}** : {freq_pct[p]:.1f}% des programmes, "
                f"{cp_pct[p]:.1f}% du budget → {msg}"
            )

    with tab2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=freq_v, y=cp_v, mode="markers+text",
            text=PILIERS, textposition="top center", textfont=dict(size=11),
            marker=dict(
                color=[COULEURS_PILIERS[p] for p in PILIERS],
                size=20, line=dict(width=2, color="white"),
            ),
            hovertemplate="<b>%{text}</b><br>Fréq : %{x:.1f}%<br>CP : %{y:.1f}%<extra></extra>",
        ))
        m, b = np.polyfit(freq_v, cp_v, 1)
        x_l = np.linspace(min(freq_v)-2, max(freq_v)+2, 50)
        fig3.add_trace(go.Scatter(
            x=x_l, y=m*x_l+b, mode="lines",
            line=dict(dash="dash", color="rgba(255,255,255,0.3)"),
            showlegend=False,
        ))
        fig3.add_shape(type="line", x0=0, x1=50, y0=0, y1=50,
                       line=dict(dash="dot", color="rgba(255,255,255,0.15)"))
        fig3.update_layout(
            title=f"Corrélation Spearman — ρ={rho:.3f}  p={pval:.3f}",
            xaxis_title="Fréquence (% programmes)",
            yaxis_title="CP (% budget)",
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            font=dict(color=TEXT_COLOR, family="Georgia"),
            height=460,
            annotations=[dict(
                x=0.02, y=0.97, xref="paper", yref="paper",
                text=f"ρ = {rho:.3f}<br>p = {pval:.3f}<br>"
                     f"{'✅ Significatif' if pval < 0.05 else '❌ Non significatif'}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.07)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1, font=dict(size=12), align="left",
            )],
        )
        st.plotly_chart(fig3, use_container_width=True)

        if pval >= 0.05:
            st.warning(
                "⚠️ La corrélation n'est pas statistiquement significative (p ≥ 0.05). "
                "Il existe un **désalignement** entre le poids discursif des piliers "
                "et les crédits qui leur sont alloués."
            )
        else:
            st.success(
                "✅ La corrélation est statistiquement significative : "
                "les piliers les plus représentés reçoivent proportionnellement plus de crédits."
            )

    with tab3:
        import pandas as pd
        df = pd.DataFrame([{
            "Pilier":        p,
            "Programmes":    n_par_pilier[p],
            "Fréquence (%)": round(freq_pct[p], 1),
            "CP (M FCFA)":   round(cp_par_pilier[p]/1e6, 0),
            "CP (%)":        round(cp_pct[p], 1),
            "Écart (pp)":    round(ecarts[p], 1),
            "Statut":        (
                "Sur-financé"   if ecarts[p] >  5 else
                "Sous-financé"  if ecarts[p] < -5 else
                "Aligné"
            ),
        } for p in PILIERS])

        st.dataframe(
            df.style.background_gradient(subset=["Écart (pp)"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

        csv = df.to_csv(index=False, sep=";").encode("utf-8")
        st.download_button(
            "⬇️ Télécharger CSV",
            data=csv,
            file_name="conformite_snd30.csv",
            mime="text/csv",
        )