
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter
import re

# ─────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CM-BUDGET TRACK",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# GESTION DE LA PAGE D'ACCUEIL (LANDING PAGE)
# ─────────────────────────────────────────────────────────────────

# 1. On initialise la mémoire de l'application
if "page_actuelle" not in st.session_state:
    st.session_state.page_actuelle = "accueil"

# 2. Fonction pour changer de page
def entrer_dashboard():
    st.session_state.page_actuelle = "dashboard"

# 3. SI ON EST SUR L'ACCUEIL : On dessine la page et on ARRETE le script
if st.session_state.page_actuelle == "accueil":
    
    # CSS spécifique pour cacher le menu latéral sur l'accueil et mettre un fond sombre élégant
    st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none !important;}
        [data-testid="stSidebar"] {display: none !important;}
        .stApp {background-color: #0F0F1A; color: #F0EAD6;}
        .titre-accueil {font-size: 3.5rem; font-weight: bold; text-align: center; margin-top: 1rem;}
        .sous-titre {font-size: 1.2rem; text-align: center; color: #999999; margin-bottom: 2rem;}
    </style>
    """, unsafe_allow_html=True)

   # --- DESIGN DE L'ACCUEIL ---
    dossier_assets = Path(__file__).resolve().parent / "assets"

    # 1. LA BANNIÈRE (Pleine largeur)
    chemin_banniere = dossier_assets / "dash_banner.png"
    if chemin_banniere.exists():
        st.image(str(chemin_banniere), use_container_width=True)

    
    

    # 2. LIGNE DU BAS : BOUTON (Gauche) ET PARTENAIRES (Droite)
    # On divise l'écran en 5 colonnes avec des tailles précises :
    # [Marge gauche, Bouton, Espace central, Partenaires, Marge droite]
    espace_g, col_bouton, espace_milieu, col_partenaires, espace_d = st.columns([0.1, 0.8, 1.6, 0.5, 0.1])

    # --- PARTIE GAUCHE : LE BOUTON ---
    with col_bouton:
        # On ajoute un peu de marge en haut pour que le bouton soit aligné au même niveau que les logos
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        
        st.button("ACCÉDER AU DASHBOARD", on_click=entrer_dashboard, use_container_width=True, type="primary")

    # --- PARTIE DROITE : LES PARTENAIRES ---
    with col_partenaires:
        st.markdown("<div style='text-align:center; color:#999999; font-size:0.5rem; margin-bottom:10px;'>AVEC LE SOUTIEN DE:</div>", unsafe_allow_html=True)
        
        # On sous-divise cette colonne en 4 pour mettre les logos sur la même ligne
        p1, p2, p3, p4 = st.columns(4)
        try:
            with p1: st.image(str(dossier_assets / "partenaire1.png"), use_container_width=True)
            with p2: st.image(str(dossier_assets / "partenaire2.png"), use_container_width=True)
            with p3: st.image(str(dossier_assets / "partenaire3.png"), use_container_width=True)
            with p4: st.image(str(dossier_assets / "partenaire4.png"), use_container_width=True)
        except:
            pass

    # Espace final en bas
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 🛑 MAGIE : On arrête le code ici. Le dashboard ne s'affichera pas !
    st.stop()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR (Ce code ne s'exécutera que si on a cliqué sur le bouton)
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# THÈMES
# ─────────────────────────────────────────────────────────────────

THEMES = {
    "Dark": {
        "bg": "#0F0F1A", "card": "#1A1A2E", "text": "#F0EAD6",
        "subtext": "#999999", "divider": "#2a2a3e", "accent": "#0558F1",
        "plot_bg": "#0F0F1A", "paper_bg": "#0F0F1A", "grid": "#1e1e30",
    },
    "Light": {
        "bg": "#F7F7F2", "card": "#FFFFFF", "text": "#1A1A2E",
        "subtext": "#555555", "divider": "#E0E0E0", "accent": "#0558F1",
        "plot_bg": "#FFFFFF", "paper_bg": "#F7F7F2", "grid": "#EEEEEE",
    },
}

COULEURS_PILIERS = {
    "Transformation Structurelle": "#F4A261",
    "Capital Humain":              "#2DC653",
    "Gouvernance":                 "#0558F1",
    "Développement Régional":      "#E8490A",
}
PILIERS = list(COULEURS_PILIERS.keys())
CODES_INCOMPRESSIBLES = {"199","203","200","201","202","195","196","197","198"}

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

# 1. On réserve 3 "boîtes" dans l'ordre d'affichage (Haut, Milieu vide, Bas)
sidebar_top = st.sidebar.container()
sidebar_spacer = st.sidebar.container()
sidebar_bottom = st.sidebar.container()

# 2. On exécute la boîte du bas EN PREMIER pour récupérer le choix du thème
with sidebar_bottom:
    #st.divider()  Petite ligne pour séparer du reste
    # st.toggle crée exactement le bouton en forme de pilule de ton image !
    mode_sombre = st.toggle("Dark/Light", value=True)
    
    # On met à jour nos variables de couleur selon le bouton
    theme_choix = "Dark" if mode_sombre else "Light"
    T = THEMES[theme_choix]

# 3. On crée un grand espace invisible pour pousser le bouton tout en bas
with sidebar_spacer:
    # 40vh = 40% de la hauteur de l'écran. 
    # Modifie le chiffre (ex: 50vh ou 30vh) si tu veux le pousser plus ou moins bas.
    st.markdown("<div style='min-height: 2vh;'></div>", unsafe_allow_html=True)

# 4. Maintenant on dessine le haut du menu
with sidebar_top:
    
    dossier_actuel = Path(__file__).resolve().parent
    dossier_assets = dossier_actuel / "assets"
    
    # --- 1. LE LOGO DE L'APPLICATION ---
    col_gauche, col_logo, col_droite = st.columns([0.5, 3, 0.5]) 
    with col_logo:
        try:
            st.image(str(dossier_assets / "logo_app.png"), use_container_width=True)
        except Exception:
            st.warning("Logo?")

    # --- 2. LE BOUTON ACCUEIL (Petit et centré) ---
    st.markdown("<div style='margin-top: 0.1rem;'></div>", unsafe_allow_html=True)
    
    # On crée 3 nouvelles colonnes pour "écraser" le bouton au milieu
    c_btn_g, c_btn_centre, c_btn_d = st.columns([1, 1.2, 1])
    
    with c_btn_centre:
        # Le bouton est maintenant contenu uniquement dans la petite colonne du milieu
        if st.button("Accueil", use_container_width=True):
            st.session_state.page_actuelle = "accueil"
            st.rerun()

    st.divider()

    # --- 2. LE MENU DE NAVIGATION ---
    section = st.radio("Navigation", [
        "VUE D'ENSEMBLE",
        "GLISSEMENT 24&25",
        "CLASSIFICATION",
        "CONFORMITÉ",
        "À PROPOS"
    ], label_visibility="collapsed")

    st.divider()

    # --- 3. LES LOGOS DES 4 PARTENAIRES (En bas) ---
    st.markdown(
        f"<div style='font-size:0.75rem;color:{T['subtext']};text-align:left;margin-bottom:0.8rem;'>"
        f"PARTENAIRES</div>", unsafe_allow_html=True)
    
    # On crée 4 petites colonnes sur la même ligne
    c1, c2, c3, c4 = st.columns(4)
    
    try:
        # On place une image dans chaque colonne
        with c1: st.image(str(dossier_assets / "partenaire1.png"), use_container_width=True)
        with c2: st.image(str(dossier_assets / "partenaire2.png"), use_container_width=True)
        with c3: st.image(str(dossier_assets / "partenaire3.png"), use_container_width=True)
        with c4: st.image(str(dossier_assets / "partenaire4.png"), use_container_width=True)
    except Exception:
        st.caption("Images partenaires introuvables")

# ─────────────────────────────────────────────────────────────────
# CSS DYNAMIQUE
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""

<style>

    /* 1. Police globale AVEC support des Emojis (🇨🇲) */
    html, body, [class*="css"], .stApp, .block-container,
    h1, h2, h3, h4, p, div, span, label, button {{
        font-family: Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol" !important;
    }}
    
    /* 2. CORRECTION BUG ICÔNES (Toutes les flèches et menus Streamlit) */
    /* On protège l'en-tête (bouton ouvrir sidebar), le bouton fermer, et les expanders */
    header span,
    [data-testid="stSidebarCollapseButton"] span,
    details summary span, 
    span[class*="material"], 
    .st-icon {{
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
    }}

    /* --- Le reste de ton design original --- */
    .stApp {{ background-color:{T["bg"]} !important; color:{T["text"]} !important; }}
    .block-container {{ padding:2rem 3rem !important; background-color:{T["bg"]} !important; }}
    h1,h2,h3 {{ color:{T["text"]} !important; }}
    p, li, span, div {{ color:{T["text"]} !important; }}
    
    .metric-card {{
        background:{T["card"]}; border-radius:4px; padding:1.1rem 1.4rem;
        border-left:8px solid {T["accent"]}; margin-bottom:0.5rem;
        box-shadow:0 2px 8px rgba(0,0,0,0.12);
    }}
    .metric-value {{ font-size:1.9rem; font-weight:bold; color:{T["text"]}; }}
    .metric-label {{ font-size:0.82rem; color:{T["subtext"]}; margin-top:0.2rem; }}
    .section-header {{
        background:linear-gradient(90deg,{T["accent"]}22,transparent);
        border-left:4px solid {T["accent"]}; padding:0.6rem 1rem;
        border-radius:0 8px 8px 0; margin:1.5rem 0 1rem 0;
        font-size:1.1rem; font-weight:bold; color:{T["text"]};
    }}
    .pilier-badge {{
        display:inline-block; padding:3px 10px; border-radius:20px;
        font-size:0.75rem; font-weight:bold; margin:2px;
    }}
    .interp-box {{
        background:{T["card"]}; border-left:3px solid {T["accent"]};
        border-radius:0 8px 8px 0; padding:0.9rem 1.2rem; margin-top:0.5rem;
        font-size:0.88rem; line-height:1.6;
    }}
    [data-testid="stSidebar"] {{ background-color:{T["card"]} !important; }}
    [data-testid="stSidebar"] * {{ color:{T["text"]} !important; }}
    
    /* --- TABS (Onglets) EN PLEINE LARGEUR --- */
    div[data-baseweb="tab-list"] {{
        display: flex !important;
        width: 100% !important;
        gap: 0 !important;
    }}
    button[data-baseweb="tab"] {{
        flex: 1 !important;
        display: flex !important;
        justify-content: center !important;
        font-size: 1rem !important;
        padding-bottom: 1rem !important;
        color: {T["subtext"]} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {T["text"]} !important;
        border-bottom-color: {T["accent"]} !important;
        border-bottom-width: 3px !important;
        font-weight: bold !important;
    }}
    
    details summary {{
        color:{T["text"]} !important; background:{T["card"]} !important;
        border-radius:8px !important; padding:0.5rem 1rem !important;
    }}
    details {{ background:{T["card"]} !important; border-radius:8px !important;
               margin-bottom:4px !important; }}
    hr {{ border-color:{T["divider"]} !important; }}
    div[data-testid="stMetricValue"] {{ color:{T["text"]} !important; }}
    /* --- CORRECTION DES MENUS DÉROULANTS (SELECTBOX) --- */

    /* 1. Le fond de la boîte principale */
    div[data-baseweb="select"] > div {{
        background-color: {T["card"]} !important;
        color: {T["text"]} !important;
        border-color: {T["subtext"]} !important;
    }}

    /* 2. Le menu déroulant (la liste qui s'ouvre) */
    div[data-baseweb="popover"], 
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"] {{
        background-color: {T["card"]} !important;
    }}

    /* 3. Les options à l'intérieur */
    li[role="option"] {{
        color: {T["text"]} !important;
    }}

    /* 4. L'option survolée ou sélectionnée (devient bleue) */
    li[role="option"]:hover, 
    li[role="option"][aria-selected="true"] {{
        background-color: {T["accent"]} !important;
        color: #FFFFFF !important;
    }}

    /* 5. La petite flèche à droite */
    svg[data-icon="chevron-down"] {{
        fill: {T["text"]} !important;
    }}
   
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def chemin(rel):
    return BASE_DIR / rel

def parse_montant(s):
    if not s: return None
    v = re.sub(r"[\s\u00a0]","",str(s)); v = re.sub(r"[^\d.]","",v)
    return float(v) if v else None

def plotly_theme(height=420, title="", margin=None):
    """Kwargs de base pour update_layout — sans xaxis/yaxis pour éviter les conflits."""
    m = margin or dict(t=50, b=30, l=40, r=20)
    return dict(
        title=dict(text=title,
                   font=dict(family="Helvetica,Arial,sans-serif", size=14, color=T["text"])),
        paper_bgcolor=T["paper_bg"],
        plot_bgcolor=T["plot_bg"],
        font=dict(color=T["text"], family="Helvetica,Arial,sans-serif", size=11),
        height=height,
        margin=m,
    )

def ax():
    """Style axes commun."""
    return dict(gridcolor=T["grid"], zerolinecolor=T["grid"],
                tickfont=dict(family="Helvetica,Arial,sans-serif", size=10))

def interp(key, texte):
    """Bouton interprétation sous un graphique."""
    with st.expander("Ce qu'il faut retenir", expanded=False):
        st.markdown(
            f"<div class='interp-box'>{texte}</div>",
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def charger_audit():
    p = chemin("data/processed/audit_correspondances.json")
    if not p.exists(): return None
    with open(p, encoding="utf-8") as f: return json.load(f)

@st.cache_data
def charger_classification():
    p = chemin("data/processed/classification_snd30_v2.json")
    if not p.exists(): return None
    with open(p, encoding="utf-8") as f: return json.load(f)

@st.cache_data
def charger_articles(annee):
    p = chemin(f"data/processed/loi_{annee.replace('-','_')}_articles.json")
    if not p.exists(): return []
    with open(p, encoding="utf-8") as f: return json.load(f)

@st.cache_data
def charger_embeddings(modele):
    p = chemin(f"data/processed/embeddings_{modele}.npz")
    if not p.exists(): return None
    data = np.load(p, allow_pickle=True)
    return {k: data[k] for k in data.files}

@st.cache_data
def calculer_umap(modele):
    try:
        import umap as umap_lib
    except ImportError:
        return None
    from sklearn.preprocessing import StandardScaler

    emb_data = charger_embeddings(modele)
    if emb_data is None: return None

    arts_2023 = charger_articles("2023-2024")
    arts_2024 = charger_articles("2024-2025")

    e23 = np.array(emb_data["embeddings_2023"])
    e24 = np.array(emb_data["embeddings_2024"])
    X   = np.vstack([e23, e24])
    Xs  = StandardScaler().fit_transform(X)

    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15,
                             min_dist=0.1, random_state=42, verbose=False)
    X2d = reducer.fit_transform(Xs)
    n1  = len(e23)

    ids23 = emb_data["ids_2023"].tolist() if "ids_2023" in emb_data else [""]*n1
    ids24 = emb_data["ids_2024"].tolist() if "ids_2024" in emb_data else [""]*len(e24)

    return {
        "X1_2d":      X2d[:n1],
        "X2_2d":      X2d[n1:],
        "titres_2023": [a.get("titre") or "Sans titre" for a in arts_2023],
        "titres_2024": [a.get("titre") or "Sans titre" for a in arts_2024],
        "ids_2023":    ids23,
        "ids_2024":    ids24,
    }

# ─────────────────────────────────────────────────────────────────
# SECTION 0 : VUE D'ENSEMBLE
# ─────────────────────────────────────────────────────────────────

if section == "VUE D'ENSEMBLE":
    st.markdown(
        f"<h1>Analyse NLP des Lois de Finances du Cameroun</h1>"
        f"<p style='color:{T['subtext']}'>Audit sémantique · Classification SND30 · "
        f"Conformité budgétaire</p>", unsafe_allow_html=True)

    arts_2023 = charger_articles("2023-2024")
    arts_2024 = charger_articles("2024-2025")
    classif   = charger_classification()
    total_cp_val = sum(parse_montant(r.get("montant_cp")) or 0 for r in (classif or []))

    metriques = [
        (len(arts_2023),                 "Articles — Loi 2023-2024",      T["accent"]),
        (len(arts_2024),                 "Articles — Loi 2024-2025",      "#F4A261"),
        (len(classif) if classif else 0, "Programmes classifiés SND30",   "#2DC653"),
        (f"{total_cp_val/1e9:,.0f} Mrd","Total CP 2024-2025 (FCFA)",     "#9B5DE5"),
    ]
    for col, (val, label, couleur) in zip(st.columns(4), metriques):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color:{couleur}'>
                <div class='metric-value' style='color:{couleur}'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Création des 3 onglets (qui prendront automatiquement toute la largeur grâce au CSS)
    tab1, tab2, tab3 = st.tabs(["CONTEXTE DE L'ETUDE", "PRINCIPAUX RESULTATS", "RECOMMANDATIONS STRATEGIQUES"])

    # --- CONTENU DE L'ONGLET 1 ---
    with tab1:
        st.markdown("""
        <div style='padding: 1rem 0;'>
            <p><i>(Texte à remplacer)</i></p>
            <p>Cette étude s'inscrit dans le cadre de l'analyse des politiques publiques du Cameroun, en se focalisant sur la transition budgétaire entre les exercices 2023-2024 et 2024-2025. L'objectif principal est d'évaluer, grâce aux techniques avancées de Traitement du Langage Naturel (NLP), la cohérence entre le discours législatif porté par les Lois de Finances et les allocations réelles des crédits budgétaires.</p>
            <p>Dans un contexte économique marqué par des contraintes fortes (service de la dette, chocs externes), le gouvernement s'appuie sur la Stratégie Nationale de Développement (SND30) pour guider son action. Il était donc crucial de mesurer mathématiquement si les priorités énoncées dans le texte de loi se traduisent effectivement par des engagements financiers équivalents, ou si le poids des dépenses incompressibles crée une distorsion entre la vision stratégique et la réalité budgétaire.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- CONTENU DE L'ONGLET 2 ---
    with tab2:
        st.markdown("""
        <div style='padding: 1rem 0;'>
            <p><i>(Texte à remplacer)</i></p>
            <p>L'audit sémantique réalisé via les modèles CamemBERT et MiniLM révèle une stabilité globale du corpus législatif, avec néanmoins des modifications ciblées signalant de nouvelles priorités ou ajustements fiscaux.</p>
            <p>Sur le plan de la classification SND30, l'analyse montre que le pilier <b>"Gouvernance"</b> absorbe plus de la moitié du budget global (56%), massivement tiré par les obligations de la dette publique. Cependant, en isolant le budget discrétionnaire (hors incompressibles), la <b>"Transformation Structurelle"</b> reprend sa place de premier pilier d'investissement (environ 31%).</p>
            <p>Le test de corrélation de Spearman confirme un lien statistique significatif (p < 0.05) entre la fréquence d'apparition thématique d'un pilier et le volume de crédits qui lui est alloué. Toutefois, le pilier <b>"Développement Régional"</b> souffre d'un sous-financement chronique.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- CONTENU DE L'ONGLET 3 ---
    with tab3:
        st.markdown("""
        <div style='padding: 1rem 0;'>
            <p><i>(Texte à remplacer)</i></p>
            <p>Au vu de ces résultats, plusieurs leviers d'action peuvent être envisagés pour optimiser la formulation et l'exécution des prochaines Lois de Finances :</p>
            <ul>
                <li style='margin-bottom: 0.5rem;'><b>Rééquilibrage du Développement Régional :</b> Il est recommandé de consolider les multiples petits programmes régionaux en projets structurants mieux financés, afin de réduire l'écart entre l'ambition décentralisatrice et la réalité des crédits alloués.</li>
                <li style='margin-bottom: 0.5rem;'><b>Sanctuarisation du Capital Humain :</b> Bien que présent dans le discours, ce pilier mérite des mécanismes de financement innovants pour éviter que ses crédits ne soient évincés par le service de la dette.</li>
                <li style='margin-bottom: 0.5rem;'><b>Transparence Sémantique :</b> L'utilisation d'un vocabulaire plus standardisé d'une année sur l'autre faciliterait le suivi automatisé des politiques publiques.</li>
                <li style='margin-bottom: 0.5rem;'><b>Déploiement d'outils d'IA :</b> L'intégration pérenne de ce type de pipeline NLP au sein des directions budgétaires permettrait aux décideurs d'avoir un tableau de bord en temps réel lors des arbitrages de la loi de finances.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    

# ─────────────────────────────────────────────────────────────────
# SECTION 1 : GLISSEMENT SÉMANTIQUE
# ─────────────────────────────────────────────────────────────────

elif section == "GLISSEMENT 24&25":
    st.markdown(
        f"<h1>AUDIT SEMANTIQUE</h1>"
        f"<p style='color:{T['subtext']}'>Évolution du discours législatif "
        f"entre 2023-2024 et 2024-2025</p>", unsafe_allow_html=True)

    audit = charger_audit()
    if not audit: st.error("audit_correspondances.json introuvable."); st.stop()

    modele  = st.selectbox("MODELE D'EMBEDDING", ["camembert", "minilm"])
    corresp = audit.get(modele, [])
    scores  = [c["score"] for c in corresp if c.get("score",0) > 0]

    n_stable  = sum(1 for c in corresp if c.get("statut") == "STABLE")
    n_modifie = sum(1 for c in corresp if c.get("statut") == "MODIFIÉ")
    n_tres    = sum(1 for c in corresp if c.get("statut") == "TRÈS MODIFIÉ / NOUVEAU")
    n_nouveau = sum(1 for c in corresp if c.get("statut") == "NOUVEAU")

    for col, (val, label, couleur) in zip(st.columns(5), [
        (f"{np.mean(scores):.3f}", "Sim. moyenne",      T["text"]),
        (n_stable,                 "Stables ",          "#2DC653"),
        (n_modifie,                "Modifiés ",          "#F4A261"),
        (n_tres,                   "Très modifiés",    T["accent"]),
        (n_nouveau,                "Nouveaux",         "#9B5DE5"),
    ]):
        col.markdown(f"""
        <div class='metric-card' style='border-left-color:{couleur}'>
            <div class='metric-value' style='color:{couleur}'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        " DISTRIBUTION","UMAP SUPERPOSE",
        "ARTICLES MODIFIES","EXPLORATEUR",
    ])

    # ── Distribution ──
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores, nbinsx=40,
                                   marker_color="#457B9D", opacity=0.85))
        for x0,x1,c,lbl in [
            (0.90,1.0,"#2DC653","Stable"),
            (0.70,0.90,"#F4A261","Modifié"),
            (0,0.70,T["accent"],"Très modifié"),
        ]:
            fig.add_vrect(x0=x0,x1=x1,fillcolor=c,opacity=0.07,
                          annotation_text=lbl,annotation_position="top left")
        fig.add_vline(x=np.mean(scores),line_dash="dash",line_color=T["text"],
                      annotation_text=f"μ={np.mean(scores):.3f}")
        fig.update_layout(**plotly_theme(420,f"Distribution similarités — {modele.upper()}"))
        fig.update_xaxes(**ax(), title_text="Score de similarité")
        fig.update_yaxes(**ax(), title_text="Nombre d'articles")
        st.plotly_chart(fig, use_container_width=True)
        interp("interp_distrib",
               f"La distribution des similarités cosinus ({modele.upper()}) montre "
               f"la similitude sémantique entre chaque article de 2023-2024 et son "
               f"meilleur correspondant dans la loi 2024-2025. Une similarité moyenne "
               f"de {np.mean(scores):.3f} indique une préservation globale du contenu "
               f"avec des modifications notables. Les articles sous 0.70 (zone rouge) "
               f"ont subi des ruptures de discours significatives.")

    # ── UMAP ──
    with tab2:
        st.markdown(
            f"<p style='color:{T['subtext']}'>○ = 2023-2024 · □ = 2024-2025 · "
            f"Même couleur = même titre dans les deux lois</p>",
            unsafe_allow_html=True)

        with st.spinner("Calcul UMAP (1-2 min)..."):
            umap_data = calculer_umap(modele)

        if umap_data is None:
            st.warning("Installez umap-learn : `pip install umap-learn`")
        else:
            X1,X2   = umap_data["X1_2d"], umap_data["X2_2d"]
            t23,t24 = umap_data["titres_2023"], umap_data["titres_2024"]
            ids23,ids24 = umap_data["ids_2023"], umap_data["ids_2024"]

            tous_titres = sorted(set(t23+t24))
            PALETTE = ["#E63946","#F4A261","#2DC653","#457B9D","#9B5DE5",
                       "#00BBF9","#F15BB5","#FEE440","#00F5D4","#FB5607",
                       "#8338EC","#3A86FF","#FF006E","#FFBE0B","#06D6A0"]
            cmap = {t: PALETTE[i%len(PALETTE)] for i,t in enumerate(tous_titres)}

            titre_filtre = st.selectbox("Filtrer par titre", ["Tous"]+tous_titres)

            fig_u = go.Figure()
            for titre in tous_titres:
                if titre_filtre != "Tous" and titre != titre_filtre: continue
                couleur = cmap[titre]
                idx1 = [i for i,t in enumerate(t23) if t==titre]
                idx2 = [i for i,t in enumerate(t24) if t==titre]
                if idx1:
                    fig_u.add_trace(go.Scatter(
                        x=X1[idx1,0], y=X1[idx1,1], mode="markers",
                        name=f"2023·{titre[:22]}", legendgroup=titre, showlegend=True,
                        marker=dict(color=couleur,size=7,symbol="circle",
                                    opacity=0.75,line=dict(width=0.5,color="white")),
                        hovertemplate=f"<b>2023-2024</b><br>{titre}<br>%{{customdata}}<extra></extra>",
                        customdata=[str(ids23[i])[:40] for i in idx1],
                    ))
                if idx2:
                    fig_u.add_trace(go.Scatter(
                        x=X2[idx2,0], y=X2[idx2,1], mode="markers",
                        name=f"2024·{titre[:22]}", legendgroup=titre, showlegend=False,
                        marker=dict(color=couleur,size=7,symbol="square",
                                    opacity=0.75,line=dict(width=0.5,color="white")),
                        hovertemplate=f"<b>2024-2025</b><br>{titre}<br>%{{customdata}}<extra></extra>",
                        customdata=[str(ids24[i])[:40] for i in idx2],
                    ))
            fig_u.update_layout(
                **plotly_theme(580,f"UMAP superposé — {modele.upper()}"),
                legend=dict(bgcolor="rgba(0,0,0,0.1)",font=dict(size=9)),
            )
            fig_u.update_xaxes(**ax(), title_text="UMAP 1")
            fig_u.update_yaxes(**ax(), title_text="UMAP 2")
            st.plotly_chart(fig_u, use_container_width=True)
            interp("interp_umap",
                   "La projection UMAP réduit les embeddings à 2 dimensions. "
                   "Les points proches (cercles et carrés de même couleur) indiquent "
                   "des articles au contenu préservé entre les deux lois. "
                   "Les points isolés signalent des articles nouveaux ou très modifiés. "
                   "Les clusters denses révèlent les blocs thématiques stables.")

    # ── Articles modifiés ──
    with tab3:
        tres_mod = sorted(
            [c for c in corresp if c.get("statut") in
             ("TRÈS MODIFIÉ / NOUVEAU","NOUVEAU") and c.get("score",0)>0],
            key=lambda x: x["score"])
        seuil_f = st.slider("Score maximum",0.0,1.0,0.70,0.01)
        filtres = [c for c in tres_mod if c["score"]<=seuil_f]
        st.markdown(f"**{len(filtres)} articles** avec score ≤ {seuil_f}")

        fig2 = go.Figure(go.Bar(
            x=[c["score"] for c in filtres[:20]],
            y=[(c.get("ref_2023") or "NOUVEAU")[:35] for c in filtres[:20]],
            orientation="h", marker_color=T["accent"],
            text=[f"{c['score']:.3f}" for c in filtres[:20]],
            textposition="outside",
        ))
        fig2.update_layout(**plotly_theme(500,"Top 20 articles les plus modifiés",
                           margin=dict(l=250,r=70,t=50,b=30)))
        fig2.update_xaxes(**ax(), title_text="Score de similarité")
        #  update_yaxes séparé pour éviter le conflit avec plotly_theme
        fig2.update_yaxes(autorange="reversed",
                          tickfont=dict(size=9,family="Helvetica,Arial,sans-serif"),
                          gridcolor=T["grid"])
        st.plotly_chart(fig2, use_container_width=True)
        interp("interp_modifies",
               "Ce graphique liste les articles ayant le plus divergé entre les deux lois. "
               "Un score bas (proche de 0) indique une quasi-réécriture. Ces articles "
               "représentent les zones de rupture législative et méritent une "
               "analyse qualitative approfondie.")

    # ── Explorateur ──
    with tab4:
        query = st.text_input("Filtrer par référence", placeholder="ex: Article VINGT")
        filtres_exp = (
            [c for c in corresp
             if query.lower() in str(c.get("ref_2023","")).lower()
             or query.lower() in str(c.get("ref_2024","")).lower()]
            if query else corresp[:50]
        )
        for c in filtres_exp[:30]:
            score  = c.get("score",0)
            statut = c.get("statut","?")
            coul_s = {"STABLE":"#2DC653","MODIFIÉ":"#F4A261",
                      "TRÈS MODIFIÉ / NOUVEAU":T["accent"],"NOUVEAU":"#9B5DE5"}.get(statut,"#888")
            with st.expander(
                f"[{score:.3f}] {c.get('ref_2023','N/A')[:45]} → {c.get('ref_2024','N/A')[:45]}"):
                col_g,col_d = st.columns(2)
                with col_g:
                    st.markdown("**Loi 2023-2024**")
                    st.caption(c.get("texte_2023","")[:300])
                with col_d:
                    st.markdown("**Loi 2024-2025**")
                    st.caption(c.get("texte_2024","")[:300])
                st.markdown(
                    f"<span class='pilier-badge' style='background:{coul_s};color:#000'>"
                    f"{statut}</span>  Score : {score:.4f}", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SECTION 2 : CLASSIFICATION SND30
# ─────────────────────────────────────────────────────────────────

elif section == "CLASSIFICATION":
    st.markdown(
        f"<h1>Classification SND30</h1>"
        f"<p style='color:{T['subtext']}'>182 programmes budgétaires classifiés "
        f"par pilier SND30</p>", unsafe_allow_html=True)

    classif = charger_classification()
    if not classif: st.error("classification_snd30_v2.json introuvable."); st.stop()

    cp_pp = {p:0.0 for p in PILIERS}; n_pp = Counter()
    for r in classif:
        cp = parse_montant(r.get("montant_cp")) or 0
        if r.get("pilier") in cp_pp: cp_pp[r["pilier"]] += cp; n_pp[r["pilier"]] += 1
    total_cp = sum(cp_pp.values())

    for col, p in zip(st.columns(4), PILIERS):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color:{COULEURS_PILIERS[p]}'>
                <div class='metric-value' style='color:{COULEURS_PILIERS[p]}'>
                    {cp_pp[p]/1e9:.1f} Mrd</div>
                <div class='metric-label'>{p}<br>
                {n_pp[p]} programmes · {100*cp_pp[p]/total_cp:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    tab1,tab2,tab3 = st.tabs(["VUE GLOBALE","TREEMAP","PROGRAMMES"])

    with tab1:
        col_g,col_d = st.columns(2)
        with col_g:
            fig = go.Figure(go.Pie(
                labels=PILIERS, values=[cp_pp[p]/1e9 for p in PILIERS],
                marker=dict(colors=[COULEURS_PILIERS[p] for p in PILIERS]),
                hole=0.45, textinfo="label+percent", textfont=dict(size=11),
            ))
            fig.update_layout(**plotly_theme(380,"CP par pilier (Mrd FCFA)"),showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            interp("interp_pie_classif",
                   "La Gouvernance représente 56% du budget total, principalement "
                   "en raison de la dette publique (2 065 Mrd FCFA). En excluant "
                   "les dépenses incompressibles, la Transformation Structurelle "
                   "devient le 1er pilier (31.6%), cohérent avec les priorités SND30.")

        with col_d:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Programmes (%)", x=PILIERS,
                y=[100*n_pp[p]/len(classif) for p in PILIERS],
                marker_color=[COULEURS_PILIERS[p] for p in PILIERS], opacity=0.9,
            ))
            fig2.add_trace(go.Bar(
                name="CP (%)", x=PILIERS,
                y=[100*cp_pp[p]/total_cp for p in PILIERS],
                marker_color=[COULEURS_PILIERS[p] for p in PILIERS], opacity=0.4,
                marker_pattern_shape="/",
            ))
            fig2.update_layout(**plotly_theme(380,"Fréquence vs CP"),
                               barmode="group",legend=dict(bgcolor="rgba(0,0,0,0.1)"))
            fig2.update_xaxes(**ax())
            fig2.update_yaxes(**ax(), title_text="%")
            st.plotly_chart(fig2, use_container_width=True)
            interp("interp_freq_cp",
                   "La Gouvernance est sur-représentée financièrement à cause de la dette. "
                   "Le Développement Régional présente le plus grand déséquilibre : "
                   "23% des programmes mais seulement 10% des crédits.")

    with tab2:
        labels,parents,values,colors = [],[],[],[]
        labels.append("Budget 2024-2025"); parents.append("")
        values.append(0); colors.append(T["bg"])
        for p in PILIERS:
            labels.append(p); parents.append("Budget 2024-2025")
            values.append(cp_pp[p]/1e6); colors.append(COULEURS_PILIERS[p])
        for r in classif:
            cp = parse_montant(r.get("montant_cp")) or 0
            if not cp: continue
            lbl = f"{r.get('code','')} · {r.get('libelle','')[:35]}"
            labels.append(lbl); parents.append(r.get("pilier",""))
            values.append(cp/1e6)
            colors.append(COULEURS_PILIERS.get(r.get("pilier",""),"#888"))
        fig3 = go.Figure(go.Treemap(
            labels=labels, parents=parents, values=values,
            marker=dict(colors=colors,line=dict(width=1,color=T["bg"])),
            textinfo="label",
            hovertemplate="<b>%{label}</b><br>CP : %{value:.0f} M FCFA<extra></extra>",
            maxdepth=2,
        ))
        fig3.update_layout(
            paper_bgcolor=T["paper_bg"],
            font=dict(color=T["text"],family="Helvetica,Arial,sans-serif"),
            height=550, margin=dict(t=10,l=5,r=5,b=5),
        )
        st.plotly_chart(fig3, use_container_width=True)
        interp("interp_treemap",
               "Le treemap montre la part relative de chaque programme dans le budget. "
               "Quelques méga-programmes (dette, routes, eau/énergie) dominent chaque pilier. "
               "La majorité des programmes reçoivent des dotations modestes.")

    with tab3:
        pf  = st.selectbox("Filtrer par pilier",["Tous"]+PILIERS)
        tri = st.selectbox("Trier par",["CP décroissant","Score décroissant"])
        filtres = classif if pf=="Tous" else [r for r in classif if r.get("pilier")==pf]
        filtres = sorted(filtres,
            key=lambda r: parse_montant(r.get("montant_cp")) or 0
            if tri=="CP décroissant" else r.get("score_dominant",0), reverse=True)
        for r in filtres[:40]:
            cp   = parse_montant(r.get("montant_cp")) or 0
            coul = COULEURS_PILIERS.get(r.get("pilier",""),"#888")
            with st.expander(
                f"[{r.get('code','')}] {r.get('libelle','')[:60]} — {cp/1e6:,.0f} M FCFA"):
                c1,c2 = st.columns([2,1])
                with c1:
                    st.markdown(f"**Objectif :** {r.get('objectif','N/A')}")
                    st.markdown(f"**Indicateur :** {r.get('indicateur','N/A')}")
                with c2:
                    st.markdown(
                        f"<span class='pilier-badge' style='background:{coul};color:#000'>"
                        f"{r.get('pilier','')}</span>", unsafe_allow_html=True)
                    st.metric("Score",f"{r.get('score_dominant',0):.3f}")
                    st.metric("CP",f"{cp/1e6:,.0f} M FCFA")

# ─────────────────────────────────────────────────────────────────
# SECTION 3 : CONFORMITÉ
# ─────────────────────────────────────────────────────────────────

elif section == "CONFORMITÉ":
    st.markdown(
        f"<h1>Conformité Discours / Budget</h1>"
        f"<p style='color:{T['subtext']}'>Alignement entre fréquence thématique "
        f"et crédits alloués</p>", unsafe_allow_html=True)

    classif = charger_classification()
    if not classif: st.error("classification_snd30_v2.json introuvable."); st.stop()

    mode  = st.radio("Périmètre",
        ["Budget total","Budget discrétionnaire (hors incompressibles)"],horizontal=True)
    progs = classif if "total" in mode else [
        r for r in classif if r.get("code","") not in CODES_INCOMPRESSIBLES]

    cp_pp = {p:0.0 for p in PILIERS}; n_pp = Counter()
    for r in progs:
        cp = parse_montant(r.get("montant_cp")) or 0
        if r.get("pilier") in cp_pp: cp_pp[r["pilier"]] += cp; n_pp[r["pilier"]] += 1
    total_n  = len(progs); total_cp = sum(cp_pp.values())
    freq_pct = {p:100*n_pp[p]/total_n   for p in PILIERS}
    cp_pct   = {p:100*cp_pp[p]/total_cp for p in PILIERS}
    ecarts   = {p:cp_pct[p]-freq_pct[p] for p in PILIERS}

    from scipy import stats as sp_stats
    rho,pval = sp_stats.spearmanr(
        [freq_pct[p] for p in PILIERS],[cp_pct[p] for p in PILIERS])

    for col,(val,label) in zip(st.columns(3),[
        (f"{rho:.3f}",  "ρ de Spearman"),
        (f"{pval:.3f}", f"p-value · {' Sig.' if pval<0.05 else ' Non sig.'}"),
        (f"{max(abs(ecarts[p]) for p in PILIERS):.1f}%","Écart max discours/budget"),
    ]):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    tab1,tab2,tab3 = st.tabs(["ÉCARTS","SPEARMAN","TABLEAU"])

    with tab1:
        col_g,col_d = st.columns(2)
        with col_g:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Fréquence (%)", x=PILIERS,
                y=list(freq_pct.values()),
                marker_color=[COULEURS_PILIERS[p] for p in PILIERS], opacity=0.9,
                text=[f"{v:.1f}%" for v in freq_pct.values()], textposition="outside",
            ))
            fig.add_trace(go.Bar(
                name="CP (%)", x=PILIERS, y=list(cp_pct.values()),
                marker_color=[COULEURS_PILIERS[p] for p in PILIERS], opacity=0.4,
                marker_pattern_shape="/",
                text=[f"{v:.1f}%" for v in cp_pct.values()], textposition="outside",
            ))
            fig.update_layout(**plotly_theme(420,"Fréquence vs CP"),
                              barmode="group",legend=dict(bgcolor="rgba(0,0,0,0.1)"))
            fig.update_xaxes(**ax())
            fig.update_yaxes(**ax(), title_text="%")
            st.plotly_chart(fig, use_container_width=True)
            interp("interp_freq_conf",
                   "Un alignement parfait verrait les deux barres à la même hauteur. "
                   "Les écarts révèlent des choix politiques implicites : certains piliers "
                   "sont mis en avant dans le discours mais sous-dotés financièrement.")

        with col_d:
            ecart_vals = [ecarts[p] for p in PILIERS]
            fig2 = go.Figure(go.Bar(
                x=ecart_vals, y=PILIERS, orientation="h",
                marker_color=["#2DC653" if e>=0 else T["accent"] for e in ecart_vals],
                text=[f"{e:+.1f}%" for e in ecart_vals], textposition="outside",
            ))
            fig2.add_vline(x=0,line_color=T["text"],line_width=1)
            fig2.update_layout(**plotly_theme(420,"Écart CP% − Fréquence%",
                               margin=dict(l=210,r=80,t=50,b=30)))
            fig2.update_xaxes(**ax(), title_text="Points de pourcentage")
            # update_yaxes séparé — pas de conflit
            fig2.update_yaxes(tickfont=dict(size=10,family="Helvetica,Arial,sans-serif"),
                              gridcolor=T["grid"])
            st.plotly_chart(fig2, use_container_width=True)
            interp("interp_ecart",
                   "Vert = pilier sur-financé par rapport à son poids discursif. "
                   "Rouge = pilier sous-financé. Le Développement Régional présente "
                   "le plus grand déséquilibre négatif : 23% des programmes, 10% des crédits.")

        st.markdown("<div class='section-header'>Synthèse</div>",
                    unsafe_allow_html=True)
        for p in sorted(PILIERS, key=lambda x: abs(ecarts[x]), reverse=True):
            e = ecarts[p]
            icone = "🟢" if e>5 else "🔴" if e<-5 else "🟡"
            msg = (f"**sur-financé** (+{e:.1f}%)" if e>5
                   else f"**sous-financé** ({e:.1f}%)" if e<-5
                   else f"alignement relatif ({e:+.1f}%)")
            st.markdown(f"{icone} **{p}** : {freq_pct[p]:.1f}% des programmes, "
                        f"{cp_pct[p]:.1f}% du budget → {msg}")

    with tab2:
        freq_v = [freq_pct[p] for p in PILIERS]
        cp_v   = [cp_pct[p]   for p in PILIERS]
        fig3   = go.Figure()
        fig3.add_trace(go.Scatter(
            x=freq_v, y=cp_v, mode="markers+text",
            text=PILIERS, textposition="top center", textfont=dict(size=11),
            marker=dict(color=[COULEURS_PILIERS[p] for p in PILIERS],
                        size=20,line=dict(width=2,color=T["text"])),
            hovertemplate="<b>%{text}</b><br>Fréq:%{x:.1f}%<br>CP:%{y:.1f}%<extra></extra>",
        ))
        m_,b_ = np.polyfit(freq_v,cp_v,1)
        xl = np.linspace(min(freq_v)-2,max(freq_v)+2,50)
        fig3.add_trace(go.Scatter(x=xl,y=m_*xl+b_,mode="lines",
            line=dict(dash="dash",color="rgba(150,150,150,0.5)"),showlegend=False))
        fig3.update_layout(
            **plotly_theme(460,f"Spearman — ρ={rho:.3f}  p={pval:.3f}"),
            annotations=[dict(
                x=0.02,y=0.97,xref="paper",yref="paper",
                text=f"ρ = {rho:.3f}<br>p = {pval:.3f}<br>"
                     f"{'Significatif' if pval<0.05 else ' Non significatif'}",
                showarrow=False,bgcolor=T["card"],bordercolor=T["divider"],
                borderwidth=1,font=dict(size=12),align="left",
            )],
        )
        fig3.update_xaxes(**ax(), title_text="Fréquence (% programmes)")
        fig3.update_yaxes(**ax(), title_text="CP (% budget)")
        st.plotly_chart(fig3, use_container_width=True)
        sig_txt = (
            "La corrélation est significative (p < 0.05) : les piliers les plus représentés "
            "reçoivent proportionnellement plus de crédits — alignement discours/budget confirmé."
            if pval < 0.05 else
            f"La corrélation (ρ={rho:.3f}) n'est pas significative (p={pval:.3f} ≥ 0.05). "
            f"Il existe un <b>désalignement discours/budget</b> : les piliers qui concentrent "
            f"le plus de programmes ne reçoivent pas proportionnellement plus de crédits. "
            f"Les choix budgétaires obéissent à une logique différente de la logique programmatique."
        )
        interp("interp_spearman", sig_txt)

    with tab3:
        import pandas as pd
        df = pd.DataFrame([{
            "Pilier":        p,
            "Programmes":    n_pp[p],
            "Fréquence (%)": round(freq_pct[p],1),
            "CP (M FCFA)":   round(cp_pp[p]/1e6,0),
            "CP (%)":        round(cp_pct[p],1),
            "Écart (pp)":    round(ecarts[p],1),
            "Statut": ("Sur-financé" if ecarts[p]>5
                       else "Sous-financé" if ecarts[p]<-5 else "Aligné"),
        } for p in PILIERS])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇Télécharger CSV",
            data=df.to_csv(index=False,sep=";").encode("utf-8"),
            file_name="conformite_snd30.csv", mime="text/csv")
        interp("interp_tableau",
               "Ce tableau synthétise les résultats de l'analyse de conformité. "
               "L'écart en points de pourcentage (pp) mesure le désalignement. "
               "Positif = sur-financé. Négatif = sous-financé. "
               "Ces données sont exportables pour une analyse complémentaire.")
        
# ─────────────────────────────────────────────────────────────────
# SECTION 5 : À PROPOS
# ─────────────────────────────────────────────────────────────────

elif section == "À PROPOS":
    st.markdown(
        f"<h1>L'Équipe du Projet</h1>"
        f"<p style='color:{T['subtext']}'>Conception, Développement et Analyse</p>", 
        unsafe_allow_html=True
    )
    st.divider()

    # --- CONFIGURATION DES MEMBRES ---
    # Remplis les infos ici. Pour les images, mets-les dans src/dashboard/assets/
    MEMBRES = [
        {
            "nom": "DOMEVENOU Wisdom",
            "role": "Data Scientist Full Stack",
            "photo": "membre1.png", # Nom du fichier dans le dossier assets
            "linkedin": "https://www.linkedin.com/in/komla-wisdom-domevenou/"
        },
        {
            "nom": "MOUSSAVOU MOUSSAVOU Lloyd",
            "role": "Data Engineer",
            "photo": "membre2.jpeg",
            "linkedin": "https://www.linkedin.com/in/aaron-moussavou-451ab9343/"
        },
        {
            "nom": "ALATSA Giovanni",
            "role": "Data Analyst",
            "photo": "membre3.jpeg",
            "linkedin": "https://www.linkedin.com/in/geovanel-dongho-alatsa-1b64281b1/"
        },
        {
            "nom": "OGNIMBA Sadri",
            "role": "Data analyst",
            "photo": "membre4.png",
            "linkedin": "https://www.linkedin.com/in/ton-profil-4/"
        },
    ]

    # On définit le chemin vers les images (au même endroit que le script)
    dossier_assets = Path(__file__).resolve().parent / "assets"

    # Création des 4 colonnes
    cols = st.columns(4)

    # On boucle sur les 4 membres et les 4 colonnes en même temps
    for col, membre in zip(cols, MEMBRES):
        with col:
            # 1. PHOTO (Rond ou Carré selon l'image d'origine)
            chemin_photo = dossier_assets / membre["photo"]
            
            # Si la photo existe, on l'affiche, sinon on met une icône générique
            if chemin_photo.exists():
                st.image(str(chemin_photo), use_container_width=True)
            else:
                # Placeholder si pas d'image (une icône utilisateur grise)
                st.markdown(f"""
                <div style='background:{T["card"]};height:150px;border-radius:10px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:3rem;color:{T["subtext"]};margin-bottom:10px;'>
                    👤
                </div>""", unsafe_allow_html=True)

            # 2. INFOS DU MEMBRE
            st.markdown(f"""
            <div style='text-align:center; margin-top:5px;'>
                <div style='font-weight:bold; font-size:1rem; color:{T["text"]}'>
                    {membre["nom"]}
                </div>
                <div style='font-size:0.85rem; color:{T["accent"]}; margin-bottom:10px;'>
                    {membre["role"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3. BOUTON LINKEDIN (Centré)
            st.link_button("LinkedIn 🔗", membre["linkedin"], use_container_width=True)

    st.divider()
    
    # Petit pied de page sympa
    st.markdown(
        f"<div style='text-align:center;color:{T['subtext']};font-size:0.8rem;margin-top:2rem'>"
        f"Projet réalisé dans le cadre du cycle ISE3 à l'ISSEA Yaoundé<br>"
        f"© 2025 · NLP Finances Cameroun"
        f"</div>", unsafe_allow_html=True)
    