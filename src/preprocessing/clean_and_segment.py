"""
clean_and_segment_v3.py
=======================
Nettoyage et segmentation des Lois de Finances du Cameroun.

Produit deux fichiers par loi :
  1. *_articles.json             → segments textuels hiérarchisés (Objectifs 1 & 2)
  2. *_lignes_budgetaires.json   → lignes CP/AE extraites des tableaux (Objectif 3)

Changement v3 :
  - Extraction budgétaire par POSITION de colonnes (pas par nom d'en-tête)
    car les tableaux du PDF sont très irréguliers
  - Les 2 dernières colonnes non-vides = AE et CP
  - Tout le texte intermédiaire = enrichissement sémantique (libellé + objectif + indicateur)
  - Filtre : ligne retenue si au moins une valeur monétaire détectée
"""

import json
import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FICHIERS = [
    {
        "entree": EXTRACTED_DIR / "loi_finances_2023_2024.json",
        "label":  "LOI_2023_2024",
        "annee":  "2023-2024",
    },
    {
        "entree": EXTRACTED_DIR / "loi_finances_2024_2025.json",
        "label":  "LOI_2024_2025",
        "annee":  "2024-2025",
    },
]

# ─────────────────────────────────────────────────────────────────
# PATTERNS STRUCTURE JURIDIQUE
# ─────────────────────────────────────────────────────────────────

RE_PARTIE = re.compile(
    r"^\s*(PREMI[EÈ]RE|DEUXI[EÈ]ME|TROISI[EÈ]ME|QUATRI[EÈ]ME|CINQUI[EÈ]ME|SECONDE?)\s+PARTIE\s*$",
    re.IGNORECASE,
)
RE_PARTIE2 = re.compile(r"^\s*(\w+)\s+PARTIE\s*$", re.IGNORECASE)
RE_TITRE = re.compile(
    r"^\s*TITRE\s+(PREMIER|DEUXI[EÈ]ME|TROISI[EÈ]ME|QUATRI[EÈ]ME|CINQUI[EÈ]ME|SIXI[EÈ]ME"
    r"|[IVX]+|\w+[ÈE]ME|\w+I[EÈ]ME)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
RE_CHAPITRE = re.compile(
    r"^\s*CHAPITRE\s+(PREMIER|DEUXI[EÈ]ME|TROISI[EÈ]ME|QUATRI[EÈ]ME|CINQUI[EÈ]ME"
    r"|UNIQUE|[IVX]+|\d+|\w+[ÈE]ME|\w+I[EÈ]ME)\s*[:\.\-]?\s*$",
    re.IGNORECASE,
)
RE_ARTICLE = re.compile(
    r"^\s*ARTICLE\s+(.+?)[.\-]\s*[-–—]?\s*(.*)$",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────

def is_noise(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) <= 2:
        return True
    if re.match(r"^\d{1,3}$", stripped):
        return True
    noise_patterns = [
        r"^PRESIDEN[A-Z\s]+$",
        r"^SECRETAR[A-Z\s]+$",
        r"^CERTIFIE[A-Z\s']+$",
        r"^[-·/\\,.'`\s]+$",
        r"^[A-Za-z\s'\-\.]{1,4}$",
        r"^REPUBLIQUE\s+DU\s+CAMEROUN",
        r"^REPUBLIC\s+OF\s+CAMEROON",
        r"^Paix\s*[-–]\s*Travail",
        r"^Peace\s*[-–]\s*Work",
    ]
    for p in noise_patterns:
        if re.match(p, stripped, re.IGNORECASE):
            return True
    return False


def clean_montant(valeur: str) -> float | None:
    """Convertit une chaîne monétaire en float."""
    if not valeur:
        return None
    # Nettoyer les caractères parasites OCR (m, c, r, p, w isolés)
    v = re.sub(r"^[a-zA-Z]\s*\n?", "", valeur.strip())
    v = re.sub(r"[\s\u00a0\u202f]", "", v)
    v = v.replace(",", ".").replace("'", "")
    v = re.sub(r"[^\d.]", "", v)
    try:
        val = float(v) if v else None
        # Filtre : les montants budgétaires sont > 0 et < 10 000 milliards
        if val and 0 < val < 1e13:
            return val
        return None
    except ValueError:
        return None


def est_montant(texte: str) -> bool:
    """Vérifie si une cellule contient un montant monétaire."""
    if not texte or not texte.strip():
        return False
    # Cherche des chiffres groupés (avec ou sans espaces)
    cleaned = re.sub(r"^[a-zA-Z\s\n]+", "", texte.strip())
    return bool(re.search(r"\d[\d\s]{3,}", cleaned))


def nettoyer_texte_cellule(texte: str) -> str:
    """Nettoie une cellule : supprime artefacts OCR, normalise les espaces."""
    if not texte:
        return ""
    # Remplacer les sauts de ligne par des espaces
    t = texte.replace("\n", " ")
    # Supprimer les caractères OCR parasites isolés (m, c, r, p, w en début)
    t = re.sub(r"^\s*[mcwrp]\s+", "", t)
    # Normaliser les espaces multiples
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def est_ligne_vide(row: list) -> bool:
    return all(str(c).strip() == "" for c in row)


def est_ligne_entete(row: list) -> bool:
    """Détecte les lignes d'en-tête (No, CODE, LIBELLE, OBJECTIF, etc.)."""
    entete_keywords = {"no", "n°", "code", "libelle", "libellé", "objectif",
                       "indicateur", "ae", "cp", "programme"}
    cells_non_vides = [str(c).strip().lower() for c in row if str(c).strip()]
    matches = sum(1 for c in cells_non_vides if any(kw in c for kw in entete_keywords))
    return matches >= 2

# ─────────────────────────────────────────────────────────────────
# EXTRACTION BUDGÉTAIRE PAR POSITION
# ─────────────────────────────────────────────────────────────────

def extraire_lignes_budgetaires(pages: list, source: str, annee: str) -> list:
    """
    Stratégie positionnelle :
    - Les 2 dernières colonnes non-vides d'une ligne = AE et CP
    - Tout le texte intermédiaire (colonnes 1 à N-2) = enrichissement sémantique
    - Une ligne est retenue si au moins une valeur monétaire est détectée
    """
    lignes = []
    compteur = 0
    chapitre_courant = ""

    for page in pages:
        for table in page.get("tables", []):
            if len(table) < 1:
                continue

            nb_cols = max(len(row) for row in table)
            if nb_cols < 2:
                continue

            for row in table:
                if est_ligne_vide(row):
                    continue
                if est_ligne_entete(row):
                    continue

                # Détecter les lignes de chapitre (ex: "CHAPITRE 01 - PRESIDENCE...")
                row_text = " ".join(str(c) for c in row if str(c).strip())
                if re.search(r"CHAPITRE\s+\d+\s*[-–]", row_text, re.IGNORECASE):
                    chapitre_courant = nettoyer_texte_cellule(row_text)
                    continue

                # Pad la ligne à nb_cols
                row_padded = list(row) + [""] * (nb_cols - len(row))

                # Identifier les indices des colonnes avec montants (de droite à gauche)
                indices_montants = []
                for idx in range(len(row_padded) - 1, -1, -1):
                    cell = str(row_padded[idx]).strip()
                    if est_montant(cell):
                        indices_montants.append(idx)
                    if len(indices_montants) == 2:
                        break

                if not indices_montants:
                    continue  # pas de montant → ignorer

                # Extraire AE et CP selon le nombre de montants trouvés
                if len(indices_montants) >= 2:
                    idx_cp = indices_montants[0]   # colonne la plus à droite = CP
                    idx_ae = indices_montants[1]   # suivante = AE
                    cp_raw = str(row_padded[idx_cp])
                    ae_raw = str(row_padded[idx_ae])
                    idx_texte_fin = min(idx_cp, idx_ae)
                elif len(indices_montants) == 1:
                    idx_cp = indices_montants[0]
                    cp_raw = str(row_padded[idx_cp])
                    ae_raw = ""
                    idx_texte_fin = idx_cp
                else:
                    continue

                montant_cp = clean_montant(cp_raw)
                montant_ae = clean_montant(ae_raw)

                # Extraire le texte sémantique (toutes les colonnes avant les montants)
                cellules_texte = [
                    nettoyer_texte_cellule(str(row_padded[i]))
                    for i in range(idx_texte_fin)
                    if str(row_padded[i]).strip()
                ]

                # Séparer code (si numérique court) du reste
                code = ""
                texte_parts = []
                for part in cellules_texte:
                    if re.match(r"^\d{1,4}$", part) and not code:
                        code = part
                    else:
                        texte_parts.append(part)

                texte_enrichi = " | ".join(texte_parts) if texte_parts else ""

                if not texte_enrichi and not code:
                    continue

                compteur += 1
                lignes.append({
                    "type":       "ligne_budgetaire",
                    "id":         f"{annee}_{compteur:04d}",
                    "annee":      annee,
                    "chapitre":   chapitre_courant,
                    "code":       code,
                    "texte":      texte_enrichi,
                    "montant_cp": montant_cp,
                    "montant_ae": montant_ae,
                    "page":       page["page_number"],
                    "source":     source,
                })

    return lignes

# ─────────────────────────────────────────────────────────────────
# PARSEUR HIÉRARCHIQUE  →  *_articles.json
# ─────────────────────────────────────────────────────────────────

def parser_articles(pages: list, source: str, annee: str) -> list:
    toutes_lignes = []
    for page in pages:
        for bloc in page["text_blocks"]:
            if not is_noise(bloc):
                toutes_lignes.append(bloc.strip())

    records = []
    current_partie    = None
    current_titre     = None
    current_chapitre  = None
    current_article   = None
    current_paragraphs = []

    def is_structural(line):
        return (RE_PARTIE.match(line) or RE_PARTIE2.match(line)
                or RE_TITRE.match(line) or RE_CHAPITRE.match(line)
                or RE_ARTICLE.match(line))

    def get_desc(lines, start_idx):
        desc = []
        j = start_idx + 1
        while j < len(lines) and len(desc) < 2:
            l = lines[j].strip()
            if l and not is_noise(l) and not is_structural(l):
                desc.append(l)
            elif is_structural(l):
                break
            j += 1
        return " ".join(desc)

    def flush_article():
        if current_article is None:
            return
        para_text = " ".join(current_paragraphs).strip()
        if not para_text:
            return
        paras = re.split(r"\s*(?=\(\d+\))", para_text)
        paras = [p.strip() for p in paras if len(p.strip()) >= 30]
        if not paras:
            paras = [para_text]
        for para in paras:
            records.append({
                "type":      "article",
                "annee":     annee,
                "partie":    current_partie,
                "titre":     current_titre,
                "chapitre":  current_chapitre,
                "reference": current_article,
                "texte":     para,
                "nb_mots":   len(para.split()),
                "source":    source,
            })

    i = 0
    while i < len(toutes_lignes):
        line = toutes_lignes[i].strip()

        if RE_PARTIE.match(line) or RE_PARTIE2.match(line):
            flush_article()
            current_article, current_paragraphs = None, []
            desc = get_desc(toutes_lignes, i)
            current_partie   = f"{line} – {desc}" if desc else line
            current_titre    = None
            current_chapitre = None
            i += 1
            continue

        if RE_TITRE.match(line):
            flush_article()
            current_article, current_paragraphs = None, []
            desc = get_desc(toutes_lignes, i)
            current_titre    = f"{line} – {desc}" if desc else line
            current_chapitre = None
            i += 1
            continue

        if RE_CHAPITRE.match(line):
            flush_article()
            current_article, current_paragraphs = None, []
            desc = get_desc(toutes_lignes, i)
            current_chapitre = f"{line} – {desc}" if desc else line
            i += 1
            continue

        m = RE_ARTICLE.match(line)
        if m:
            flush_article()
            art_num   = m.group(1).strip().rstrip(".")
            art_title = m.group(2).strip()
            current_article    = f"Article {art_num}" + (f" – {art_title}" if art_title else "")
            current_paragraphs = [art_title] if art_title else []
            i += 1
            continue

        if current_article is not None and line:
            current_paragraphs.append(line)

        i += 1

    flush_article()
    return records

# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def main():
    for config in FICHIERS:
        entree = config["entree"]
        label  = config["label"]
        annee  = config["annee"]

        print(f"\n{'='*60}")
        print(f"  Traitement : {label}")
        print(f"{'='*60}")

        if not entree.exists():
            print(f"  [ERREUR] Fichier introuvable : {entree}")
            continue

        with open(entree, encoding="utf-8") as f:
            document = json.load(f)

        pages  = document["pages"]
        source = document["source_file"]

        # ── 1. Articles ──
        print("  [1/2] Parsing hiérarchique des articles...")
        articles = parser_articles(pages, source, annee)

        sortie_articles = PROCESSED_DIR / f"loi_{annee.replace('-', '_')}_articles.json"
        with open(sortie_articles, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        avec_partie   = sum(1 for a in articles if a["partie"])
        avec_titre    = sum(1 for a in articles if a["titre"])
        avec_chapitre = sum(1 for a in articles if a["chapitre"])
        print(f"        {len(articles)} segments → {sortie_articles.name}")
        print(f"        Avec partie: {avec_partie} | titre: {avec_titre} | chapitre: {avec_chapitre}")

        # ── 2. Lignes budgétaires ──
        print("  [2/2] Extraction des lignes budgétaires (stratégie positionnelle)...")
        nb_tableaux = sum(len(p["tables"]) for p in pages)

        if nb_tableaux == 0:
            print(f"        Aucun tableau (PDF scanné) → fichier vide créé")
            lignes = []
        else:
            lignes = extraire_lignes_budgetaires(pages, source, annee)

        sortie_budget = PROCESSED_DIR / f"loi_{annee.replace('-', '_')}_lignes_budgetaires.json"
        with open(sortie_budget, "w", encoding="utf-8") as f:
            json.dump(lignes, f, ensure_ascii=False, indent=2)

        print(f"        {len(lignes)} lignes budgétaires → {sortie_budget.name}")

        if lignes:
            avec_cp  = sum(1 for l in lignes if l["montant_cp"] is not None)
            avec_ae  = sum(1 for l in lignes if l["montant_ae"] is not None)
            avec_txt = sum(1 for l in lignes if l["texte"])
            print(f"        Avec CP: {avec_cp} | Avec AE: {avec_ae} | Avec texte: {avec_txt}")

            # Aperçu des 3 premières lignes
            print("\n        Aperçu (3 premières lignes) :")
            for l in lignes[:3]:
                print(f"          [{l['page']}] {l['texte'][:60]}...")
                print(f"                CP={l['montant_cp']} | AE={l['montant_ae']}")

    print(f"\n{'='*60}")
    print("  Segmentation terminée. Fichiers dans data/processed/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()