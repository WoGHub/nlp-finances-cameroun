"""
extract_budget_camelot.py
=========================
Extraction des lignes budgétaires (pages 82-108) de la Loi de Finances
2024-2025 avec camelot (gestion correcte des tableaux complexes).

Produit :
  data/processed/loi_2024_2025_lignes_budgetaires.json
"""

import re
import json
import camelot
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PDF_PATH      = BASE_DIR / "data" / "raw" / "LOI DES FINANCES 2024-2025.pdf"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PAGES_BUDGET = "82-108"

# ─────────────────────────────────────────────────────────────────
# NETTOYAGE
# ─────────────────────────────────────────────────────────────────

def nettoyer_texte(texte: str) -> str:
    """Nettoie une cellule texte : sauts de ligne, espaces multiples, artefacts OCR."""
    if not texte:
        return ""
    t = texte.replace("\n", " ")
    # Supprimer caractères OCR parasites isolés (m, c, r, p, w, ·)
    t = re.sub(r"\b[mcwrp·]\b", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clean_montant(valeur: str) -> float | None:
    """
    Convertit une chaîne monétaire en float.
    Gère les espaces dans les nombres : '23 690 61 0' → 23690610
    """
    if not valeur:
        return None
    v = str(valeur).replace("\n", " ")
    # Supprimer lettres parasites OCR en début/fin
    v = re.sub(r"^[a-zA-Z·\s]+", "", v)
    v = re.sub(r"[a-zA-Z·\s]+$", "", v)
    # Supprimer tous les espaces (y compris dans les nombres)
    v = re.sub(r"\s+", "", v)
    v = v.replace(",", ".").replace("'", "")
    v = re.sub(r"[^\d.]", "", v)
    try:
        val = float(v) if v else None
        if val and 0 < val < 1e13:
            return val
        return None
    except ValueError:
        return None


def est_ligne_entete(row: list) -> bool:
    """Détecte les lignes d'en-tête."""
    keywords = {"no", "n°", "code", "libelle", "libellé", "objectif",
                "indicateur", "ae", "cp", "programme"}
    texte_row = " ".join(str(c).lower() for c in row)
    return sum(1 for kw in keywords if kw in texte_row) >= 2


def est_ligne_chapitre(row: list) -> bool:
    """Détecte les lignes de titre de chapitre."""
    texte = " ".join(str(c) for c in row)
    return bool(re.search(r"CHAPITRE\s+\d+\s*[-–]", texte, re.IGNORECASE))


def extraire_chapitre(row: list) -> str:
    """Extrait le nom du chapitre depuis une ligne."""
    texte = nettoyer_texte(" ".join(str(c) for c in row))
    m = re.search(r"(CHAPITRE\s+\d+\s*[-–].+)", texte, re.IGNORECASE)
    return m.group(1).strip() if m else texte


# ─────────────────────────────────────────────────────────────────
# DÉTECTION DES COLONNES AE / CP
# ─────────────────────────────────────────────────────────────────

def detecter_colonnes_ae_cp(df) -> tuple[int | None, int | None]:
    """
    Dans un DataFrame camelot, identifie les indices des colonnes AE et CP
    en cherchant ces mots dans les premières lignes.
    Retourne (idx_ae, idx_cp).
    """
    idx_ae, idx_cp = None, None

    for row_idx in range(min(3, len(df))):
        for col_idx in range(len(df.columns)):
            cell = str(df.iloc[row_idx, col_idx]).strip().upper()
            cell_clean = re.sub(r"\s+", "", cell)
            if "AE" == cell_clean and idx_ae is None:
                idx_ae = col_idx
            if "CP" == cell_clean and idx_cp is None:
                idx_cp = col_idx

    # Fallback : si pas trouvé par nom, prendre les 2 dernières colonnes
    n_cols = len(df.columns)
    if idx_cp is None:
        idx_cp = n_cols - 1
    if idx_ae is None:
        idx_ae = n_cols - 2

    return idx_ae, idx_cp


def detecter_colonne_code(df) -> int | None:
    """Identifie la colonne CODE/No."""
    for row_idx in range(min(3, len(df))):
        for col_idx in range(len(df.columns)):
            cell = str(df.iloc[row_idx, col_idx]).strip().upper()
            if re.match(r"^(NO|N°|CODE|COD)$", cell):
                return col_idx
    return 0  # fallback première colonne


# ─────────────────────────────────────────────────────────────────
# EXTRACTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────

def extraire_lignes_budgetaires() -> list:
    print(f"  Lecture des tableaux (pages {PAGES_BUDGET}) avec camelot...")
    print(f"  (Cela peut prendre 1-2 minutes...)")

    tables = camelot.read_pdf(
        str(PDF_PATH),
        pages=PAGES_BUDGET,
        flavor="lattice",
        line_scale=40,
    )

    print(f"  {len(tables)} tableaux détectés")

    lignes = []
    compteur = 0
    chapitre_courant = ""

    for t_idx, table in enumerate(tables):
        df = table.df
        page_num = table.parsing_report.get("page", "?")
        precision = table.parsing_report.get("accuracy", 0)

        if df.shape[0] < 2 or df.shape[1] < 3:
            continue

        # Détecter les colonnes AE et CP
        idx_ae, idx_cp = detecter_colonnes_ae_cp(df)
        idx_code = detecter_colonne_code(df)

        # Colonnes textuelles = tout ce qui n'est pas AE/CP/code numéro
        cols_texte = [i for i in range(len(df.columns))
                      if i not in {idx_ae, idx_cp, idx_code}
                      and i != idx_code]

        # Parcourir les lignes (skip en-têtes)
        debut_donnees = 0
        for row_idx in range(min(4, len(df))):
            row = df.iloc[row_idx].tolist()
            if est_ligne_entete(row):
                debut_donnees = row_idx + 1

        for row_idx in range(debut_donnees, len(df)):
            row = df.iloc[row_idx].tolist()

            # Ligne vide
            if all(str(c).strip() == "" for c in row):
                continue

            # Ligne de chapitre
            if est_ligne_chapitre(row):
                chapitre_courant = extraire_chapitre(row)
                continue

            # Extraire AE et CP
            cp_raw = str(row[idx_cp]).strip() if idx_cp < len(row) else ""
            ae_raw = str(row[idx_ae]).strip() if idx_ae < len(row) else ""

            montant_cp = clean_montant(cp_raw)
            montant_ae = clean_montant(ae_raw)

            # Ignorer lignes sans aucun montant
            if montant_cp is None and montant_ae is None:
                continue

            # Extraire le code
            code_raw = str(row[idx_code]).strip() if idx_code < len(row) else ""
            # Prendre le dernier nombre dans la cellule code (souvent "1\n168")
            codes = re.findall(r"\d+", code_raw)
            code = codes[-1] if codes else ""

            # Extraire et enrichir le texte (toutes colonnes textuelles)
            parties_texte = []
            for col_idx in range(len(df.columns)):
                if col_idx in {idx_ae, idx_cp}:
                    continue
                cell = nettoyer_texte(str(row[col_idx]))
                # Ignorer les cellules qui ne sont que des numéros courts
                if cell and not re.match(r"^\d{1,3}$", cell):
                    parties_texte.append(cell)

            texte_enrichi = " | ".join(p for p in parties_texte if p)

            if not texte_enrichi:
                continue

            compteur += 1
            lignes.append({
                "type":        "ligne_budgetaire",
                "id":          f"2024-2025_{compteur:04d}",
                "annee":       "2024-2025",
                "chapitre":    chapitre_courant,
                "code":        code,
                "texte":       texte_enrichi,
                "montant_cp":  montant_cp,
                "montant_ae":  montant_ae,
                "page":        page_num,
                "precision":   round(precision, 1),
                "source":      PDF_PATH.name,
            })

    return lignes


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  EXTRACTION BUDGÉTAIRE — camelot")
    print("="*60)

    if not PDF_PATH.exists():
        print(f"  [ERREUR] PDF introuvable : {PDF_PATH}")
        return

    lignes = extraire_lignes_budgetaires()

    # Sauvegarde
    sortie = PROCESSED_DIR / "loi_2024_2025_lignes_budgetaires.json"
    with open(sortie, "w", encoding="utf-8") as f:
        json.dump(lignes, f, ensure_ascii=False, indent=2)

    # Statistiques
    avec_cp  = sum(1 for l in lignes if l["montant_cp"] is not None)
    avec_ae  = sum(1 for l in lignes if l["montant_ae"] is not None)
    total_cp = sum(l["montant_cp"] for l in lignes if l["montant_cp"])

    print(f"\n   {len(lignes)} lignes extraites")
    print(f"     Avec CP  : {avec_cp}")
    print(f"     Avec AE  : {avec_ae}")
    print(f"     Total CP : {total_cp:,.0f} milliers FCFA")
    print(f"     Sauvegardé : {sortie}")

    # Aperçu
    print("\n  Aperçu (5 premières lignes) :")
    for l in lignes[:5]:
        print(f"\n    [{l['page']}] {l['texte'][:70]}...")
        print(f"          CP={l['montant_cp']} | AE={l['montant_ae']}")

    print(f"\n{'='*60}")
    print("  Prochaine étape : python src/classification/zero_shot_snd30.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()