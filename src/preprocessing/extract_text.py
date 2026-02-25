"""
extract_text_v2.py
==================
Extraction du texte des Lois de Finances du Cameroun.

- LOI 2023-2024 : PDF scanné → OCR via Tesseract
- LOI 2024-2025 : PDF numérique
    • Texte général : toutes les pages (pdfplumber)
    • Tableaux budgétaires : pages 82-108 uniquement (pdfplumber)

Sortie : data/extracted/loi_finances_XXXX_XXXX.json
Format identique à extract_text_v1 pour compatibilité avec clean_and_segment.py
"""

import json
import re
import sys
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

# Chemins des outils OCR (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\Library\bin"

# Chemins des fichiers
BASE_DIR      = Path(__file__).resolve().parent.parent.parent  # racine du projet
RAW_DIR       = BASE_DIR / "data" / "raw"
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

FICHIERS = {
    "loi_2023_2024": {
        "pdf":    RAW_DIR / "LOI DES FINANCES 2023-2024.pdf",
        "sortie": EXTRACTED_DIR / "loi_finances_2023_2024.json",
        "mode":   "ocr",        # PDF scanné
        "label":  "LOI_2023_2024",
    },
    "loi_2024_2025": {
        "pdf":    RAW_DIR / "LOI DES FINANCES 2024-2025.pdf",
        "sortie": EXTRACTED_DIR / "loi_finances_2024_2025.json",
        "mode":   "numerique",  # PDF natif
        "label":  "LOI_2024_2025",
        "pages_budget": (82, 108),  # pages Moyens des Politiques Publiques (1-indexé)
    },
}

# OCR : langue française + configuration pour texte juridique dense
OCR_CONFIG = "--oem 3 --psm 6 -l fra"

# ─────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────

def is_header_or_footer(line: str) -> bool:
    """Détecte les en-têtes, pieds de page et artefacts de scan."""
    stripped = line.strip()
    if len(stripped) <= 2:
        return True
    # Numéro de page seul
    if re.match(r"^\d{1,3}$", stripped):
        return True
    # Tampons présidentiels fréquents dans les lois camerounaises
    noise_patterns = [
        r"^PRESIDEN[A-Z\s]+$",
        r"^SECRETAR[A-Z\s]+$",
        r"^CERTIFIE[A-Z\s']+$",
        r"^[-·/\\,.'`\s]+$",
        r"^[A-Za-z\s'\-\.]{1,4}$",
        r"^REPUBLIQUE\s+DU\s+CAMEROUN",
        r"^REPUBLIC\s+OF\s+CAMEROON",
        r"^Paix\s*[-–]\s*Travail\s*[-–]\s*Patrie",
        r"^Peace\s*[-–]\s*Work\s*[-–]\s*Fatherland",
    ]
    for pattern in noise_patterns:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True
    return False


def nettoyer_lignes(lignes: list[str]) -> list[str]:
    """Supprime les lignes parasites d'une liste."""
    return [l for l in lignes if not is_header_or_footer(l)]


def construire_page_json(num_page: int, text_blocks: list[str], tables: list) -> dict:
    """Construit le dictionnaire d'une page au format standard."""
    return {
        "page_number": num_page,
        "text_blocks": text_blocks,
        "tables": tables,
    }

# ─────────────────────────────────────────────────────────────────
# EXTRACTION OCR  (loi 2023-2024)
# ─────────────────────────────────────────────────────────────────

def extraire_ocr(pdf_path: Path, label: str) -> dict:
    """
    Convertit chaque page en image puis applique Tesseract (français).
    Retourne un document au format standard.
    """
    print(f"  [OCR] Conversion du PDF en images...")
    images = convert_from_path(
        str(pdf_path),
        dpi=300,
        poppler_path=POPPLER_PATH,
    )
    print(f"  [OCR] {len(images)} pages détectées. Lancement de l'OCR...")

    pages = []
    for i, image in enumerate(images, start=1):
        if i % 10 == 0:
            print(f"    → Page {i}/{len(images)}")

        texte_brut = pytesseract.image_to_string(image, config=OCR_CONFIG)
        lignes_brutes = texte_brut.split("\n")
        lignes_propres = nettoyer_lignes(lignes_brutes)

        pages.append(construire_page_json(
            num_page=i,
            text_blocks=lignes_propres,
            tables=[],  # pas d'extraction de tableaux sur PDF scanné
        ))

    return {
        "document":    label,
        "source_file": pdf_path.name,
        "total_pages": len(images),
        "mode":        "ocr",
        "pages":       pages,
    }

# ─────────────────────────────────────────────────────────────────
# EXTRACTION NUMÉRIQUE  (loi 2024-2025)
# ─────────────────────────────────────────────────────────────────

def extraire_numerique(pdf_path: Path, label: str, pages_budget: tuple[int, int] = None) -> dict:
    """
    Extrait le texte et les tableaux d'un PDF natif via pdfplumber.
    Si pages_budget est fourni (start, end), les tableaux ne sont extraits
    que sur cette plage de pages (Moyens des Politiques Publiques).
    """
    budget_start, budget_end = pages_budget if pages_budget else (1, 9999)

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        print(f"  [PDF] {total} pages détectées.")

        for i, page in enumerate(pdf.pages, start=1):
            if i % 20 == 0:
                print(f"    → Page {i}/{total}")

            # ── Texte ──
            texte = page.extract_text() or ""
            lignes = nettoyer_lignes(texte.split("\n"))

            # ── Tableaux (uniquement sur la plage budgétaire) ──
            tableaux = []
            if budget_start <= i <= budget_end:
                try:
                    raw_tables = page.extract_tables()
                    for table in (raw_tables or []):
                        # Nettoyer les cellules None
                        table_propre = [
                            [str(cell).strip() if cell is not None else "" for cell in row]
                            for row in table
                        ]
                        if table_propre:
                            tableaux.append(table_propre)
                except Exception as e:
                    print(f"    [WARN] Erreur tableau page {i} : {e}")

            pages.append(construire_page_json(
                num_page=i,
                text_blocks=lignes,
                tables=tableaux,
            ))

    return {
        "document":      label,
        "source_file":   pdf_path.name,
        "total_pages":   total,
        "mode":          "numerique",
        "pages_budget":  list(pages_budget) if pages_budget else None,
        "pages":         pages,
    }

# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def main():
    for nom, config in FICHIERS.items():
        pdf_path = config["pdf"]
        sortie   = config["sortie"]
        label    = config["label"]
        mode     = config["mode"]

        print(f"\n{'='*60}")
        print(f"  Traitement : {label}")
        print(f"  Fichier    : {pdf_path.name}")
        print(f"  Mode       : {mode.upper()}")
        print(f"{'='*60}")

        if not pdf_path.exists():
            print(f"  [ERREUR] Fichier introuvable : {pdf_path}")
            print(f"  → Vérifiez que le PDF est bien dans data/raw/")
            continue

        try:
            if mode == "ocr":
                document = extraire_ocr(pdf_path, label)
            else:
                pages_budget = config.get("pages_budget")
                document = extraire_numerique(pdf_path, label, pages_budget)

            # Sauvegarde JSON
            with open(sortie, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)

            # Statistiques
            total_blocs    = sum(len(p["text_blocks"]) for p in document["pages"])
            total_tableaux = sum(len(p["tables"]) for p in document["pages"])
            print(f"\n  ✅ Extraction terminée")
            print(f"     Pages         : {document['total_pages']}")
            print(f"     Blocs texte   : {total_blocs}")
            print(f"     Tableaux      : {total_tableaux}")
            print(f"     Sauvegardé    : {sortie}")

        except Exception as e:
            print(f"\n  [ERREUR] {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  Toutes les extractions sont terminées.")
    print("  Prochaine étape : python src/preprocessing/clean_and_segment.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()