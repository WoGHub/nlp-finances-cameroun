"""
compute_embeddings.py
=====================
Calcule les embeddings des articles des deux lois de finances.

Modèles utilisés :
  1. camembert-base           → français, précis sur textes juridiques
  2. paraphrase-multilingual-MiniLM-L12-v2 → multilingue, rapide

Produit :
  data/processed/embeddings_camembert.npz
  data/processed/embeddings_minilm.npz

Chaque fichier .npz contient :
  - embeddings_2023  : matrice (N1, dim)
  - embeddings_2024  : matrice (N2, dim)
  - ids_2023         : liste des références d'articles loi 2023-2024
  - ids_2024         : liste des références d'articles loi 2024-2025
"""

import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODELES = {
    "camembert": "dangvantuan/sentence-camembert-base",
    "minilm":    "paraphrase-multilingual-MiniLM-L12-v2",
}

# Taille des batchs (réduit pour CPU)
BATCH_SIZE = 16

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DES ARTICLES
# ─────────────────────────────────────────────────────────────────

def charger_articles(annee: str) -> tuple[list[str], list[str]]:
    """
    Charge les articles d'une loi.
    Retourne (textes, references).
    Le texte enrichi = titre hiérarchique + texte de l'article.
    """
    chemin = PROCESSED_DIR / f"loi_{annee.replace('-', '_')}_articles.json"
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}\n→ Lancez d'abord clean_and_segment_v3.py")

    with open(chemin, encoding="utf-8") as f:
        articles = json.load(f)

    textes = []
    references = []

    for art in articles:
        # Enrichissement : ajouter le contexte hiérarchique au texte
        contexte_parts = []
        if art.get("titre"):
            contexte_parts.append(art["titre"])
        if art.get("chapitre"):
            contexte_parts.append(art["chapitre"])

        texte_final = art["texte"]
        if contexte_parts:
            texte_final = " | ".join(contexte_parts) + " | " + texte_final

        # Tronquer à 512 mots pour éviter les dépassements de contexte
        mots = texte_final.split()
        if len(mots) > 512:
            texte_final = " ".join(mots[:512])

        textes.append(texte_final)
        references.append(art.get("reference", f"art_{len(references)}"))

    print(f"  Loi {annee} : {len(textes)} articles chargés")
    return textes, references


# ─────────────────────────────────────────────────────────────────
# CALCUL DES EMBEDDINGS
# ─────────────────────────────────────────────────────────────────

def calculer_embeddings(
    textes: list[str],
    model: SentenceTransformer,
    nom_modele: str,
    annee: str,
) -> np.ndarray:
    """Calcule les embeddings par batch avec affichage de progression."""
    print(f"    Encodage {annee} avec {nom_modele}...")
    t0 = time.time()

    embeddings = model.encode(
        textes,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # normalisation L2 pour cosine similarity
        convert_to_numpy=True,
    )

    duree = time.time() - t0
    print(f"    ✅ {len(textes)} embeddings calculés en {duree:.1f}s | dim={embeddings.shape[1]}")
    return embeddings


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  CALCUL DES EMBEDDINGS")
    print("="*60)

    # ── Chargement des articles ──
    print("\n[1/3] Chargement des articles...")
    textes_2023, refs_2023 = charger_articles("2023-2024")
    textes_2024, refs_2024 = charger_articles("2024-2025")

    resultats = {}

    # ── Embeddings par modèle ──
    for nom, model_id in MODELES.items():
        print(f"\n[{'2' if nom == 'camembert' else '3'}/3] Modèle : {model_id}")
        print(f"  Chargement du modèle (peut prendre 1-2 min au premier lancement)...")

        t0 = time.time()
        model = SentenceTransformer(model_id)
        print(f"  Modèle chargé en {time.time()-t0:.1f}s")

        emb_2023 = calculer_embeddings(textes_2023, model, nom, "2023-2024")
        emb_2024 = calculer_embeddings(textes_2024, model, nom, "2024-2025")

        # Sauvegarde
        sortie = PROCESSED_DIR / f"embeddings_{nom}.npz"
        np.savez(
            sortie,
            embeddings_2023=emb_2023,
            embeddings_2024=emb_2024,
            ids_2023=np.array(refs_2023, dtype=object),
            ids_2024=np.array(refs_2024, dtype=object),
            textes_2023=np.array(textes_2023, dtype=object),
            textes_2024=np.array(textes_2024, dtype=object),
        )
        print(f"  Sauvegardé : {sortie.name}")

        resultats[nom] = {
            "emb_2023": emb_2023,
            "emb_2024": emb_2024,
            "refs_2023": refs_2023,
            "refs_2024": refs_2024,
        }

        # Libérer la mémoire
        del model

    # ── Aperçu : top similarités inter-lois ──
    print("\n" + "="*60)
    print("  APERÇU : Articles les plus similaires entre les deux lois")
    print("="*60)

    for nom, data in resultats.items():
        print(f"\n  Modèle : {nom.upper()}")

        # Matrice de similarité cosinus (N1 x N2)
        # Les embeddings sont déjà normalisés → produit scalaire = cosine sim
        sim_matrix = data["emb_2023"] @ data["emb_2024"].T

        # Pour chaque article 2023, trouver son meilleur correspondant 2024
        meilleurs_scores = sim_matrix.max(axis=1)
        meilleurs_idx    = sim_matrix.argmax(axis=1)

        # Trier par score décroissant
        top_indices = np.argsort(meilleurs_scores)[::-1][:5]

        print(f"  {'Score':>6}  {'Article 2023-2024':<35}  {'→ Article 2024-2025'}")
        print(f"  {'-'*80}")
        for idx in top_indices:
            score   = meilleurs_scores[idx]
            ref_old = data["refs_2023"][idx][:33]
            ref_new = data["refs_2024"][meilleurs_idx[idx]][:33]
            print(f"  {score:.4f}  {ref_old:<35}  → {ref_new}")

        # Distribution globale des similarités
        scores_flat = meilleurs_scores
        print(f"\n  Similarité moyenne  : {scores_flat.mean():.4f}")
        print(f"  Similarité médiane  : {np.median(scores_flat):.4f}")
        print(f"  Articles > 0.90 sim : {(scores_flat > 0.90).sum()} / {len(scores_flat)}")
        print(f"  Articles < 0.60 sim : {(scores_flat < 0.60).sum()} / {len(scores_flat)}")

    print(f"\n{'='*60}")
    print("  Embeddings calculés et sauvegardés dans data/processed/")
    print("  Prochaine étape : python src/embeddings/semantic_audit.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()