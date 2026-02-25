import camelot

PDF_PATH = r"data/raw/LOI DES FINANCES 2024-2025.pdf"

print("Test camelot sur pages 82-84...")
tables = camelot.read_pdf(
    PDF_PATH,
    pages="82,83,84",
    flavor="lattice",  # pour tableaux avec bordures
)

print(f"{len(tables)} tableaux détectés")
for i, table in enumerate(tables):
    print(f"\n--- Tableau {i+1} (page {table.parsing_report['page']}) ---")
    print(f"  Précision : {table.parsing_report['accuracy']:.1f}%")
    print(f"  Shape : {table.df.shape}")
    print(table.df.head(4).to_string())