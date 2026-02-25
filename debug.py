import json

with open(r"C:\ISE3\NLP\nlp-finances-cameroun\data\extracted\loi_finances_2024_2025.json", encoding="utf-8") as f:
    doc = json.load(f)

for page in doc["pages"]:
    if 82 <= page["page_number"] <= 90 and page["tables"]:
        print(f"\n=== PAGE {page['page_number']} ===")
        for table in page["tables"]:
            print(f"  Nb colonnes : {len(table[0]) if table else 0}")
            for i, row in enumerate(table[:5]):  # 5 premières lignes
                print(f"  Ligne {i} : {row}")
            print()