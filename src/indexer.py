import pandas as pd
import numpy as np

# Charger le dataset échantillon
df = pd.read_csv('../data/processed/healthcare_jobs_sample_2000.csv')

print(f"\n✅ Dataset chargé : {len(df):,} offres")

# ============================================================================
# STRATÉGIE : Combiner titre + description
# ============================================================================

print(f"\n📊 CRÉATION DU TEXTE POUR SBERT :")
print("-" * 80)

# Option 1 : Simple concatenation
# text = title + " | " + description

# Option 2 : Ajouter le niveau d'expérience (si disponible)
# text = title + " [Experience: " + experience + "] | " + description

# On va avec Option 1 pour commencer (simple et efficace)

df['text_for_sbert'] = df['title'] + " | " + df['description'].astype(str)

# Statistiques sur les textes créés
text_lengths = df['text_for_sbert'].str.split().str.len()

print(f"✅ Textes créés pour SBERT")
print(f"\n📊 STATISTIQUES :")
print(f"   Longueur moyenne  : {text_lengths.mean():.0f} mots")
print(f"   Longueur médiane  : {text_lengths.median():.0f} mots")
print(f"   Min               : {text_lengths.min():.0f} mots")
print(f"   Max               : {text_lengths.max():.0f} mots")

print(f"\n⚠️ SBERT TRUNCATION (limite ~400 mots) :")
over_limit = (text_lengths > 400).sum()
print(f"   Textes > 400 mots : {over_limit:,} ({over_limit/len(df)*100:.1f}%)")
print(f"   → Ces textes seront automatiquement tronqués par SBERT")

# Exemples
print(f"\n📋 EXEMPLES DE TEXTES (3 premiers) :")
print("=" * 80)
for i in range(3):
    text = df.iloc[i]['text_for_sbert']
    print(f"\nEXEMPLE {i+1}:")
    print(f"Length: {len(text.split())} mots")
    print(f"Preview: {text[:200]}...")
    print("-" * 80)

# Sauvegarder le dataset avec la nouvelle colonne
output_path = '../data/processed/healthcare_jobs_for_sbert.csv'
df.to_csv(output_path, index=False)

print(f"\n{'='*80}")
print(f"✅ DATASET PRÉPARÉ SAUVEGARDÉ : {output_path}")
print(f"{'='*80}")
print(f"\n🎯 Prêt pour l'encodage SBERT !")