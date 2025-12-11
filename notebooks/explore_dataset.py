
"""
Script d'exploration approfondie du dataset Healthcare
Pour créer les requêtes d'évaluation basées sur les données réelles

Usage:
    python explore_dataset.py

Outputs:
    - Statistiques complètes dans la console
    - 6 graphiques PNG
    - Recommandations pour les requêtes
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)


def print_section(title):
    """Affiche un titre de section formaté"""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)


def load_data(filepath):
    """Charge le dataset"""
    print_section("CHARGEMENT DES DONNÉES")
    df = pd.read_csv(filepath)
    print(f"\n✅ Dataset chargé : {len(df):,} offres")
    print(f"\nColonnes disponibles :")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        print(f"  • {col:35s} : {non_null:5d} / {len(df)} ({pct:5.1f}% remplis)")
    return df


def analyze_titles(df):
    """Analyse des titres"""
    print_section("1. ANALYSE DES TITRES")
    
    # Stats de base
    title_lengths = df['title'].str.len()
    title_word_counts = df['title'].str.split().str.len()
    
    print("\n📏 Longueur des titres :")
    print(f"  • Min     : {title_lengths.min()} caractères")
    print(f"  • Max     : {title_lengths.max()} caractères")
    print(f"  • Moyenne : {title_lengths.mean():.1f} caractères")
    print(f"  • Médiane : {title_lengths.median():.0f} caractères")
    
    print("\n📝 Nombre de mots par titre :")
    print(f"  • Min     : {title_word_counts.min()} mots")
    print(f"  • Max     : {title_word_counts.max()} mots")
    print(f"  • Moyenne : {title_word_counts.mean():.1f} mots")
    print(f"  • Médiane : {title_word_counts.median():.0f} mots")
    
    # Graphique
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(title_word_counts, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Nombre de mots')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution : Nombre de mots par titre')
    axes[0].axvline(title_word_counts.mean(), color='r', linestyle='--', 
                    label=f'Moyenne: {title_word_counts.mean():.1f}')
    axes[0].legend()
    
    axes[1].hist(title_lengths, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Longueur (caractères)')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution : Longueur des titres')
    axes[1].axvline(title_lengths.mean(), color='r', linestyle='--', 
                    label=f'Moyenne: {title_lengths.mean():.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('titles_distribution.png', dpi=150, bbox_inches='tight')
    print("\n💾 Graphique sauvegardé : titles_distribution.png")
    plt.close()
    
    # Mots les plus fréquents
    print("\n🔤 MOTS LES PLUS FRÉQUENTS DANS LES TITRES")
    print("-"*80)
    
    all_title_words = []
    for title in df['title'].dropna():
        words = str(title).lower().split()
        words = [re.sub(r'[^a-z0-9]', '', w) for w in words]
        words = [w for w in words if len(w) > 2]
        all_title_words.extend(words)
    
    word_freq = Counter(all_title_words)
    
    print(f"\nTotal de mots uniques : {len(word_freq):,}")
    print(f"Total d'occurrences : {sum(word_freq.values()):,}")
    
    print("\n📌 Top 50 mots :")
    print(f"{'Rang':>4s} {'Mot':20s} {'Count':>8s} {'% des offres':>15s} {'Barre':s}")
    print("-"*80)
    
    for rank, (word, count) in enumerate(word_freq.most_common(50), 1):
        pct = (count / len(df)) * 100
        bar = '█' * int(pct / 2)
        print(f"{rank:4d} {word:20s} {count:8d} {pct:14.1f}% {bar}")
    
    # Graphique top 30
    top_30_words = word_freq.most_common(30)
    words, counts = zip(*top_30_words)
    
    plt.figure(figsize=(14, 8))
    plt.barh(range(len(words)), counts, color='steelblue', edgecolor='black')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Fréquence')
    plt.title('Top 30 Mots les Plus Fréquents dans les Titres', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('top30_title_words.png', dpi=150, bbox_inches='tight')
    print("\n💾 Graphique sauvegardé : top30_title_words.png")
    plt.close()
    
    return word_freq


def analyze_categories(df):
    """Analyse des catégories de jobs"""
    print_section("2. CATÉGORIES DE JOBS")
    
    categories_keywords = {
        'Registered Nurses (RN)': ['registered nurse', 'rn ', ' rn', 'staff nurse'],
        'Licensed Practical Nurses (LPN)': ['lpn', 'licensed practical nurse', 'lvn'],
        'Certified Nursing Assistants (CNA)': ['cna', 'nursing assistant', 'nurse aide'],
        'Nurse Practitioners': ['nurse practitioner', 'np ', ' np'],
        'Physical Therapists': ['physical therapist', 'pt ', ' pt', 'physiotherapist'],
        'Occupational Therapists': ['occupational therapist', 'ot ', ' ot'],
        'Speech Therapists': ['speech therapist', 'slp', 'speech pathologist'],
        'Respiratory Therapists': ['respiratory therapist', 'respiratory care'],
        'Medical Assistants': ['medical assistant', ' ma ', 'clinical assistant'],
        'Physicians/Doctors': ['physician', 'doctor', 'md ', ' md', 'hospitalist'],
        'Technicians': ['technician', ' tech ', 'technologist', 'lab tech', 'surgical tech'],
        'Dental': ['dental', 'dentist', 'hygienist', 'orthodont'],
        'Pharmacy': ['pharmacy', 'pharmacist', 'pharm'],
        'Mental Health': ['mental', 'psychologist', 'counselor', 'behavioral health'],
        'Social Workers': ['social worker', 'case manager', 'caseworker'],
        'Emergency/Trauma': ['emergency', 'trauma', ' er ', ' icu', 'critical care'],
        'Surgical': ['surgical', 'surgery', 'operating room', ' or ', 'perioperative'],
        'Pediatric': ['pediatric', 'peds', 'children', 'neonatal', 'nicu'],
        'Home Health': ['home health', 'home care', 'visiting nurse'],
        'Administrative': ['manager', 'director', 'coordinator', 'administrator', 'supervisor'],
        'Medical Records': ['medical records', 'health information', 'coder', 'coding'],
        'Ultrasound': ['ultrasound', 'sonographer', 'sonography', 'echo tech'],
        'Anesthesia': ['anesthesia', 'anesthetist', 'crna'],
        'Radiology': ['radiology', 'radiologic', 'imaging', 'mri', 'ct tech', 'xray'],
    }
    
    category_counts = {}
    category_examples = {}
    
    for category, keywords in categories_keywords.items():
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        mask = df['title'].str.lower().str.contains(pattern, regex=True, na=False)
        count = mask.sum()
        category_counts[category] = count
        
        if count > 0:
            examples = df[mask]['title'].sample(min(3, count)).tolist()
            category_examples[category] = examples
    
    print(f"\n{'Catégorie':45s} {'Count':>8s} {'% du total':>12s} {'Barre':s}")
    print("-"*100)
    
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(df)) * 100
        bar = '█' * int(pct / 2)
        print(f"{cat:45s} {count:8d} {pct:11.1f}% {bar}")
    
    # Offres non catégorisées
    all_masks = pd.Series([False] * len(df))
    for keywords in categories_keywords.values():
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        all_masks |= df['title'].str.lower().str.contains(pattern, regex=True, na=False)
    
    uncategorized = (~all_masks).sum()
    print(f"\n⚠️  Offres NON catégorisées : {uncategorized} ({(uncategorized/len(df))*100:.1f}%)")
    
    # Graphique
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    cats, cnts = zip(*top_categories)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(cats)), cnts, color='coral', edgecolor='black')
    plt.yticks(range(len(cats)), cats, fontsize=10)
    plt.xlabel('Nombre d\'offres', fontsize=12)
    plt.title('Top 15 Catégories de Jobs Healthcare', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('categories_distribution.png', dpi=150, bbox_inches='tight')
    print("\n💾 Graphique sauvegardé : categories_distribution.png")
    plt.close()
    
    # Exemples
    print("\n📋 EXEMPLES DE TITRES PAR CATÉGORIE (Top 10)")
    print("-"*80)
    for category, examples in list(category_examples.items())[:10]:
        print(f"\n🏷️  {category} ({category_counts[category]} offres) :")
        for i, title in enumerate(examples, 1):
            print(f"   {i}. {title}")
    
    return category_counts, category_examples


def analyze_bigrams_trigrams(df):
    """Analyse des bigrams et trigrams"""
    print_section("3. BIGRAMS & TRIGRAMS")
    
    # BIGRAMS
    bigrams = []
    for title in df['title'].dropna():
        words = str(title).lower().split()
        words = [re.sub(r'[^a-z]', '', w) for w in words if len(re.sub(r'[^a-z]', '', w)) > 2]
        for i in range(len(words)-1):
            bigrams.append(f"{words[i]} {words[i+1]}")
    
    bigram_freq = Counter(bigrams)
    
    print("\n📌 Top 30 BIGRAMS (2 mots consécutifs) :")
    print(f"{'Rang':>4s} {'Bigram':35s} {'Count':>8s} {'%':>8s}")
    print("-"*60)
    
    for rank, (bigram, count) in enumerate(bigram_freq.most_common(30), 1):
        pct = (count / len(df)) * 100
        print(f"{rank:4d} {bigram:35s} {count:8d} {pct:7.1f}%")
    
    # TRIGRAMS
    trigrams = []
    for title in df['title'].dropna():
        words = str(title).lower().split()
        words = [re.sub(r'[^a-z]', '', w) for w in words if len(re.sub(r'[^a-z]', '', w)) > 2]
        for i in range(len(words)-2):
            trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    trigram_freq = Counter(trigrams)
    
    print("\n📌 Top 20 TRIGRAMS (3 mots consécutifs) :")
    print(f"{'Rang':>4s} {'Trigram':45s} {'Count':>8s} {'%':>8s}")
    print("-"*70)
    
    for rank, (trigram, count) in enumerate(trigram_freq.most_common(20), 1):
        pct = (count / len(df)) * 100
        print(f"{rank:4d} {trigram:45s} {count:8d} {pct:7.1f}%")
    
    return bigram_freq, trigram_freq


def analyze_experience(df):
    """Analyse de la distribution par expérience"""
    print_section("4. NIVEAU D'EXPÉRIENCE")
    
    exp_dist = df['formatted_experience_level'].value_counts()
    
    print(f"\n{'Niveau':35s} {'Count':>8s} {'%':>8s} {'Barre':s}")
    print("-"*80)
    
    for level, count in exp_dist.items():
        pct = (count / len(df)) * 100
        bar = '█' * int(pct / 2)
        print(f"{str(level):35s} {count:8d} {pct:7.1f}% {bar}")
    
    # Graphique
    plt.figure(figsize=(10, 6))
    exp_dist.plot(kind='bar', color='teal', edgecolor='black', alpha=0.8)
    plt.xlabel('Niveau d\'expérience', fontsize=12)
    plt.ylabel('Nombre d\'offres', fontsize=12)
    plt.title('Distribution par Niveau d\'Expérience', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('experience_distribution.png', dpi=150, bbox_inches='tight')
    print("\n💾 Graphique sauvegardé : experience_distribution.png")
    plt.close()
    
    return exp_dist


def analyze_descriptions(df):
    """Analyse des descriptions"""
    print_section("5. ANALYSE DES DESCRIPTIONS")
    
    desc_lengths = df['description'].str.len()
    desc_word_counts = df['description'].str.split().str.len()
    
    print("\n📏 Longueur des descriptions :")
    print(f"  • Min     : {desc_lengths.min():,} caractères")
    print(f"  • Max     : {desc_lengths.max():,} caractères")
    print(f"  • Moyenne : {desc_lengths.mean():,.1f} caractères")
    print(f"  • Médiane : {desc_lengths.median():,.0f} caractères")
    
    print("\n📝 Nombre de mots par description :")
    print(f"  • Min     : {desc_word_counts.min():,} mots")
    print(f"  • Max     : {desc_word_counts.max():,} mots")
    print(f"  • Moyenne : {desc_word_counts.mean():,.1f} mots")
    print(f"  • Médiane : {desc_word_counts.median():,.0f} mots")
    
    # Graphiques
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(desc_word_counts, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0].set_xlabel('Nombre de mots')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution : Nombre de mots par description')
    axes[0].axvline(desc_word_counts.mean(), color='r', linestyle='--', 
                    label=f'Moyenne: {desc_word_counts.mean():.1f}')
    axes[0].legend()
    
    axes[1].hist(desc_lengths, bins=40, edgecolor='black', alpha=0.7, color='purple')
    axes[1].set_xlabel('Longueur (caractères)')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution : Longueur des descriptions')
    axes[1].axvline(desc_lengths.mean(), color='r', linestyle='--', 
                    label=f'Moyenne: {desc_lengths.mean():.0f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('descriptions_distribution.png', dpi=150, bbox_inches='tight')
    print("\n💾 Graphique sauvegardé : descriptions_distribution.png")
    plt.close()
    
    # Mots fréquents dans descriptions
    print("\n🔤 MOTS FRÉQUENTS DANS LES DESCRIPTIONS (hors stopwords)")
    print("-"*80)
    
    stopwords = {
        'the', 'and', 'for', 'with', 'you', 'will', 'are', 'this', 'that', 'have',
        'from', 'they', 'been', 'has', 'your', 'our', 'can', 'all', 'were', 'not',
        'but', 'what', 'their', 'said', 'each', 'which', 'she', 'who', 'one', 'had',
        'her', 'him', 'more', 'when', 'there', 'them', 'these', 'than', 'into', 'only',
    }
    
    all_desc_words = []
    for desc in df['description'].dropna():
        words = str(desc).lower().split()
        words = [re.sub(r'[^a-z]', '', w) for w in words]
        words = [w for w in words if len(w) > 3 and w not in stopwords]
        all_desc_words.extend(words)
    
    desc_word_freq = Counter(all_desc_words)
    
    print(f"\nTotal de mots uniques : {len(desc_word_freq):,}")
    print("\n📌 Top 40 mots :")
    print(f"{'Rang':>4s} {'Mot':20s} {'Count':>10s} {'% des offres':>15s}")
    print("-"*55)
    
    for rank, (word, count) in enumerate(desc_word_freq.most_common(40), 1):
        appears_in = df['description'].str.lower().str.contains(word, na=False).sum()
        pct = (appears_in / len(df)) * 100
        print(f"{rank:4d} {word:20s} {count:10,d} {pct:14.1f}%")


def analyze_locations(df):
    """Analyse des locations"""
    print_section("6. ANALYSE DES LOCATIONS")
    
    location_counts = df['location'].value_counts()
    
    print(f"\nTotal de locations uniques : {len(location_counts):,}")
    print(f"\n📌 Top 20 locations :")
    print(f"{'Rang':>4s} {'Location':50s} {'Count':>8s} {'%':>8s}")
    print("-"*75)
    
    for rank, (location, count) in enumerate(location_counts.head(20).items(), 1):
        pct = (count / len(df)) * 100
        print(f"{rank:4d} {str(location)[:50]:50s} {count:8d} {pct:7.1f}%")
    
    # États
    states = []
    for loc in df['location'].dropna():
        match = re.search(r'\b([A-Z]{2})\b$', str(loc))
        if match:
            states.append(match.group(1))
    
    if states:
        state_freq = Counter(states)
        print(f"\n🗺️ Top 15 États :")
        print(f"{'Rang':>4s} {'État':10s} {'Count':>8s} {'%':>8s}")
        print("-"*35)
        
        for rank, (state, count) in enumerate(state_freq.most_common(15), 1):
            pct = (count / len(states)) * 100
            print(f"{rank:4d} {state:10s} {count:8d} {pct:7.1f}%")


def generate_recommendations(category_counts, bigram_freq, trigram_freq, df):
    """Génère les recommandations pour les requêtes"""
    print_section("💡 RECOMMANDATIONS POUR LES REQUÊTES D'ÉVALUATION")
    
    print("\n" + "="*80)
    print("✅ REQUÊTES SPÉCIFIQUES à inclure (>50 offres)")
    print("="*80)
    
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= 50:
            print(f"  • {cat:45s} : {count:4d} offres")
    
    print("\n" + "="*80)
    print("✅ BIGRAMS pertinents (>20 occurrences)")
    print("="*80)
    
    for bigram, count in bigram_freq.most_common(20):
        if count > 20:
            print(f"  • '{bigram}' ({count} occurrences)")
    
    print("\n" + "="*80)
    print("✅ TRIGRAMS pertinents (>10 occurrences)")
    print("="*80)
    
    for trigram, count in trigram_freq.most_common(15):
        if count > 10:
            print(f"  • '{trigram}' ({count} occurrences)")
    
    print("\n" + "="*80)
    print("⚠️ REQUÊTES À ÉVITER (peu de données)")
    print("="*80)
    
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1]):
        if count < 20:
            print(f"  • {cat} : seulement {count} offres")
    
    print("\n" + "="*80)
    print("📊 RÉPARTITION SUGGÉRÉE DES 55 REQUÊTES")
    print("="*80)
    print("""
1. SPECIFIC (20 requêtes) - Catégories >50 offres
   → Nurses RN/LPN/CNA variations
   → Therapists (PT, OT, Speech, Respiratory)
   → Medical/Dental/Pharmacy
   → Emergency/Surgical/Pediatric

2. MODERATE (15 requêtes) - Combinaisons larges
   → "nurse critical care"
   → "therapist rehabilitation"
   → "medical technician"

3. GENERIC (10 requêtes) - Très vagues
   → "healthcare job"
   → "medical position"
   → "health services"

4. OUT_OF_SCOPE (10 requêtes) - Hors Healthcare
   → "software engineer"
   → "data scientist"
   → "truck driver"
    """)


def print_summary(df, word_freq, category_counts, bigram_freq, exp_dist):
    """Affiche le résumé exécutif"""
    print_section("📊 RÉSUMÉ EXÉCUTIF")
    
    print(f"""
📁 DATASET : {len(df):,} offres Healthcare

📝 TITRES :
   • Moyenne : {df['title'].str.split().str.len().mean():.1f} mots
   • Top 3 mots : {', '.join([w for w, _ in word_freq.most_common(3)])}

📄 DESCRIPTIONS :
   • Moyenne : {df['description'].str.split().str.len().mean():.0f} mots
   • Médiane : {df['description'].str.split().str.len().median():.0f} mots

🏷️ CATÉGORIES PRINCIPALES :
""")
    
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = (count / len(df)) * 100
        print(f"   • {cat:40s} : {count:4d} ({pct:5.1f}%)")
    
    print(f"""
📊 DISTRIBUTION EXPÉRIENCE :
""")
    
    for level, count in exp_dist.head(3).items():
        pct = (count / len(df)) * 100
        print(f"   • {str(level):30s} : {count:4d} ({pct:5.1f}%)")
    
    print(f"""
🎯 STATISTIQUES CLÉS :
   • {sum(1 for c in category_counts.values() if c >= 50)} catégories avec >50 offres
   • {len(bigram_freq)} bigrams uniques
   • {len(word_freq)} mots uniques dans les titres

✅ FICHIERS GÉNÉRÉS :
   • titles_distribution.png
   • top30_title_words.png
   • categories_distribution.png
   • experience_distribution.png
   • descriptions_distribution.png
    """)


def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print("🔍 EXPLORATION APPROFONDIE DU DATASET HEALTHCARE")
    print("="*80)
    print("\nObjectif : Comprendre les données pour créer des requêtes d'évaluation pertinentes\n")
    
    # Chargement
    df = load_data('../data/processed/healthcare_jobs_sample_2000.csv')
    
    # Analyses
    word_freq = analyze_titles(df)
    category_counts, category_examples = analyze_categories(df)
    bigram_freq, trigram_freq = analyze_bigrams_trigrams(df)
    exp_dist = analyze_experience(df)
    analyze_descriptions(df)
    analyze_locations(df)
    
    # Recommandations
    generate_recommendations(category_counts, bigram_freq, trigram_freq, df)
    
    # Résumé
    print_summary(df, word_freq, category_counts, bigram_freq, exp_dist)
    
    print("\n" + "="*80)
    print("✅ EXPLORATION TERMINÉE")
    print("="*80)
    print("\n🎯 Prochaine étape : Créer queries.json avec 55 requêtes basées sur ces insights\n")


if __name__ == "__main__":
    main()