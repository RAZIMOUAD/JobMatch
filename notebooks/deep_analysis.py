#!/usr/bin/env python3
"""
Analyse APPROFONDIE du dataset Healthcare
Focus : Offres non catégorisées + Patterns par catégorie + Skills + Génération des 55 requêtes

Usage: python deep_analysis.py
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import json


def print_section(title):
    """Affiche un titre de section"""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)


def analyze_uncategorized_jobs(df, categories_keywords):
    """Analyse approfondie des offres non catégorisées"""
    print_section("ANALYSE DES OFFRES NON CATÉGORISÉES")
    
    # Identifier les non catégorisées
    all_masks = pd.Series([False] * len(df))
    for keywords in categories_keywords.values():
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        all_masks |= df['title'].str.lower().str.contains(pattern, regex=True, na=False)
    
    uncategorized_df = df[~all_masks]
    
    print(f"\n⚠️ {len(uncategorized_df)} offres non catégorisées ({(len(uncategorized_df)/len(df))*100:.1f}%)")
    
    # Mots fréquents dans ces titres
    uncat_words = []
    for title in uncategorized_df['title']:
        words = str(title).lower().split()
        words = [re.sub(r'[^a-z]', '', w) for w in words if len(re.sub(r'[^a-z]', '', w)) > 2]
        uncat_words.extend(words)
    
    uncat_word_freq = Counter(uncat_words)
    
    print("\n🔤 Top 30 mots dans les titres NON CATÉGORISÉS :")
    print(f"{'Rang':>4s} {'Mot':20s} {'Count':>8s} {'% des non-cat':>15s}")
    print("-"*55)
    
    for rank, (word, count) in enumerate(uncat_word_freq.most_common(30), 1):
        pct = (count / len(uncategorized_df)) * 100
        print(f"{rank:4d} {word:20s} {count:8d} {pct:14.1f}%")
    
    # Bigrams dans les non catégorisées
    uncat_bigrams = []
    for title in uncategorized_df['title']:
        words = str(title).lower().split()
        words = [re.sub(r'[^a-z]', '', w) for w in words if len(re.sub(r'[^a-z]', '', w)) > 2]
        for i in range(len(words)-1):
            uncat_bigrams.append(f"{words[i]} {words[i+1]}")
    
    uncat_bigram_freq = Counter(uncat_bigrams)
    
    print("\n📌 Top 20 BIGRAMS dans les NON CATÉGORISÉS :")
    print(f"{'Rang':>4s} {'Bigram':30s} {'Count':>8s}")
    print("-"*50)
    
    for rank, (bigram, count) in enumerate(uncat_bigram_freq.most_common(20), 1):
        print(f"{rank:4d} {bigram:30s} {count:8d}")
    
    # Exemples de titres non catégorisés
    print("\n📋 EXEMPLES DE TITRES NON CATÉGORISÉS (30 aléatoires) :")
    print("-"*80)
    for i, title in enumerate(uncategorized_df['title'].sample(min(30, len(uncategorized_df))), 1):
        print(f"  {i:2d}. {title}")
    
    # Nouvelles catégories potentielles
    print("\n💡 NOUVELLES CATÉGORIES POTENTIELLES (basées sur les patterns) :")
    print("-"*80)
    
    potential_categories = {
        'Dietitians/Nutritionists': ['dietitian', 'nutritionist', 'dietary'],
        'Lab/Pathology': ['laboratory', 'pathology', 'histology', 'cytology'],
        'Imaging/Radiology Extended': ['imaging', 'radiography', 'mammography'],
        'Environmental Services': ['environmental', 'housekeeping', 'evs'],
        'Recreation Therapy': ['recreation', 'recreational therapist', 'activity'],
        'Quality/Compliance': ['quality', 'compliance', 'accreditation'],
        'Research': ['research', 'clinical trial', 'study coordinator'],
        'Education/Training': ['educator', 'instructor', 'trainer', 'clinical education'],
        'Chaplain/Spiritual': ['chaplain', 'spiritual', 'pastoral'],
        'Transport/Logistics': ['transport', 'logistics', 'supply chain'],
    }
    
    for cat_name, keywords in potential_categories.items():
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        mask = uncategorized_df['title'].str.lower().str.contains(pattern, regex=True, na=False)
        count = mask.sum()
        if count > 0:
            print(f"\n✅ {cat_name} : {count} offres trouvées")
            examples = uncategorized_df[mask]['title'].head(3).tolist()
            for ex in examples:
                print(f"   • {ex}")
    
    return uncategorized_df, uncat_word_freq


def analyze_category_patterns(df, categories_keywords):
    """Analyse les patterns de rédaction par catégorie"""
    print_section("PATTERNS SPÉCIFIQUES PAR CATÉGORIE")
    
    # Analyser les 5 plus grandes catégories
    categories_to_analyze = [
        ('Registered Nurses (RN)', ['registered nurse', 'rn ', ' rn', 'staff nurse']),
        ('Physical Therapists', ['physical therapist', 'pt ', ' pt', 'physiotherapist']),
        ('Medical Assistants', ['medical assistant', ' ma ', 'clinical assistant']),
        ('Licensed Practical Nurses (LPN)', ['lpn', 'licensed practical nurse']),
        ('Certified Nursing Assistants (CNA)', ['cna', 'nursing assistant', 'nurse aide']),
    ]
    
    category_patterns = {}
    
    for cat_name, keywords in categories_to_analyze:
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        mask = df['title'].str.lower().str.contains(pattern, regex=True, na=False)
        subset = df[mask]
        
        if len(subset) == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"📋 {cat_name} ({len(subset)} offres)")
        print(f"{'='*80}")
        
        # Mots spécifiques à cette catégorie
        cat_words = []
        for title in subset['title']:
            words = str(title).lower().split()
            words = [re.sub(r'[^a-z]', '', w) for w in words if len(re.sub(r'[^a-z]', '', w)) > 2]
            cat_words.extend(words)
        
        cat_word_freq = Counter(cat_words)
        
        # Exclure les mots-clés de la catégorie elle-même
        exclude_words = set()
        for kw in keywords:
            exclude_words.update(kw.split())
        
        filtered_words = [(w, c) for w, c in cat_word_freq.most_common(20) 
                         if w not in exclude_words and w not in ['and', 'the', 'for', 'per']]
        
        print(f"\n🔤 Top 15 mots distinctifs :")
        for word, count in filtered_words[:15]:
            pct = (count / len(subset)) * 100
            print(f"   • {word:20s} : {count:4d} ({pct:5.1f}%)")
        
        # Bigrams spécifiques
        cat_bigrams = []
        for title in subset['title']:
            words = str(title).lower().split()
            words = [re.sub(r'[^a-z]', '', w) for w in words if len(re.sub(r'[^a-z]', '', w)) > 2]
            for i in range(len(words)-1):
                cat_bigrams.append(f"{words[i]} {words[i+1]}")
        
        cat_bigram_freq = Counter(cat_bigrams)
        
        # Exclure les bigrams contenant les mots-clés
        filtered_bigrams = [(b, c) for b, c in cat_bigram_freq.most_common(15)
                           if not any(kw.replace(' ', '') in b.replace(' ', '') for kw in keywords)]
        
        print(f"\n📌 Top 10 bigrams distinctifs :")
        for bigram, count in filtered_bigrams[:10]:
            print(f"   • '{bigram}' ({count} occurrences)")
        
        # Exemples variés
        print(f"\n📋 Exemples de titres (5 aléatoires) :")
        for title in subset['title'].sample(min(5, len(subset))):
            print(f"   • {title}")
        
        category_patterns[cat_name] = {
            'count': len(subset),
            'top_words': filtered_words[:10],
            'top_bigrams': filtered_bigrams[:10]
        }
    
    return category_patterns


def analyze_skills(df):
    """Analyse les skills demandés"""
    print_section("ANALYSE DES SKILLS")
    
    skills_available = df['skill_name'].notna().sum()
    print(f"\n✅ Offres avec skills : {skills_available} / {len(df)} ({(skills_available/len(df))*100:.1f}%)")
    
    if skills_available > 0:
        # Extraire tous les skills
        all_skills = []
        for skills in df['skill_name'].dropna():
            # Skills peuvent être des listes ou séparés par virgules
            skills_list = re.split(r'[,;|]', str(skills))
            skills_list = [s.strip().lower() for s in skills_list if len(s.strip()) > 2]
            all_skills.extend(skills_list)
        
        skills_freq = Counter(all_skills)
        
        print(f"\nTotal de skills uniques : {len(skills_freq):,}")
        print(f"Total d'occurrences : {sum(skills_freq.values()):,}")
        
        print("\n📌 Top 40 skills les plus demandés :")
        print(f"{'Rang':>4s} {'Skill':40s} {'Count':>8s} {'%':>8s}")
        print("-"*65)
        
        for rank, (skill, count) in enumerate(skills_freq.most_common(40), 1):
            pct = (count / skills_available) * 100
            print(f"{rank:4d} {skill:40s} {count:8d} {pct:7.1f}%")
        
        return skills_freq
    
    return None


def generate_55_queries(category_counts, bigram_freq, trigram_freq, category_patterns, uncategorized_analysis):
    """Génère les 55 requêtes d'évaluation basées sur les données réelles"""
    print_section("GÉNÉRATION DES 55 REQUÊTES D'ÉVALUATION")
    
    queries = {
        "metadata": {
            "created_date": "2024-12-04",
            "total_queries": 55,
            "dataset_size": 1998,
            "purpose": "Evaluation rigoureuse du moteur de recherche Healthcare"
        },
        "queries": []
    }
    
    query_id = 1
    
    # ========================================================================
    # CATÉGORIE 1 : SPECIFIC (20 requêtes)
    # ========================================================================
    print("\n" + "="*80)
    print("1️⃣ REQUÊTES SPECIFIC (20 requêtes)")
    print("="*80)
    
    specific_queries = [
        # Registered Nurses variations (492 offres)
        ("registered nurse emergency", "specific", "RN avec spécialisation urgence"),
        ("rn intensive care unit", "specific", "RN en soins intensifs"),
        ("travel registered nurse", "specific", "RN en mission"),
        ("registered nurse med surg", "specific", "RN en médecine-chirurgie"),
        
        # Licensed Practical Nurses (82 offres)
        ("licensed practical nurse", "specific", "LPN général"),
        ("lpn home health", "specific", "LPN en soins à domicile"),
        
        # Certified Nursing Assistants (83 offres)
        ("certified nursing assistant", "specific", "CNA général"),
        ("cna telemetry", "specific", "CNA en télémétrie"),
        
        # Physical Therapists (64 offres)
        ("physical therapist", "specific", "PT général"),
        ("physical therapist home health", "specific", "PT à domicile"),
        
        # Medical Assistants (51 offres)
        ("medical assistant", "specific", "MA général"),
        ("certified medical assistant", "specific", "MA certifié"),
        
        # Physicians (71 offres)
        ("physician emergency medicine", "specific", "Médecin urgentiste"),
        ("hospitalist physician", "specific", "Médecin hospitalier"),
        
        # Technicians (184 offres)
        ("surgical technician", "specific", "Technicien chirurgical"),
        ("laboratory technician", "specific", "Technicien de laboratoire"),
        ("radiology technologist", "specific", "Technicien en radiologie"),
        
        # Other specific
        ("occupational therapist", "specific", "Ergothérapeute"),
        ("speech language pathologist", "specific", "Orthophoniste"),
        ("dental hygienist", "specific", "Hygiéniste dentaire"),
    ]
    
    for query_text, category, expected_behavior in specific_queries:
        queries["queries"].append({
            "id": query_id,
            "query": query_text,
            "category": category,
            "expected_behavior": expected_behavior,
            "min_expected_results": 3,
            "relevant_job_ids": []
        })
        print(f"  {query_id:2d}. '{query_text}' → {expected_behavior}")
        query_id += 1
    
    # ========================================================================
    # CATÉGORIE 2 : MODERATE (15 requêtes)
    # ========================================================================
    print("\n" + "="*80)
    print("2️⃣ REQUÊTES MODERATE (15 requêtes)")
    print("="*80)
    
    moderate_queries = [
        ("nurse critical care", "moderate", "Infirmière en soins critiques"),
        ("nurse practitioner", "moderate", "Infirmière praticienne"),
        ("patient care technician", "moderate", "Technicien en soins aux patients"),
        ("case manager healthcare", "moderate", "Gestionnaire de cas médical"),
        ("clinical coordinator", "moderate", "Coordinateur clinique"),
        ("therapist rehabilitation", "moderate", "Thérapeute en réhabilitation"),
        ("medical technician", "moderate", "Technicien médical général"),
        ("healthcare administrator", "moderate", "Administrateur de santé"),
        ("social worker hospital", "moderate", "Travailleur social hospitalier"),
        ("respiratory care practitioner", "moderate", "Praticien en soins respiratoires"),
        ("mental health counselor", "moderate", "Conseiller en santé mentale"),
        ("nurse pediatric", "moderate", "Infirmière pédiatrique"),
        ("clinic manager", "moderate", "Directeur de clinique"),
        ("pharmacy technician", "moderate", "Technicien en pharmacie"),
        ("health educator", "moderate", "Éducateur en santé"),
    ]
    
    for query_text, category, expected_behavior in moderate_queries:
        queries["queries"].append({
            "id": query_id,
            "query": query_text,
            "category": category,
            "expected_behavior": expected_behavior,
            "min_expected_results": 2,
            "relevant_job_ids": []
        })
        print(f"  {query_id:2d}. '{query_text}' → {expected_behavior}")
        query_id += 1
    
    # ========================================================================
    # CATÉGORIE 3 : GENERIC (10 requêtes)
    # ========================================================================
    print("\n" + "="*80)
    print("3️⃣ REQUÊTES GENERIC (10 requêtes)")
    print("="*80)
    
    generic_queries = [
        ("healthcare job", "generic", "N'importe quel job healthcare"),
        ("medical position", "generic", "Poste médical général"),
        ("health services", "generic", "Services de santé"),
        ("hospital job", "generic", "Emploi hospitalier"),
        ("patient care", "generic", "Soins aux patients"),
        ("clinical position", "generic", "Poste clinique"),
        ("nursing job", "generic", "Emploi en soins infirmiers"),
        ("healthcare professional", "generic", "Professionnel de santé"),
        ("medical staff", "generic", "Personnel médical"),
        ("health career", "generic", "Carrière en santé"),
    ]
    
    for query_text, category, expected_behavior in generic_queries:
        queries["queries"].append({
            "id": query_id,
            "query": query_text,
            "category": category,
            "expected_behavior": expected_behavior,
            "min_expected_results": 1,
            "relevant_job_ids": []
        })
        print(f"  {query_id:2d}. '{query_text}' → {expected_behavior}")
        query_id += 1
    
    # ========================================================================
    # CATÉGORIE 4 : OUT_OF_SCOPE (10 requêtes)
    # ========================================================================
    print("\n" + "="*80)
    print("4️⃣ REQUÊTES OUT_OF_SCOPE (10 requêtes)")
    print("="*80)
    
    out_of_scope_queries = [
        ("software engineer", "out_of_scope", "NE doit PAS retourner de résultats"),
        ("salesforce developer", "out_of_scope", "NE doit PAS retourner de résultats"),
        ("data scientist python", "out_of_scope", "NE doit PAS retourner de résultats"),
        ("truck driver", "out_of_scope", "NE doit PAS retourner de résultats"),
        ("accountant financial", "out_of_scope", "NE doit PAS retourner de résultats"),
        ("graphic designer", "out_of_scope", "NE doit PAS retourner de résultats"),
        ("marketing manager", "out_of_scope", "NE doit PAS retourner de résultats"),
        ("construction worker", "out_of_scope", "NE doit PAS retourner de résultats"),
        ("retail sales associate", "out_of_scope", "NE doit PAS retourner de résultats"),
        ("teacher elementary school", "out_of_scope", "NE doit PAS retourner de résultats"),
    ]
    
    for query_text, category, expected_behavior in out_of_scope_queries:
        queries["queries"].append({
            "id": query_id,
            "query": query_text,
            "category": category,
            "expected_behavior": expected_behavior,
            "min_expected_results": 0,
            "relevant_job_ids": []
        })
        print(f"  {query_id:2d}. '{query_text}' → {expected_behavior}")
        query_id += 1
    
    # Sauvegarder
    output_file = '../data/evaluation/queries.json'
    
    # Créer le dossier si nécessaire
    import os
    os.makedirs('../data/evaluation', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(queries, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ FICHIER GÉNÉRÉ")
    print("="*80)
    print(f"\n💾 Fichier sauvegardé : {output_file}")
    print(f"\nTotal de requêtes : {len(queries['queries'])}")
    print(f"  • SPECIFIC      : 20 requêtes")
    print(f"  • MODERATE      : 15 requêtes")
    print(f"  • GENERIC       : 10 requêtes")
    print(f"  • OUT_OF_SCOPE  : 10 requêtes")
    
    return queries


def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print("🔬 ANALYSE APPROFONDIE DU DATASET HEALTHCARE")
    print("="*80)
    print("\nObjectif : Extraire TOUS les insights pour créer les meilleures requêtes\n")
    
    # Charger
    df = pd.read_csv('../data/processed/healthcare_jobs_sample_2000.csv')
    print(f"✅ Dataset chargé : {len(df):,} offres\n")
    
    # Catégories de base
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
    }
    
    # Compter les catégories
    category_counts = {}
    for category, keywords in categories_keywords.items():
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        mask = df['title'].str.lower().str.contains(pattern, regex=True, na=False)
        category_counts[category] = mask.sum()
    
    # Bigrams/Trigrams
    bigrams = []
    for title in df['title'].dropna():
        words = str(title).lower().split()
        words = [re.sub(r'[^a-z]', '', w) for w in words if len(re.sub(r'[^a-z]', '', w)) > 2]
        for i in range(len(words)-1):
            bigrams.append(f"{words[i]} {words[i+1]}")
    bigram_freq = Counter(bigrams)
    
    trigrams = []
    for title in df['title'].dropna():
        words = str(title).lower().split()
        words = [re.sub(r'[^a-z]', '', w) for w in words if len(re.sub(r'[^a-z]', '', w)) > 2]
        for i in range(len(words)-2):
            trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    trigram_freq = Counter(trigrams)
    
    # Analyses approfondies
    uncategorized_df, uncat_word_freq = analyze_uncategorized_jobs(df, categories_keywords)
    category_patterns = analyze_category_patterns(df, categories_keywords)
    skills_freq = analyze_skills(df)
    
    # Génération des 55 requêtes
    queries = generate_55_queries(category_counts, bigram_freq, trigram_freq, 
                                   category_patterns, uncat_word_freq)
    
    print("\n" + "="*80)
    print("✅ ANALYSE APPROFONDIE TERMINÉE")
    print("="*80)
    print("\n🎯 Prochaine étape : Annoter le ground truth avec create_ground_truth.py\n")


if __name__ == "__main__":
    main()