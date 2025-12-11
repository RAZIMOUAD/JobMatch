#!/usr/bin/env python3
"""
SCRIPT 1 : DIAGNOSTIC COMPLET DU DATASET
Vérifie si les jobs problématiques existent réellement dans le dataset

Requêtes critiques identifiées (depuis annotations réelles) :
- #8  : cna telemetry (0/20 pertinents) ❌ ÉCHEC TOTAL
- #10 : physical therapist home health (2/20) ⚠️ TRÈS PEU
- #13 : physician emergency medicine (0/20) ❌ ÉCHEC TOTAL
- #14 : hospitalist physician (1/20) ⚠️ TRÈS PEU
"""

import pandas as pd
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def print_section(title, symbol="="):
    print("\n" + symbol*80)
    print(f"{title}")
    print(symbol*80)


def search_jobs(df, patterns, logic='AND', case_sensitive=False):
    """
    Recherche flexible avec logique AND/OR
    
    Args:
        df: DataFrame
        patterns: Liste de strings à chercher
        logic: 'AND' ou 'OR'
        case_sensitive: Sensibilité à la casse
    """
    if not patterns:
        return df[df['title'].isna()]  # Retourne vide
    
    masks = []
    for pattern in patterns:
        mask = df['title'].str.contains(pattern, case=case_sensitive, na=False, regex=False)
        masks.append(mask)
    
    if logic == 'AND':
        combined = masks[0]
        for mask in masks[1:]:
            combined = combined & mask
    else:  # OR
        combined = masks[0]
        for mask in masks[1:]:
            combined = combined | mask
    
    return df[combined]


def analyze_query_failure(df, query_id, query_text, relevant_job_ids):
    """Analyse détaillée d'une requête avec échec"""
    
    print(f"\n🔍 REQUÊTE #{query_id} : '{query_text}'")
    print(f"   Résultats pertinents annotés : {len(relevant_job_ids)}/20")
    
    # Extraire mots-clés
    keywords = [k for k in query_text.split() if len(k) > 2]
    
    print(f"\n📋 Mots-clés extraits : {keywords}")
    
    # Test 1 : Recherche avec tous les mots-clés (AND)
    results_and = search_jobs(df, keywords, logic='AND')
    print(f"\n   Test AND ({' + '.join(keywords)}) : {len(results_and)} résultats")
    
    if len(results_and) > 0:
        print("   ✅ JOBS TROUVÉS dans le dataset :")
        for idx, row in results_and.head(5).iterrows():
            in_annotation = "✓ ANNOTÉ" if row['job_id'] in relevant_job_ids else ""
            print(f"      • [{row['job_id']}] {row['title'][:70]} {in_annotation}")
        
        if len(results_and) > 5:
            print(f"      ... et {len(results_and) - 5} autres")
        
        return 'FOUND_IN_DATASET'
    
    # Test 2 : Recherche avec mots-clés séparément (OR)
    print(f"\n   Test OR (au moins un mot-clé) :")
    for keyword in keywords:
        results = search_jobs(df, [keyword])
        print(f"      • '{keyword}' : {len(results)} résultats")
    
    # Test 3 : Recherche variations
    print(f"\n   Test variations/synonymes :")
    
    variations = {
        'cna': ['CNA', 'Certified Nursing Assistant', 'Nursing Assistant'],
        'telemetry': ['Telemetry', 'Tele', 'Cardiac Monitor'],
        'physician': ['Physician', 'Doctor', 'MD', 'DO'],
        'emergency': ['Emergency', 'ER', 'Emergency Department', 'ED'],
        'hospitalist': ['Hospitalist', 'Hospital Medicine'],
        'physical': ['Physical Therapist', 'PT', 'Physiotherapist'],
        'home': ['Home Health', 'Home Care', 'Home-Based'],
    }
    
    for main_kw in keywords:
        main_lower = main_kw.lower()
        if main_lower in variations:
            for var in variations[main_lower]:
                results = search_jobs(df, [var])
                if len(results) > 0:
                    print(f"      • '{var}' : {len(results)} résultats")
    
    return 'NOT_FOUND_IN_DATASET'


def main():
    print_section("🔬 DIAGNOSTIC 1 : ANALYSE DU DATASET", "=")
    print("\nObjectif : Vérifier si les jobs problématiques existent dans les 1,998 offres")
    print("Focus : 4 requêtes avec échec total ou quasi-total\n")
    
    # ========================================================================
    # CHARGEMENT DES DONNÉES
    # ========================================================================
    print("📂 Chargement des données...")
    
    try:
        df = pd.read_csv('../data/processed/healthcare_jobs_sample_2000.csv')
        print(f"   ✅ Dataset : {len(df)} offres chargées")
    except FileNotFoundError:
        print("   ❌ Fichier healthcare_jobs_sample_2000.csv introuvable")
        print("   Chemin attendu : ../data/processed/healthcare_jobs_sample_2000.csv")
        return
    
    try:
        with open('../data/evaluation/queries_with_ground_truth.json', 'r') as f:
            data = json.load(f)
        annotations = data['queries']
        print(f"   ✅ Annotations : {len(annotations)} requêtes\n")
    except FileNotFoundError:
        print("   ❌ Fichier queries_with_ground_truth.json introuvable")
        return
    
    # ========================================================================
    # STATISTIQUES GLOBALES
    # ========================================================================
    print_section("📊 STATISTIQUES GLOBALES DU DATASET")
    
    print("\n🏥 Métiers Healthcare (top mots dans titres) :")
    
    metiers_keywords = {
        'Registered Nurse': ['Registered Nurse', 'RN '],
        'Licensed Practical Nurse': ['Licensed Practical', 'LPN'],
        'Certified Nursing Assistant': ['Nursing Assistant', 'CNA'],
        'Physical Therapist': ['Physical Therapist', 'PT '],
        'Occupational Therapist': ['Occupational Therapist', 'OT '],
        'Medical Assistant': ['Medical Assistant', 'MA '],
        'Physician': ['Physician', 'Doctor', ' MD ', ' DO '],
        'Nurse Practitioner': ['Nurse Practitioner', 'NP '],
        'Technician/Tech': ['Technician', 'Tech '],
    }
    
    for metier, keywords in metiers_keywords.items():
        results = search_jobs(df, keywords, logic='OR')
        pct = (len(results) / len(df)) * 100
        print(f"   • {metier:30s} : {len(results):4d} ({pct:5.1f}%)")
    
    print("\n🏥 Spécialisations (dans titres) :")
    
    specializations = {
        'Emergency/ER': ['Emergency', ' ER ', 'Emergency Department'],
        'ICU/Critical Care': ['ICU', 'Critical Care', 'Intensive Care'],
        'Med Surg': ['Med Surg', 'Medical Surgical', 'Med-Surg'],
        'Telemetry': ['Telemetry', ' Tele '],
        'Home Health': ['Home Health', 'Home Care', 'Home-Based'],
        'Pediatric': ['Pediatric', 'Peds', 'PICU', 'NICU'],
        'OR/Surgery': ['Operating Room', ' OR ', 'Surgical', 'Surgery'],
    }
    
    for spec, keywords in specializations.items():
        results = search_jobs(df, keywords, logic='OR')
        pct = (len(results) / len(df)) * 100
        print(f"   • {spec:25s} : {len(results):4d} ({pct:5.1f}%)")
    
    # ========================================================================
    # ANALYSE DES 4 REQUÊTES PROBLÉMATIQUES
    # ========================================================================
    print_section("🔴 ANALYSE DES REQUÊTES AVEC ÉCHEC")
    
    problematic_queries = [
        (8, 'cna telemetry', 0),
        (13, 'physician emergency medicine', 0),
        (14, 'hospitalist physician', 1),
        (10, 'physical therapist home health', 2),
    ]
    
    results_summary = {}
    
    for query_id, query_text, n_annotated in problematic_queries:
        query_data = next((q for q in annotations if q['id'] == query_id), None)
        if not query_data:
            continue
        
        relevant_ids = query_data['relevant_job_ids']
        
        status = analyze_query_failure(df, query_id, query_text, relevant_ids)
        results_summary[query_id] = status
        
        print("\n" + "-"*80)
    
    # ========================================================================
    # ANALYSE DÉTAILLÉE : CNA TELEMETRY (cas critique)
    # ========================================================================
    print_section("🔬 FOCUS : CNA TELEMETRY (Requête #8 - ÉCHEC TOTAL)")
    
    print("\n📊 Distribution des jobs Telemetry par métier :")
    tele_jobs = search_jobs(df, ['Telemetry', 'Tele'], logic='OR')
    print(f"   Total jobs Telemetry : {len(tele_jobs)}")
    
    if len(tele_jobs) > 0:
        print("\n   Breakdown par métier :")
        for keyword in ['RN', 'LPN', 'CNA', 'Tech', 'Nurse']:
            subset = search_jobs(tele_jobs, [keyword])
            if len(subset) > 0:
                print(f"      • {keyword:15s} : {len(subset):3d} jobs")
                if keyword == 'CNA':
                    print(f"         Exemples :")
                    for idx, row in subset.head(3).iterrows():
                        print(f"         → [{row['job_id']}] {row['title'][:65]}")
    
    # ========================================================================
    # RÉSUMÉ ET RECOMMANDATIONS
    # ========================================================================
    print_section("📋 RÉSUMÉ DES CONSTATS", "=")
    
    print("\n🔴 PROBLÈMES IDENTIFIÉS :\n")
    
    for query_id, query_text, n_annotated in problematic_queries:
        status = results_summary.get(query_id, 'UNKNOWN')
        
        if status == 'FOUND_IN_DATASET':
            print(f"   #{query_id} '{query_text}':")
            print(f"      ⚠️  Jobs EXISTENT dans dataset mais seulement {n_annotated}/20 trouvés")
            print(f"      → Problème : ALGORITHME ou RANKING")
        else:
            print(f"   #{query_id} '{query_text}':")
            print(f"      ❌ Jobs ABSENTS ou TRÈS RARES dans dataset ({n_annotated}/20)")
            print(f"      → Problème : DATASET INCOMPLET")
        print()
    
    print("🎯 RECOMMANDATIONS :\n")
    print("   1. Si jobs EXISTENT mais non trouvés :")
    print("      → Passer au DIAGNOSTIC 2 (vérifier encodage SBERT + ranking)")
    print("      → Problème probable : algorithme de recherche défaillant")
    print()
    print("   2. Si jobs ABSENTS du dataset :")
    print("      → Documenter les limitations dans le rapport")
    print("      → Considérer enrichir le dataset (si critique)")
    print("      → Adapter les conclusions de l'évaluation")
    
    print_section("✅ DIAGNOSTIC 1 TERMINÉ", "=")
    print("\n🔬 Prochaine étape : python diagnostic_algorithm.py\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption détectée")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()