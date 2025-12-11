#!/usr/bin/env python3
"""
SCRIPT 2 : DIAGNOSTIC DE L'ALGORITHME DE RECHERCHE
Vérifie que SBERT encode correctement et que le ranking fonctionne

Tests :
1. Encodage SBERT des requêtes problématiques
2. Calcul des distances (L2 euclidienne)
3. Ranking des résultats (top-20)
4. Comparaison avec les annotations manuelles
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json


def print_section(title, symbol="="):
    print("\n" + symbol*80)
    print(f"{title}")
    print(symbol*80)


def test_encoding(model, test_queries):
    """Test l'encodage SBERT"""
    print_section("🧪 TEST 1 : ENCODAGE SBERT")
    
    print("\n📊 Vérification de l'encodage :")
    
    for query in test_queries:
        embedding = model.encode(query)
        
        print(f"\n   Requête : '{query}'")
        print(f"      Shape       : {embedding.shape}")
        print(f"      Dimension   : {embedding.shape[0]}D (attendu: 384D)")
        print(f"      Range       : [{embedding.min():.3f}, {embedding.max():.3f}]")
        print(f"      Norme L2    : {np.linalg.norm(embedding):.3f}")
        
        # Vérifications
        if embedding.shape[0] != 384:
            print(f"      ❌ ERREUR : Dimension incorrecte!")
        else:
            print(f"      ✅ Encodage OK")


def test_similarity(model, query, similar_titles, dissimilar_titles):
    """Test la similarité sémantique"""
    
    query_emb = model.encode(query)
    
    print(f"\n🔍 Test similarité pour : '{query}'")
    print(f"\n   ✅ Titres SIMILAIRES (distance attendue < 0.8) :")
    
    for title in similar_titles:
        title_emb = model.encode(title)
        distance = np.linalg.norm(query_emb - title_emb)
        
        status = "✅" if distance < 0.8 else "⚠️"
        print(f"      {status} [{distance:.3f}] {title}")
    
    print(f"\n   ❌ Titres DIFFÉRENTS (distance attendue > 0.9) :")
    
    for title in dissimilar_titles:
        title_emb = model.encode(title)
        distance = np.linalg.norm(query_emb - title_emb)
        
        status = "✅" if distance > 0.9 else "⚠️"
        print(f"      {status} [{distance:.3f}] {title}")


def test_ranking(engine, query_id, query_text, relevant_job_ids, df):
    """Test le ranking pour une requête spécifique"""
    
    print(f"\n🎯 Requête #{query_id} : '{query_text}'")
    print(f"   Jobs pertinents attendus : {len(relevant_job_ids)}")
    
    # Faire la recherche
    search_result = engine.search(query_text, k=20, strict_threshold=10.0, generic_threshold=10.0)
    all_results = search_result['results'] + search_result['generic_results']
    
    if len(all_results) == 0:
        print(f"   ❌ AUCUN résultat retourné!")
        return
    
    print(f"   Résultats retournés : {len(all_results)}")
    
    # Analyser le ranking
    print(f"\n   📋 Top-20 résultats :")
    
    n_relevant_in_top20 = 0
    n_irrelevant_in_top20 = 0
    
    for i, result in enumerate(all_results[:20], 1):
        job_id = result['job_id']
        distance = result['distance']
        title = result['title'][:60]
        
        is_relevant = job_id in relevant_job_ids
        
        if is_relevant:
            n_relevant_in_top20 += 1
            print(f"      ✅ #{i:2d} [{distance:.3f}] {title}")
        else:
            n_irrelevant_in_top20 += 1
            if i <= 5:  # Montrer les 5 premiers non pertinents
                print(f"      ❌ #{i:2d} [{distance:.3f}] {title}")
    
    print(f"\n   📊 Résumé :")
    print(f"      Pertinents dans top-20    : {n_relevant_in_top20}/{len(relevant_job_ids)}")
    print(f"      Non pertinents dans top-20 : {n_irrelevant_in_top20}")
    
    # Trouver où sont les jobs pertinents manquants
    found_ids = [r['job_id'] for r in all_results]
    missing_ids = [job_id for job_id in relevant_job_ids if job_id not in found_ids]
    
    if missing_ids:
        print(f"\n   ⚠️  {len(missing_ids)} jobs pertinents ABSENTS du top-20 :")
        
        # Chercher leur position réelle
        for job_id in missing_ids[:3]:  # Montrer 3 exemples
            job = df[df['job_id'] == job_id].iloc[0]
            print(f"      → [{job_id}] {job['title'][:60]}")
            
            # Calculer sa distance
            job_emb = engine.model.encode(job['title'])
            query_emb = engine.model.encode(query_text)
            distance = np.linalg.norm(job_emb - query_emb)
            
            print(f"         Distance calculée : {distance:.3f}")
    
    # Analyse des distances
    distances = [r['distance'] for r in all_results[:20]]
    print(f"\n   📊 Statistiques distances (top-20) :")
    print(f"      Min : {min(distances):.3f}")
    print(f"      Max : {max(distances):.3f}")
    print(f"      Moy : {np.mean(distances):.3f}")
    print(f"      Med : {np.median(distances):.3f}")


def main():
    print_section("🔬 DIAGNOSTIC 2 : ALGORITHME DE RECHERCHE", "=")
    print("\nObjectif : Vérifier que SBERT et le ranking fonctionnent correctement")
    print("Tests : Encodage, Similarité, Ranking\n")
    
    # ========================================================================
    # CHARGEMENT
    # ========================================================================
    print("📂 Chargement des données...")
    
    try:
        from src.search import JobSearchEngine
        engine = JobSearchEngine(data_dir='../data/processed', debug=False)
        print(f"   ✅ SearchEngine : {len(engine.jobs_df)} offres")
    except Exception as e:
        print(f"   ❌ Erreur chargement SearchEngine : {e}")
        return
    
    try:
        df = pd.read_csv('../data/processed/healthcare_jobs_sample_2000.csv')
        print(f"   ✅ Dataset : {len(df)} offres")
    except FileNotFoundError:
        print("   ❌ Dataset introuvable")
        return
    
    try:
        with open('../data/evaluation/queries_with_ground_truth.json', 'r') as f:
            data = json.load(f)
        annotations = data['queries']
        print(f"   ✅ Annotations : {len(annotations)} requêtes\n")
    except FileNotFoundError:
        print("   ❌ Annotations introuvables")
        return
    
    model = engine.model
    
    # ========================================================================
    # TEST 1 : ENCODAGE
    # ========================================================================
    test_queries = [
        "registered nurse emergency",
        "cna telemetry",
        "physician emergency medicine",
        "hospitalist physician",
    ]
    
    test_encoding(model, test_queries)
    
    # ========================================================================
    # TEST 2 : SIMILARITÉ SÉMANTIQUE
    # ========================================================================
    print_section("🧪 TEST 2 : SIMILARITÉ SÉMANTIQUE")
    
    # Test 1 : CNA Telemetry
    test_similarity(
        model,
        query="cna telemetry",
        similar_titles=[
            "Certified Nursing Assistant - Telemetry Unit",
            "CNA Telemetry Full Time",
            "Nursing Assistant Cardiac Telemetry",
        ],
        dissimilar_titles=[
            "Registered Nurse Telemetry",
            "Physical Therapist",
            "Medical Receptionist",
        ]
    )
    
    # Test 2 : Physician Emergency
    test_similarity(
        model,
        query="physician emergency medicine",
        similar_titles=[
            "Emergency Medicine Physician",
            "Physician - Emergency Department",
            "ER Doctor / Emergency Physician",
        ],
        dissimilar_titles=[
            "Physician Assistant Emergency",
            "Emergency Room Nurse",
            "Hospitalist Physician",
        ]
    )
    
    # ========================================================================
    # TEST 3 : RANKING
    # ========================================================================
    print_section("🧪 TEST 3 : RANKING DES RÉSULTATS")
    
    problematic_queries = [
        (8, 'cna telemetry', 0),
        (13, 'physician emergency medicine', 0),
        (14, 'hospitalist physician', 1),
        (10, 'physical therapist home health', 2),
    ]
    
    for query_id, query_text, n_annotated in problematic_queries:
        query_data = next((q for q in annotations if q['id'] == query_id), None)
        if not query_data:
            continue
        
        relevant_ids = query_data['relevant_job_ids']
        
        test_ranking(engine, query_id, query_text, relevant_ids, df)
        print("\n" + "-"*80)
    
    # ========================================================================
    # TEST 4 : COMPARAISON AVEC SUCCÈS
    # ========================================================================
    print_section("🧪 TEST 4 : COMPARAISON AVEC REQUÊTES RÉUSSIES")
    
    print("\n📊 Comparons avec une requête qui FONCTIONNE BIEN :\n")
    
    # Requête #5 : LPN (20/20 pertinents)
    query_data = next((q for q in annotations if q['id'] == 5), None)
    if query_data:
        test_ranking(engine, 5, query_data['query'], query_data['relevant_job_ids'], df)
    
    # ========================================================================
    # RÉSUMÉ
    # ========================================================================
    print_section("📋 RÉSUMÉ DES CONSTATS", "=")
    
    print("\n🔍 CONSTATS :\n")
    print("   1. Encodage SBERT :")
    print("      → Vérifier si dimension = 384D pour toutes les requêtes")
    print("      → Vérifier si la norme L2 est raisonnable (0.5-1.5)")
    print()
    print("   2. Similarité sémantique :")
    print("      → Les titres similaires ont-ils des distances < 0.8 ?")
    print("      → Les titres différents ont-ils des distances > 0.9 ?")
    print("      → Si NON → Modèle SBERT inadapté au domaine Healthcare")
    print()
    print("   3. Ranking :")
    print("      → Les jobs pertinents sont-ils dans le top-20 ?")
    print("      → Si NON mais EXISTENT dans dataset → Problème de ranking")
    print("      → Distances des pertinents vs non-pertinents se chevauchent ?")
    
    print_section("✅ DIAGNOSTIC 2 TERMINÉ", "=")
    print("\n🔬 Prochaine étape : python diagnostic_annotations.py\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interruption détectée")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()