#!/usr/bin/env python3
"""
SCRIPT 3 : DIAGNOSTIC DES ANNOTATIONS
Analyse statistique complète du ground truth annoté

Analyses :
1. Distribution des résultats pertinents par catégorie
2. Distribution des distances pour pertinents vs non-pertinents
3. Chevauchement des distributions (problème de seuil unique)
4. Cohérence des annotations
5. Recommandations pour l'évaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def print_section(title, symbol="="):
    print("\n" + symbol*80)
    print(f"{title}")
    print(symbol*80)


def calculate_distribution_stats(values):
    """Calcule des statistiques sur une distribution"""
    if not values:
        return None
    
    return {
        'count': len(values),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
    }


def analyze_distance_distributions(annotations, engine):
    """Analyse les distributions de distances pour pertinents vs non-pertinents"""
    
    print_section("📊 ANALYSE DES DISTANCES")
    
    # Collecter les distances par catégorie
    distances_by_category = {
        'specific': {'relevant': [], 'irrelevant': []},
        'moderate': {'relevant': [], 'irrelevant': []},
        'generic': {'relevant': [], 'irrelevant': []},
    }
    
    print("\n🔍 Calcul des distances pour chaque requête...")
    print("   (Cela peut prendre 1-2 minutes)\n")
    
    for i, query_data in enumerate(annotations, 1):
        query_text = query_data['query']
        category = query_data['category']
        relevant_ids = set(query_data['relevant_job_ids'])
        
        if category == 'out_of_scope':
            continue
        
        # Faire la recherche
        search_result = engine.search(query_text, k=20, strict_threshold=10.0, generic_threshold=10.0)
        all_results = search_result['results'] + search_result['generic_results']
        
        # Collecter les distances
        for result in all_results[:20]:
            job_id = result['job_id']
            distance = result['distance']
            
            if job_id in relevant_ids:
                distances_by_category[category]['relevant'].append(distance)
            else:
                distances_by_category[category]['irrelevant'].append(distance)
        
        if i % 10 == 0:
            print(f"   Progression : {i}/{len(annotations)} requêtes")
    
    # Afficher les statistiques
    print_section("📈 STATISTIQUES DES DISTANCES")
    
    for category in ['specific', 'moderate', 'generic']:
        print(f"\n🏷️  Catégorie : {category.upper()}")
        
        relevant_dists = distances_by_category[category]['relevant']
        irrelevant_dists = distances_by_category[category]['irrelevant']
        
        if not relevant_dists:
            print("   ⚠️  Aucune distance disponible pour cette catégorie")
            continue
        
        rel_stats = calculate_distribution_stats(relevant_dists)
        irrel_stats = calculate_distribution_stats(irrelevant_dists) if irrelevant_dists else None
        
        print(f"\n   ✅ PERTINENTS ({rel_stats['count']} résultats) :")
        print(f"      Min    : {rel_stats['min']:.3f}")
        print(f"      Q25    : {rel_stats['q25']:.3f}")
        print(f"      Median : {rel_stats['median']:.3f}")
        print(f"      Q75    : {rel_stats['q75']:.3f}")
        print(f"      Max    : {rel_stats['max']:.3f}")
        print(f"      Mean   : {rel_stats['mean']:.3f} ± {rel_stats['std']:.3f}")
        
        if irrel_stats:
            print(f"\n   ❌ NON PERTINENTS ({irrel_stats['count']} résultats) :")
            print(f"      Min    : {irrel_stats['min']:.3f}")
            print(f"      Q25    : {irrel_stats['q25']:.3f}")
            print(f"      Median : {irrel_stats['median']:.3f}")
            print(f"      Q75    : {irrel_stats['q75']:.3f}")
            print(f"      Max    : {irrel_stats['max']:.3f}")
            print(f"      Mean   : {irrel_stats['mean']:.3f} ± {irrel_stats['std']:.3f}")
            
            # Analyser le chevauchement
            overlap_start = max(rel_stats['min'], irrel_stats['min'])
            overlap_end = min(rel_stats['max'], irrel_stats['max'])
            
            if overlap_end > overlap_start:
                print(f"\n   ⚠️  CHEVAUCHEMENT détecté : [{overlap_start:.3f}, {overlap_end:.3f}]")
                print(f"      → Un seuil unique NE peut PAS séparer parfaitement")
                
                # Calculer le meilleur seuil (point d'équilibre)
                best_threshold = (rel_stats['max'] + irrel_stats['min']) / 2
                print(f"      → Seuil optimal empirique : {best_threshold:.3f}")
            else:
                print(f"\n   ✅ PAS de chevauchement : seuil = {rel_stats['max']:.3f}")
    
    return distances_by_category


def analyze_annotation_consistency(annotations):
    """Analyse la cohérence des annotations"""
    
    print_section("🔍 COHÉRENCE DES ANNOTATIONS")
    
    # Grouper par catégorie
    by_category = defaultdict(list)
    for query_data in annotations:
        category = query_data['category']
        n_relevant = len(query_data['relevant_job_ids'])
        by_category[category].append(n_relevant)
    
    print("\n📊 Distribution des résultats pertinents :\n")
    
    for category in ['specific', 'moderate', 'generic', 'out_of_scope']:
        values = by_category[category]
        if not values:
            continue
        
        stats = calculate_distribution_stats(values)
        
        print(f"   {category.upper():15s} ({len(values):2d} requêtes) :")
        print(f"      Min    : {stats['min']:.0f}/20")
        print(f"      Q25    : {stats['q25']:.0f}/20")
        print(f"      Median : {stats['median']:.0f}/20")
        print(f"      Q75    : {stats['q75']:.0f}/20")
        print(f"      Max    : {stats['max']:.0f}/20")
        print(f"      Mean   : {stats['mean']:.1f}/20")
        print()
    
    # Identifier les anomalies
    print_section("⚠️  ANOMALIES DÉTECTÉES")
    
    anomalies = []
    
    for query_data in annotations:
        category = query_data['category']
        n_relevant = len(query_data['relevant_job_ids'])
        
        # Règles d'anomalies
        is_anomaly = False
        reason = ""
        
        if category == 'specific' and n_relevant == 0:
            is_anomaly = True
            reason = "SPECIFIC avec 0 pertinents (échec total)"
        elif category == 'specific' and n_relevant <= 2:
            is_anomaly = True
            reason = "SPECIFIC avec très peu de pertinents"
        elif category == 'moderate' and n_relevant <= 3:
            is_anomaly = True
            reason = "MODERATE avec très peu de pertinents"
        elif category == 'generic' and n_relevant < 20:
            is_anomaly = True
            reason = "GENERIC devrait avoir 20/20 pertinents"
        elif category == 'out_of_scope' and n_relevant > 5:
            is_anomaly = True
            reason = "OUT_OF_SCOPE avec trop de pertinents"
        
        if is_anomaly:
            anomalies.append({
                'id': query_data['id'],
                'query': query_data['query'],
                'category': category,
                'n_relevant': n_relevant,
                'reason': reason
            })
    
    if anomalies:
        print(f"\n  {len(anomalies)} anomalies détectées :\n")
        for anom in anomalies:
            print(f"   #{anom['id']:2d} | {anom['query']:40s} | {anom['category']:12s} | {anom['n_relevant']:2d}/20")
            print(f"        → {anom['reason']}")
            print()
    else:
        print("\n✅ Aucune anomalie majeure détectée")


def generate_recommendations(annotations, distances_by_category):
    """Génère des recommandations pour l'évaluation"""
    
    print_section("🎯 RECOMMANDATIONS POUR L'ÉVALUATION", "=")
    
    print("\n📋 STRATÉGIES D'ÉVALUATION :\n")
    
    # Compter les problèmes
    n_problems = sum(1 for q in annotations if len(q['relevant_job_ids']) <= 2 and q['category'] in ['specific', 'moderate'])
    
    print("1. GESTION DES REQUÊTES PROBLÉMATIQUES :\n")
    print(f"   • Nombre de requêtes avec 0-2 pertinents : {n_problems}")
    print(f"   • Recommandation : Évaluer AVEC et SANS ces requêtes")
    print(f"   • Documenter les limitations du dataset")
    print()
    
    print("2. SEUILS PAR CATÉGORIE :\n")
    
    # Proposer des seuils optimaux
    for category in ['specific', 'moderate', 'generic']:
        rel_dists = distances_by_category[category]['relevant']
        irrel_dists = distances_by_category[category]['irrelevant']
        
        if not rel_dists:
            continue
        
        rel_stats = calculate_distribution_stats(rel_dists)
        
        if irrel_dists:
            irrel_stats = calculate_distribution_stats(irrel_dists)
            proposed_threshold = (rel_stats['q75'] + irrel_stats['q25']) / 2
        else:
            proposed_threshold = rel_stats['q75']
        
        print(f"   • {category.upper():10s} : Seuil suggéré = {proposed_threshold:.2f}")
        print(f"      - Pertinents    : {rel_stats['q75']:.3f} (Q75)")
        if irrel_dists:
            print(f"      - Non pertinents : {irrel_stats['q25']:.3f} (Q25)")
        print()
    
    print("3. MÉTRIQUES À CALCULER :\n")
    print("   • Precision@K (K=5, 10, 20)")
    print("   • Recall")
    print("   • F1-Score")
    print("   • MAP (Mean Average Precision)")
    print("   • Calculer pour CHAQUE catégorie séparément")
    print()
    
    print("4. TESTS DE SEUIL :\n")
    print("   • Tester seuils : 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0")
    print("   • Générer courbes Precision-Recall")
    print("   • Identifier le seuil qui maximise F1-Score")
    print()
    
    print("5. RAPPORT FINAL :\n")
    print("   • Documenter les échecs (CNA Telemetry, Physician Emergency, etc.)")
    print("   • Expliquer pourquoi un seuil unique ne fonctionne pas")
    print("   • Proposer des solutions (seuils dynamiques, fine-tuning, etc.)")


def main():
    print_section("🔬 DIAGNOSTIC 3 : ANALYSE DES ANNOTATIONS", "=")
    print("\nObjectif : Analyser statistiquement le ground truth annoté")
    print("Identifier les problèmes et proposer des solutions\n")
    
    # ========================================================================
    # CHARGEMENT
    # ========================================================================
    print("📂 Chargement des données...")
    
    try:
        with open('../data/evaluation/queries_with_ground_truth.json', 'r') as f:
            data = json.load(f)
        annotations = data['queries']
        print(f"   ✅ Annotations : {len(annotations)} requêtes")
    except FileNotFoundError:
        print("   ❌ Fichier queries_with_ground_truth.json introuvable")
        return
    
    try:
        from src.search import JobSearchEngine
        engine = JobSearchEngine(data_dir='../data/processed', debug=False)
        print(f"   ✅ SearchEngine : {len(engine.jobs_df)} offres\n")
    except Exception as e:
        print(f"   ❌ Erreur chargement SearchEngine : {e}")
        return
    
    # ========================================================================
    # ANALYSES
    # ========================================================================
    
    # Analyse 1 : Cohérence
    analyze_annotation_consistency(annotations)
    
    # Analyse 2 : Distributions de distances
    distances_by_category = analyze_distance_distributions(annotations, engine)
    
    # Recommandations
    generate_recommendations(annotations, distances_by_category)
    
    # ========================================================================
    # RÉSUMÉ
    # ========================================================================
    print_section("✅ DIAGNOSTIC 3 TERMINÉ", "=")
    print("\n🎯 PROCHAINES ÉTAPES :\n")
    print("   1. Si diagnostics 1-3 révèlent des problèmes majeurs :")
    print("      → Corriger les problèmes AVANT l'évaluation")
    print("      → Dataset incomplet ? → Enrichir ou documenter")
    print("      → Algorithme cassé ? → Débugger")
    print()
    print("   2. Si tout semble OK :")
    print("      → Lancer l'évaluation : python evaluate_system.py")
    print("      → Calculer les métriques avec différents seuils")
    print("      → Générer les graphiques")
    print()
    print("   3. Documenter dans le rapport :")
    print("      → Tous les constats des 3 diagnostics")
    print("      → Les limitations identifiées")
    print("      → Les recommandations pour améliorer le système\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption détectée")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()