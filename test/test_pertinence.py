"""
Test de pertinence - Trouver le seuil optimal
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search import JobSearchEngine

# Initialiser
print("Initialisation du moteur de recherche...")
engine = JobSearchEngine(data_dir='../data/processed')

# Tests
print("\n" + "="*80)
print("TEST DE PERTINENCE - Analyse des distances")
print("="*80)

test_queries = {
    "PERTINENT": [
        "emergency room nurse",
        "physical therapist sports rehabilitation",
        "medical assistant entry level",
        "dental hygienist part time",
        "mental health counselor licensed"
    ],
    "NON PERTINENT": [
        "Salesforce developer",
        "software engineer Python",
        "data scientist machine learning",
        "truck driver CDL",
        "accountant finance tax",
        "graphic designer Adobe"
    ],
    "ZONE GRISE": [
        "IT support",
        "receptionist",
        "manager",
        "administrative assistant"
    ]
}

results_all = {}

for category, queries in test_queries.items():
    print(f"\n{'='*80}")
    print(f"📊 CATÉGORIE : {category}")
    print(f"{'='*80}")
    
    results_all[category] = []
    
    for query in queries:
        # Chercher sans filtre
        results = engine.search(query, k=10)
        
        if len(results) > 0:
            distances = [r['distance'] for r in results]
            min_dist = min(distances)
            max_dist = max(distances)
            avg_dist = sum(distances) / len(distances)
        else:
            min_dist = max_dist = avg_dist = float('inf')
        
        results_all[category].append({
            'query': query,
            'min': min_dist,
            'max': max_dist,
            'avg': avg_dist,
            'count': len(results)
        })
        
        print(f"\n🔍 '{query}'")
        print(f"   Distance min : {min_dist:.3f}")
        print(f"   Distance max : {max_dist:.3f}")
        print(f"   Distance moy : {avg_dist:.3f}")
        
        if len(results) > 0:
            print(f"   Top 3 titres :")
            for i, r in enumerate(results[:3], 1):
                print(f"      {i}. [{r['distance']:.3f}] {r['title']}")

# Analyse globale
print("\n" + "="*80)
print("📊 ANALYSE GLOBALE")
print("="*80)

import numpy as np

for category, results_list in results_all.items():
    distances_min = [r['min'] for r in results_list if r['min'] != float('inf')]
    
    if distances_min:
        print(f"\n{category} ({len(distances_min)} requêtes) :")
        print(f"   Distance min globale : {min(distances_min):.3f}")
        print(f"   Distance max globale : {max(distances_min):.3f}")
        print(f"   Moyenne des min      : {np.mean(distances_min):.3f}")
        print(f"   Écart-type           : {np.std(distances_min):.3f}")

# Recommandation
print("\n" + "="*80)
print("💡 RECOMMANDATION DE SEUIL")
print("="*80)

pertinent_distances = [r['max'] for r in results_all['PERTINENT'] if r['max'] != float('inf')]
non_pertinent_distances = [r['min'] for r in results_all['NON PERTINENT'] if r['min'] != float('inf')]

if pertinent_distances and non_pertinent_distances:
    max_pertinent = max(pertinent_distances)
    min_non_pertinent = min(non_pertinent_distances)
    
    # Seuil optimal = milieu entre les deux
    recommended_threshold = (max_pertinent + min_non_pertinent) / 2
    
    print(f"\nDistance max des requêtes PERTINENTES     : {max_pertinent:.3f}")
    print(f"Distance min des requêtes NON PERTINENTES : {min_non_pertinent:.3f}")
    print(f"\n🎯 SEUIL RECOMMANDÉ : {recommended_threshold:.3f}")
    print(f"\nUtiliser dans search.py :")
    print(f"   distance_threshold={recommended_threshold:.2f}")
    
    # Vérification
    print(f"\n✅ Avec ce seuil :")
    
    # Combien de pertinents passent ?
    pertinent_pass = sum(1 for r in results_all['PERTINENT'] if r['max'] <= recommended_threshold)
    print(f"   Requêtes pertinentes acceptées : {pertinent_pass}/{len(results_all['PERTINENT'])} ({pertinent_pass/len(results_all['PERTINENT'])*100:.0f}%)")
    
    # Combien de non-pertinents sont rejetés ?
    non_pertinent_reject = sum(1 for r in results_all['NON PERTINENT'] if r['min'] > recommended_threshold)
    print(f"   Requêtes non-pertinentes rejetées : {non_pertinent_reject}/{len(results_all['NON PERTINENT'])} ({non_pertinent_reject/len(results_all['NON PERTINENT'])*100:.0f}%)")

print("\n" + "="*80)