#!/usr/bin/env python3
"""
Script de test pour vérifier la méthode search()
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.search import JobSearchEngine

# Charger le système
print("🔧 Chargement du système...")
engine = JobSearchEngine(data_dir='../data/processed', debug=False)
print(f"✅ Système chargé : {len(engine.jobs_df)} offres\n")

# Tester une recherche
test_query = "registered nurse"
print(f"🔍 Test de recherche : '{test_query}'")
print("-" * 80)

# Tester une recherche
test_query = "registered nurse"
print(f"🔍 Test de recherche : '{test_query}'")
print("-" * 80)

search_result = engine.search(test_query, k=5)

print(f"\n📊 Structure du résultat :")
print(f"   • Type de requête    : {search_result['query_type']}")
print(f"   • Résultats stricts  : {search_result['strict_count']}")
print(f"   • Résultats génériques: {search_result['generic_count']}")

# Combiner tous les résultats
all_results = search_result['results'] + search_result['generic_results']

if all_results:
    print(f"\n📋 Structure du premier résultat :")
    first_result = all_results[0]
    
    for key, value in first_result.items():
        if isinstance(value, str) and len(value) > 60:
            value = value[:60] + "..."
        print(f"   • {key:20s} : {value}")
    
    print("\n✅ Clés disponibles :", list(first_result.keys()))
    
    if 'distance' in first_result:
        print("✅ La clé 'distance' existe")
        print(f"   Exemple distance : {first_result['distance']:.4f}")
    else:
        print("❌ La clé 'distance' n'existe pas")
        
    print(f"\n📊 Distances des {len(all_results)} résultats :")
    for i, res in enumerate(all_results, 1):
        print(f"   {i}. Distance: {res['distance']:.4f} - {res['title'][:50]}")
else:
    print("❌ Aucun résultat")