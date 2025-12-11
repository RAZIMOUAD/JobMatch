#!/usr/bin/env python3
"""
Outil d'annotation SEMI-AUTOMATIQUE du ground truth
Pré-sélectionne automatiquement les résultats pertinents, tu valides/modifies

Usage: python annotation_tool.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import pandas as pd
from src.search import JobSearchEngine


def load_queries(filepath):
    """Charge les requêtes depuis queries.json"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def auto_select_relevant(query_category, results, distances):
    """
    Pré-sélection automatique basée sur des règles heuristiques
    
    Règles :
    - SPECIFIC : distance < 0.7 → pertinent
    - MODERATE : distance < 0.8 → pertinent
    - GENERIC : distance < 0.9 → pertinent
    - OUT_OF_SCOPE : RIEN (tout rejeté par défaut)
    """
    auto_selected = []
    
    if query_category == "out_of_scope":
        # Rien de sélectionné par défaut
        return []
    
    # Seuils par catégorie
    thresholds = {
        "specific": 0.7,
        "moderate": 0.8,
        "generic": 0.9
    }
    
    threshold = thresholds.get(query_category, 0.8)
    
    for i, (result, dist) in enumerate(zip(results, distances)):
        if dist < threshold:
            auto_selected.append(i + 1)  # 1-indexed
    
    return auto_selected


def display_results(query_data, results, distances, auto_selected):
    """Affiche les résultats de manière claire"""
    query = query_data['query']
    category = query_data['category']
    expected = query_data['expected_behavior']
    
    print("\n" + "="*80)
    print(f"REQUÊTE {query_data['id']}/{55}")
    print("="*80)
    print(f"📝 Texte      : '{query}'")
    print(f"🏷️  Catégorie : {category.upper()}")
    print(f"💡 Attendu   : {expected}")
    
    if len(results) == 0:
        print("\n⚠️  AUCUN résultat trouvé (distance > 1.0)")
        return
    
    print(f"\n📊 {len(results)} résultats trouvés :")
    print(f"\n{'Sel':>3s} {'#':>3s} {'Dist':>6s} {'ID':>8s} {'Titre':s}")
    print("-"*80)
    
    for i, (result, dist) in enumerate(zip(results, distances), 1):
        is_selected = "✓" if i in auto_selected else " "
        job_id = result['job_id']
        title = result['title'][:60]  # Tronquer si trop long
        
        print(f"[{is_selected}] {i:2d}. {dist:5.3f}  {job_id:8d}  {title}")


def get_user_validation(auto_selected, num_results, category):
    """Demande validation/modification à l'utilisateur"""
    
    if category == "out_of_scope":
        print("\n⚠️  Requête OUT_OF_SCOPE → Aucun résultat ne devrait être pertinent")
        response = input("\nConfirmes-tu qu'AUCUN résultat n'est pertinent ? (o/n) [o] : ").strip().lower()
        
        if response == '' or response == 'o':
            return []
        else:
            selected = input("⚠️  Entre les numéros des résultats pertinents (ex: 1,3,5) : ").strip()
            if selected:
                return [int(x.strip()) for x in selected.split(',')]
            return []
    
    if category == "generic":
        print("\nℹ️  Requête GENERIC → Accepter tout job Healthcare raisonnable")
    
    print(f"\n🤖 Pré-sélection automatique : {len(auto_selected)} résultats")
    if auto_selected:
        print(f"   Numéros : {', '.join(map(str, auto_selected))}")
    
    print("\nOptions :")
    print("  [ENTRÉE]  = Accepter la pré-sélection")
    print("  'all'     = Tout sélectionner")
    print("  '1,3,5'   = Sélectionner manuellement")
    print("  'none'    = Rien sélectionner")
    
    response = input("\nTon choix : ").strip().lower()
    
    if response == '':
        # Accepter la pré-sélection
        return auto_selected
    elif response == 'all':
        return list(range(1, num_results + 1))
    elif response == 'none':
        return []
    else:
        try:
            return [int(x.strip()) for x in response.split(',')]
        except:
            print("❌ Format invalide, on garde la pré-sélection")
            return auto_selected


def save_progress(data, output_file):
    """Sauvegarde la progression"""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    print("\n" + "="*80)
    print("🔍 ANNOTATION SEMI-AUTOMATIQUE DU GROUND TRUTH")
    print("="*80)
    print("\nObjectif : Identifier les résultats pertinents pour chaque requête")
    print("Temps estimé : 1h30 (1-2 min par requête)\n")
    
    # Chemins
    queries_file = '../data/evaluation/queries.json'
    output_file = '../data/evaluation/queries_with_ground_truth.json'
    
    # Charger les requêtes
    print("📂 Chargement des requêtes...")
    data = load_queries(queries_file)
    queries = data['queries']
    
    # Charger le système de recherche
    print("🔧 Chargement du système de recherche...")
    engine = JobSearchEngine(data_dir='../data/processed', debug=False)
    print(f"✅ Système chargé : {len(engine.jobs_df)} offres indexées\n")
    
    # Vérifier si fichier de progression existe
    start_from = 0
    if os.path.exists(output_file):
        response = input(f"📁 Fichier {output_file} existe déjà. Reprendre ? (o/n) [o] : ").strip().lower()
        if response == '' or response == 'o':
            with open(output_file, 'r') as f:
                data = json.load(f)
            queries = data['queries']
            # Trouver la première requête non annotée
            for i, q in enumerate(queries):
                if not q['relevant_job_ids']:
                    start_from = i
                    break
            print(f"✅ Reprise à partir de la requête {start_from + 1}")
    
    # Annotation
    print("\n" + "="*80)
    print("DÉBUT DE L'ANNOTATION")
    print("="*80)
    
    for i in range(start_from, len(queries)):
        query_data = queries[i]
        query_text = query_data['query']
        category = query_data['category']
        
        # Recherche avec k=20 résultats (utilise seuils par défaut)
        search_result = engine.search(query_text, k=20, strict_threshold=10.0, generic_threshold=10.0)
        
        # Combiner résultats stricts et génériques
        results = search_result['results'] + search_result['generic_results']
        
        if len(results) == 0:
            print("\n" + "="*80)
            print(f"REQUÊTE {i+1}/{len(queries)}")
            print("="*80)
            print(f" '{query_text}' ({category})")
            print("\n AUCUN résultat trouvé")
            query_data['relevant_job_ids'] = []
            save_progress(data, output_file)
            print(f"\n💾 Progression sauvegardée ({i+1}/{len(queries)})")
            
            if i < len(queries) - 1:
                input("\n⏸️  Appuie sur ENTRÉE pour continuer...")
            continue
        
        # Extraire distances
        distances = [r['distance'] for r in results]
        
        # Pré-sélection automatique
        auto_selected = auto_select_relevant(category, results, distances)
        
        # Afficher
        display_results(query_data, results, distances, auto_selected)
        
        # Validation utilisateur
        selected_ranks = get_user_validation(auto_selected, len(results), category)
        
        # Convertir en job_ids
        selected_job_ids = [results[rank - 1]['job_id'] for rank in selected_ranks]
        query_data['relevant_job_ids'] = selected_job_ids
        
        print(f"\n✅ {len(selected_job_ids)} résultats pertinents sélectionnés")
        
        # Sauvegarder après chaque requête
        save_progress(data, output_file)
        print(f"💾 Progression sauvegardée ({i+1}/{len(queries)})")
        
        # Continue ou pause
        if i < len(queries) - 1:
            response = input("\n⏸️  Continuer ? (ENTRÉE=oui, 'q'=quitter) : ").strip().lower()
            if response == 'q':
                print("\n💾 Progression sauvegardée. Tu peux reprendre plus tard.")
                break
    
    # Résumé final
    print("\n" + "="*80)
    print("✅ ANNOTATION TERMINÉE")
    print("="*80)
    
    total_annotated = sum(1 for q in queries if q['relevant_job_ids'] or q['category'] == 'out_of_scope')
    print(f"\n📊 Statistiques :")
    print(f"   • Total requêtes : {len(queries)}")
    print(f"   • Annotées      : {total_annotated}")
    print(f"   • Restantes     : {len(queries) - total_annotated}")
    
    # Stats par catégorie
    print(f"\n📋 Par catégorie :")
    for cat in ['specific', 'moderate', 'generic', 'out_of_scope']:
        cat_queries = [q for q in queries if q['category'] == cat]
        total_relevant = sum(len(q['relevant_job_ids']) for q in cat_queries)
        avg_relevant = total_relevant / len(cat_queries) if cat_queries else 0
        print(f"   • {cat.upper():15s} : {len(cat_queries)} requêtes, "
              f"moy. {avg_relevant:.1f} résultats pertinents")
    
    print(f"\n💾 Fichier final : {output_file}")
    print(f"\n🎯 Prochaine étape : python evaluate_system.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption détectée. Progression sauvegardée.")
        print("Tu peux reprendre avec : python annotation_tool.py")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()