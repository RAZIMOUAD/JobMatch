#!/usr/bin/env python3
"""
Script d'évaluation COMPLÈTE du système
Teste plusieurs seuils, choisit l'optimal, génère les graphiques

Usage: python evaluate_system.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.search import JobSearchEngine
from evaluation import SearchEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


def plot_threshold_analysis(results, output_dir='../data/evaluation'):
    """Génère les graphiques d'analyse des seuils"""
    
    # Extraire les données
    thresholds = [r['threshold'] for r in results]
    precisions = [r['global']['mean_precision'] for r in results]
    recalls = [r['global']['mean_recall'] for r in results]
    f1s = [r['global']['mean_f1'] for r in results]
    maps = [r['global']['map'] for r in results]
    
    # Trouver l'optimal
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    
    # Créer les graphiques
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precision vs Threshold
    axes[0, 0].plot(thresholds, precisions, marker='o', linewidth=2, markersize=6)
    axes[0, 0].axvline(best_threshold, color='r', linestyle='--', label=f'Optimal: {best_threshold:.2f}')
    axes[0, 0].set_xlabel('Threshold', fontsize=11)
    axes[0, 0].set_ylabel('Precision', fontsize=11)
    axes[0, 0].set_title('Precision vs Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall vs Threshold
    axes[0, 1].plot(thresholds, recalls, marker='o', linewidth=2, markersize=6, color='orange')
    axes[0, 1].axvline(best_threshold, color='r', linestyle='--', label=f'Optimal: {best_threshold:.2f}')
    axes[0, 1].set_xlabel('Threshold', fontsize=11)
    axes[0, 1].set_ylabel('Recall', fontsize=11)
    axes[0, 1].set_title('Recall vs Threshold', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1-Score vs Threshold
    axes[1, 0].plot(thresholds, f1s, marker='o', linewidth=2, markersize=6, color='green')
    axes[1, 0].axvline(best_threshold, color='r', linestyle='--', label=f'Optimal: {best_threshold:.2f}')
    axes[1, 0].set_xlabel('Threshold', fontsize=11)
    axes[1, 0].set_ylabel('F1-Score', fontsize=11)
    axes[1, 0].set_title('F1-Score vs Threshold (Métrique principale)', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    axes[1, 1].plot(recalls, precisions, marker='o', linewidth=2, markersize=6, color='purple')
    axes[1, 1].scatter(recalls[best_idx], precisions[best_idx], color='r', s=100, 
                      zorder=5, label=f'Optimal: {best_threshold:.2f}')
    axes[1, 1].set_xlabel('Recall', fontsize=11)
    axes[1, 1].set_ylabel('Precision', fontsize=11)
    axes[1, 1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'threshold_optimization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Graphiques sauvegardés : {output_path}")
    
    plt.close()


def plot_category_analysis(best_result, output_dir='../data/evaluation'):
    """Génère les graphiques par catégorie"""
    
    categories = list(best_result['by_category'].keys())
    precisions = [best_result['by_category'][cat]['mean_precision'] for cat in categories]
    recalls = [best_result['by_category'][cat]['mean_recall'] for cat in categories]
    f1s = [best_result['by_category'][cat]['mean_f1'] for cat in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='orange')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='green')
    
    ax.set_xlabel('Catégorie de requêtes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Performance par Catégorie (Threshold={best_result["threshold"]:.2f})', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.upper() for cat in categories])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs au-dessus des barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'performance_by_category.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Graphiques par catégorie : {output_path}")
    
    plt.close()


def save_evaluation_report(optimization_result, output_dir='../data/evaluation'):
    """Sauvegarde le rapport d'évaluation complet"""
    
    best_result = optimization_result['best_result']
    
    report = {
        'optimal_threshold': float(optimization_result['best_threshold']),
        'global_metrics': {
            'precision': float(best_result['global']['mean_precision']),
            'recall': float(best_result['global']['mean_recall']),
            'f1_score': float(best_result['global']['mean_f1']),
            'map': float(best_result['global']['map'])
        },
        'by_category': {
            cat: {
                'precision': float(metrics['mean_precision']),
                'recall': float(metrics['mean_recall']),
                'f1_score': float(metrics['mean_f1']),
                'map': float(metrics['map']),
                'num_queries': int(metrics['num_queries'])
            }
            for cat, metrics in best_result['by_category'].items()
        },
        'all_thresholds_tested': [
            {
                'threshold': float(r['threshold']),
                'precision': float(r['global']['mean_precision']),
                'recall': float(r['global']['mean_recall']),
                'f1_score': float(r['global']['mean_f1']),
                'map': float(r['global']['map'])
            }
            for r in optimization_result['all_results']
        ]
    }
    
    output_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Rapport d'évaluation : {output_path}")


def print_detailed_analysis(best_result):
    """Affiche une analyse détaillée dans la console"""
    
    print("\n" + "="*80)
    print("📊 ANALYSE DÉTAILLÉE PAR CATÉGORIE")
    print("="*80)
    
    for category, metrics in best_result['by_category'].items():
        print(f"\n📋 {category.upper()}")
        print("-"*80)
        print(f"   Nombre de requêtes : {metrics['num_queries']}")
        print(f"   Precision          : {metrics['mean_precision']:.3f}")
        print(f"   Recall             : {metrics['mean_recall']:.3f}")
        print(f"   F1-Score           : {metrics['mean_f1']:.3f}")
        print(f"   MAP                : {metrics['map']:.3f}")
    
    # Exemples de requêtes
    print("\n" + "="*80)
    print("📋 EXEMPLES DE REQUÊTES PAR CATÉGORIE")
    print("="*80)
    
    for category, queries_metrics in best_result['details'].items():
        print(f"\n🏷️  {category.upper()} :")
        
        # Montrer les 3 meilleures et 3 pires
        sorted_queries = sorted(queries_metrics, key=lambda x: x['f1'], reverse=True)
        
        print("\n   ✅ Meilleures performances :")
        for q in sorted_queries[:3]:
            print(f"      • '{q['query']}' : F1={q['f1']:.3f}, "
                  f"P={q['precision']:.3f}, R={q['recall']:.3f}")
        
        if len(sorted_queries) > 3:
            print("\n   ⚠️  Performances faibles :")
            for q in sorted_queries[-3:]:
                print(f"      • '{q['query']}' : F1={q['f1']:.3f}, "
                      f"P={q['precision']:.3f}, R={q['recall']:.3f}")


def main():
    print("\n" + "="*80)
    print("🔬 ÉVALUATION COMPLÈTE DU SYSTÈME JOBMATCH")
    print("="*80)
    print("\nObjectif : Trouver le seuil optimal et générer les métriques\n")
    
    # Chemins
    ground_truth_file = '../data/evaluation/queries_with_ground_truth.json'
    
    # Vérifier que le ground truth existe
    if not os.path.exists(ground_truth_file):
        print("❌ ERREUR : Fichier ground truth non trouvé !")
        print(f"   Attendu : {ground_truth_file}")
        print("\n🔧 Tu dois d'abord annoter avec : python annotation_tool.py")
        return
    
    # Charger le système
    print("🔧 Chargement du système de recherche...")
    engine = JobSearchEngine(data_dir='../data/processed', debug=False)
    print(f"✅ Système chargé : {len(engine.jobs_df)} offres indexées")
    
    # Charger l'évaluateur
    print("\n📂 Chargement du ground truth...")
    evaluator = SearchEvaluator(ground_truth_file)
    
    # Définir les seuils à tester
    thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    # Optimisation
    optimization_result = evaluator.find_optimal_threshold(engine, thresholds, k=10)
    
    # Analyse détaillée
    print_detailed_analysis(optimization_result['best_result'])
    
    # Générer les graphiques
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DES GRAPHIQUES")
    print("="*80)
    
    plot_threshold_analysis(optimization_result['all_results'])
    plot_category_analysis(optimization_result['best_result'])
    
    # Sauvegarder le rapport
    print("\n" + "="*80)
    print("💾 SAUVEGARDE DU RAPPORT")
    print("="*80)
    
    save_evaluation_report(optimization_result)
    
    # Résumé final
    print("\n" + "="*80)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*80)
    
    best_threshold = optimization_result['best_threshold']
    best_metrics = optimization_result['best_result']['global']
    
    print(f"""
📊 RÉSULTATS FINAUX :

   🎯 Seuil optimal : {best_threshold:.2f}
   
   📈 Métriques globales :
      • Precision : {best_metrics['mean_precision']:.3f}
      • Recall    : {best_metrics['mean_recall']:.3f}
      • F1-Score  : {best_metrics['mean_f1']:.3f}
      • MAP       : {best_metrics['map']:.3f}

📁 Fichiers générés :
   • threshold_optimization.png
   • performance_by_category.png
   • evaluation_report.json

🎯 Prochaine étape :
   Mettre à jour Flask avec le seuil optimal : {best_threshold:.2f}
   Fichier à modifier : app/app.py
   Ligne à changer : distance_threshold={best_threshold:.2f}
    """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()