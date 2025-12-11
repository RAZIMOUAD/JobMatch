#!/usr/bin/env python3
"""
Script d'évaluation COMPLET et UNIFIÉ du système JobMatch
Contient SearchEvaluator + évaluation + génération graphiques

Usage: python evaluate_complete.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.search import JobSearchEngine
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Set


# ============================================================================
# PARTIE 1 : SearchEvaluator (métriques IR)
# ============================================================================

class SearchEvaluator:
    """Classe pour évaluer un système de recherche d'information"""
    
    def __init__(self, ground_truth_file: str):
        with open(ground_truth_file, 'r') as f:
            self.data = json.load(f)
        
        self.queries = self.data['queries']
        print(f"✅ Chargé {len(self.queries)} requêtes avec ground truth")
    
    def precision_at_k(self, returned_ids: List[int], relevant_ids: Set[int], k: int = 10) -> float:
        """Calcule Precision@K"""
        if len(returned_ids) == 0:
            return 0.0
        
        top_k = returned_ids[:k]
        relevant_in_top_k = sum(1 for job_id in top_k if job_id in relevant_ids)
        
        return relevant_in_top_k / min(k, len(top_k))
    
    def recall(self, returned_ids: List[int], relevant_ids: Set[int]) -> float:
        """Calcule le Recall"""
        if len(relevant_ids) == 0:
            return 1.0 if len(returned_ids) == 0 else 0.0
        
        relevant_returned = sum(1 for job_id in returned_ids if job_id in relevant_ids)
        
        return relevant_returned / len(relevant_ids)
    
    def f1_score(self, precision: float, recall: float) -> float:
        """Calcule le F1-Score"""
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, returned_ids: List[int], relevant_ids: Set[int]) -> float:
        """Calcule Average Precision (AP)"""
        if len(relevant_ids) == 0:
            return 1.0 if len(returned_ids) == 0 else 0.0
        
        if len(returned_ids) == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for k, job_id in enumerate(returned_ids, 1):
            if job_id in relevant_ids:
                relevant_count += 1
                precision_at_this_k = relevant_count / k
                precision_sum += precision_at_this_k
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / len(relevant_ids)
    
    def evaluate_threshold(self, search_engine, threshold: float, k: int = 10) -> Dict:
        """Évalue le système avec un seuil donné"""
        
        metrics_by_category = {
            'specific': [],
            'moderate': [],
            'generic': [],
            'out_of_scope': []
        }
        
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_aps = []
        
        for query_data in self.queries:
            query = query_data['query']
            category = query_data['category']
            relevant_ids = set(query_data['relevant_job_ids'])
            
            # Chercher avec ce seuil
            search_result = search_engine.search(query, k=k, strict_threshold=threshold, generic_threshold=threshold)
            
            # Combiner résultats stricts + génériques
            all_results = search_result['results'] + search_result['generic_results']
            results = all_results[:k]
            returned_ids = [r['job_id'] for r in results]
            
            # Calculer métriques
            precision = self.precision_at_k(returned_ids, relevant_ids, k=k)
            recall = self.recall(returned_ids, relevant_ids)
            f1 = self.f1_score(precision, recall)
            ap = self.average_precision(returned_ids, relevant_ids)
            
            # Stocker
            metrics_by_category[category].append({
                'query': query,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ap': ap,
                'num_returned': len(returned_ids),
                'num_relevant': len(relevant_ids),
                'num_correct': sum(1 for job_id in returned_ids if job_id in relevant_ids)
            })
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            all_aps.append(ap)
        
        # Calcul moyennes globales
        return {
            'threshold': threshold,
            'global': {
                'mean_precision': np.mean(all_precisions),
                'mean_recall': np.mean(all_recalls),
                'mean_f1': np.mean(all_f1s),
                'map': np.mean(all_aps)
            },
            'by_category': {
                cat: {
                    'mean_precision': np.mean([m['precision'] for m in metrics]),
                    'mean_recall': np.mean([m['recall'] for m in metrics]),
                    'mean_f1': np.mean([m['f1'] for m in metrics]),
                    'map': np.mean([m['ap'] for m in metrics]),
                    'num_queries': len(metrics)
                }
                for cat, metrics in metrics_by_category.items() if metrics
            },
            'details': metrics_by_category
        }
    
    def find_optimal_threshold(self, search_engine, thresholds: List[float], k: int = 10) -> Dict:
        """Teste plusieurs seuils et trouve l'optimal"""
        
        results = []
        
        print("\n" + "="*80)
        print("🔬 OPTIMISATION DU SEUIL DE DISTANCE")
        print("="*80)
        print(f"\n⏳ Test de {len(thresholds)} seuils...\n")
        
        for threshold in thresholds:
            print(f"  Testing threshold {threshold:.2f}...", end=" ")
            result = self.evaluate_threshold(search_engine, threshold, k=k)
            results.append(result)
            print(f"F1={result['global']['mean_f1']:.3f}, MAP={result['global']['map']:.3f}")
        
        # Trouver le meilleur F1
        best_idx = max(range(len(results)), key=lambda i: results[i]['global']['mean_f1'])
        best_result = results[best_idx]
        
        print("\n" + "="*80)
        print("🎯 SEUIL OPTIMAL TROUVÉ")
        print("="*80)
        print(f"\n   Threshold : {best_result['threshold']:.2f}")
        print(f"   Precision : {best_result['global']['mean_precision']:.3f}")
        print(f"   Recall    : {best_result['global']['mean_recall']:.3f}")
        print(f"   F1-Score  : {best_result['global']['mean_f1']:.3f}")
        print(f"   MAP       : {best_result['global']['map']:.3f}")
        
        return {
            'all_results': results,
            'best_result': best_result,
            'best_threshold': best_result['threshold']
        }


# ============================================================================
# PARTIE 2 : Génération des graphiques
# ============================================================================

def plot_threshold_analysis(results, output_dir='../data/evaluation'):
    """Génère les graphiques d'analyse des seuils"""
    
    thresholds = [r['threshold'] for r in results]
    precisions = [r['global']['mean_precision'] for r in results]
    recalls = [r['global']['mean_recall'] for r in results]
    f1s = [r['global']['mean_f1'] for r in results]
    maps = [r['global']['map'] for r in results]
    
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    
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


# ============================================================================
# PARTIE 3 : Main
# ============================================================================

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
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption détectée")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()