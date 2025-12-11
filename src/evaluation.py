#!/usr/bin/env python3
"""
Module d'évaluation pour le système de recherche
Implémente : Precision@K, Recall, F1-Score, MAP
"""

import numpy as np
import json
from typing import List, Dict, Set


class SearchEvaluator:
    """
    Classe pour évaluer un système de recherche d'information
    """
    
    def __init__(self, ground_truth_file: str):
        """
        Args:
            ground_truth_file: Chemin vers queries_with_ground_truth.json
        """
        with open(ground_truth_file, 'r') as f:
            self.data = json.load(f)
        
        self.queries = self.data['queries']
        print(f"✅ Chargé {len(self.queries)} requêtes avec ground truth")
    
    def precision_at_k(self, returned_ids: List[int], relevant_ids: Set[int], k: int = 10) -> float:
        """
        Calcule Precision@K
        
        Precision@K = Nombre de résultats pertinents dans top K / K
        """
        if len(returned_ids) == 0:
            return 0.0
        
        top_k = returned_ids[:k]
        relevant_in_top_k = sum(1 for job_id in top_k if job_id in relevant_ids)
        
        return relevant_in_top_k / min(k, len(top_k))
    
    def recall(self, returned_ids: List[int], relevant_ids: Set[int]) -> float:
        """
        Calcule le Recall
        
        Recall = Résultats pertinents retournés / Total pertinents existants
        """
        if len(relevant_ids) == 0:
            # Cas spécial : requête out-of-scope
            # Si aucun pertinent et aucun retourné → parfait (recall = 1.0)
            # Si aucun pertinent mais des retournés → faux positifs (recall = 0.0)
            return 1.0 if len(returned_ids) == 0 else 0.0
        
        relevant_returned = sum(1 for job_id in returned_ids if job_id in relevant_ids)
        
        return relevant_returned / len(relevant_ids)
    
    def f1_score(self, precision: float, recall: float) -> float:
        """
        Calcule le F1-Score
        
        F1 = 2 × (Precision × Recall) / (Precision + Recall)
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, returned_ids: List[int], relevant_ids: Set[int]) -> float:
        """
        Calcule Average Precision (AP)
        
        AP = (1/|relevant|) × Σ(Precision@k × rel(k))
        où rel(k) = 1 si le k-ème résultat est pertinent, 0 sinon
        """
        if len(relevant_ids) == 0:
            # Out-of-scope : si rien retourné → parfait, sinon mauvais
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
        """
        Évalue le système avec un seuil donné
        
        Args:
            search_engine: Instance de JobSearchEngine
            threshold: Seuil de distance à tester
            k: Nombre de résultats à considérer
        
        Returns:
            Dict avec métriques globales et par catégorie
        """
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
            
            # Chercher avec ce seuil (utilise strict_threshold comme seuil unique)
            search_result = search_engine.search(query, k=k, strict_threshold=threshold, generic_threshold=threshold)
            
            # Prendre tous les résultats (stricts + génériques puisqu'ils ont même seuil)
            all_results = search_result['results'] + search_result['generic_results']
            
            # Prendre les k meilleurs
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
        """
        Teste plusieurs seuils et trouve l'optimal
        
        Args:
            search_engine: Instance de JobSearchEngine
            thresholds: Liste de seuils à tester
            k: Nombre de résultats à considérer
        
        Returns:
            Dict avec tous les résultats et le seuil optimal
        """
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