#!/usr/bin/env python3
"""
STEP 2 : CROSS-ENCODER RERANKING

Principe :
1. Stage 1 (Hybrid) : Récupère 100 candidats (cast wide net)
2. Stage 2 (Cross-Encoder) : Rerank précisément les 100 candidats

Résultat : Les VRAIS matchs montent en top-10
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Dict
import time

# Importer le HybridRetriever du Step 1
try:
    from step1_hybrid_retrieval import HybridRetriever
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from step1_hybrid_retrieval import HybridRetriever


class TwoStageRetriever:
    """
    Système de recherche en 2 étapes :
    - Stage 1 : HybridRetriever (rapide, large)
    - Stage 2 : CrossEncoder (lent, précis)
    """
    
    def __init__(self, data_dir='../data/processed'):
        print("\n🔧 Initialisation du TwoStageRetriever...")
        
        # Stage 1 : Hybrid Retriever
        print("\n📊 STAGE 1 : Hybrid Retriever")
        self.hybrid_retriever = HybridRetriever(data_dir=data_dir)
        
        # Stage 2 : Cross-Encoder
        print("\n🧠 STAGE 2 : Cross-Encoder")
        print("   ⏳ Chargement du modèle (peut prendre 10-20 secondes)...")
        
        # Modèle pré-entraîné pour ranking
        # ms-marco-MiniLM = entraîné sur Microsoft MARCO (passage ranking)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        print("   ✅ Cross-Encoder chargé")
        print("\n✅ TwoStageRetriever prêt\n")
    
    def search(self, 
               query: str, 
               k: int = 10,
               alpha: float = 0.5,
               candidates_k: int = 100) -> List[Dict]:
        """
        Recherche en 2 étapes
        
        Args:
            query: Requête utilisateur
            k: Nombre de résultats finaux
            alpha: Poids SBERT dans hybrid (0.5 = équilibre)
            candidates_k: Nombre de candidats pour Stage 1
        
        Returns:
            Top-K résultats après reranking
        """
        
        start_time = time.time()
        
        # ====================================================================
        # STAGE 1 : CANDIDATE RETRIEVAL (Hybrid)
        # ====================================================================
        
        stage1_start = time.time()
        
        candidates = self.hybrid_retriever.search(
            query=query,
            k=candidates_k,
            alpha=alpha
        )
        
        stage1_time = time.time() - stage1_start
        
        if not candidates:
            return []
        
        # ====================================================================
        # STAGE 2 : CROSS-ENCODER RERANKING
        # ====================================================================
        
        stage2_start = time.time()
        
        # Créer les paires (query, job_title)
        query_job_pairs = [
            [query, candidate['title']] 
            for candidate in candidates
        ]
        
        # Prédire les scores de pertinence
        # Score entre -10 et +10 (plus haut = plus pertinent)
        ce_scores = self.cross_encoder.predict(query_job_pairs)
        
        stage2_time = time.time() - stage2_start
        
        # ====================================================================
        # FUSION : Combiner scores Hybrid + Cross-Encoder
        # ====================================================================
        
        # Normaliser scores Cross-Encoder (0-1)
        ce_scores_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-6)
        
        # Ajouter les scores à chaque candidat
        for i, candidate in enumerate(candidates):
            candidate['ce_score'] = float(ce_scores[i])
            candidate['ce_score_norm'] = float(ce_scores_norm[i])
            
            # Score final = 70% Cross-Encoder + 30% Hybrid
            # Cross-Encoder est plus fiable, donc poids plus élevé
            candidate['final_score'] = (
                0.7 * ce_scores_norm[i] + 
                0.3 * candidate['score']
            )
        
        # Trier par score final
        candidates_sorted = sorted(
            candidates, 
            key=lambda x: x['final_score'], 
            reverse=True
        )
        
        # Top-K finaux
        results = candidates_sorted[:k]
        
        total_time = time.time() - start_time
        
        # Ajouter timing info au premier résultat
        if results:
            results[0]['_timing'] = {
                'stage1_ms': round(stage1_time * 1000, 1),
                'stage2_ms': round(stage2_time * 1000, 1),
                'total_ms': round(total_time * 1000, 1)
            }
        
        return results


def test_two_stage():
    """
    Test du système 2-stage sur les requêtes problématiques
    """
    print("="*80)
    print("🧪 TEST DU TWO-STAGE RETRIEVER")
    print("="*80)
    
    # Initialiser
    retriever = TwoStageRetriever(data_dir='../data/processed')
    
    # Requêtes problématiques
    test_queries = [
        ("cna telemetry", "Problématique : 0/2 trouvés avec SBERT seul"),
        ("physician emergency medicine", "Problématique : 0/2 trouvés"),
        ("hospitalist physician", "Problématique : 1/3 trouvés"),
        ("registered nurse emergency", "Fonctionne déjà bien"),
        ("software engineer", "OUT_OF_SCOPE : ne devrait rien retourner")
    ]
    
    for query, description in test_queries:
        print("\n" + "="*80)
        print(f"📋 Requête : '{query}'")
        print(f"   {description}")
        print("-"*80)
        
        # Recherche
        results = retriever.search(query, k=10, alpha=0.5, candidates_k=100)
        
        if not results:
            print("   ❌ Aucun résultat")
            continue
        
        # Afficher timing
        if '_timing' in results[0]:
            timing = results[0]['_timing']
            print(f"\n⏱️  Timing : Stage1={timing['stage1_ms']}ms | "
                  f"Stage2={timing['stage2_ms']}ms | "
                  f"Total={timing['total_ms']}ms")
        
        print(f"\n✅ {len(results)} résultats trouvés")
        
        # Afficher top-10
        for i, result in enumerate(results, 1):
            # Vérifier pertinence (basique)
            title_lower = result['title'].lower()
            query_words = query.lower().split()
            
            # Compter combien de mots de la requête sont dans le titre
            matches = sum(1 for word in query_words if word in title_lower)
            is_relevant = matches >= len(query_words) * 0.6  # 60% des mots
            
            marker = "✅" if is_relevant else "⚠️"
            
            print(f"\n{marker} #{i:2d} [Final: {result['final_score']:.3f}] {result['title'][:65]}")
            print(f"      Hybrid: {result['score']:.3f} | "
                  f"CE: {result['ce_score']:.2f} | "
                  f"SBERT: {result['sbert_score']:.3f} | "
                  f"BM25: {result['bm25_score']:.3f}")


def compare_stages():
    """
    Comparaison directe : Hybrid seul vs Two-Stage
    """
    print("\n" + "="*80)
    print("📊 COMPARAISON : Hybrid seul vs Two-Stage")
    print("="*80)
    
    retriever = TwoStageRetriever(data_dir='../data/processed')
    
    # Requête critique
    query = "cna telemetry"
    
    print(f"\n📋 Requête : '{query}'")
    print("   Résultat attendu : 2 jobs 'CNA Telemetry' dans le dataset\n")
    
    # ====================================================================
    # Hybrid seul (Stage 1)
    # ====================================================================
    
    print("🔵 HYBRID SEUL (Stage 1 uniquement) :")
    print("-"*80)
    
    hybrid_results = retriever.hybrid_retriever.search(query, k=10, alpha=0.5)
    
    for i, result in enumerate(hybrid_results, 1):
        title_lower = result['title'].lower()
        is_relevant = 'cna' in title_lower and 'telemetry' in title_lower
        marker = "✅" if is_relevant else "❌"
        
        print(f"{marker} #{i:2d} [{result['score']:.3f}] {result['title'][:70]}")
        print(f"      SBERT: {result['sbert_score']:.3f} | BM25: {result['bm25_score']:.3f}")
    
    # ====================================================================
    # Two-Stage (Hybrid + Cross-Encoder)
    # ====================================================================
    
    print("\n🟢 TWO-STAGE (Hybrid + Cross-Encoder) :")
    print("-"*80)
    
    twostage_results = retriever.search(query, k=10, alpha=0.5, candidates_k=100)
    
    for i, result in enumerate(twostage_results, 1):
        title_lower = result['title'].lower()
        is_relevant = 'cna' in title_lower and 'telemetry' in title_lower
        marker = "✅" if is_relevant else "❌"
        
        print(f"{marker} #{i:2d} [Final: {result['final_score']:.3f}] {result['title'][:70]}")
        print(f"      Hybrid: {result['score']:.3f} | CE: {result['ce_score']:.2f}")
    
    # ====================================================================
    # Analyse
    # ====================================================================
    
    print("\n" + "="*80)
    print("📊 ANALYSE")
    print("="*80)
    
    # Compter pertinents
    hybrid_relevant = sum(
        1 for r in hybrid_results 
        if 'cna' in r['title'].lower() and 'telemetry' in r['title'].lower()
    )
    
    twostage_relevant = sum(
        1 for r in twostage_results 
        if 'cna' in r['title'].lower() and 'telemetry' in r['title'].lower()
    )
    
    print(f"\n📊 Pertinents dans top-10 :")
    print(f"   Hybrid seul   : {hybrid_relevant}/2 ({hybrid_relevant*50}%)")
    print(f"   Two-Stage     : {twostage_relevant}/2 ({twostage_relevant*50}%)")
    
    if twostage_relevant > hybrid_relevant:
        print(f"\n Amélioration : +{twostage_relevant - hybrid_relevant} résultat(s) pertinent(s)")
    elif twostage_relevant == hybrid_relevant:
        print(f"\n Même nombre de pertinents, mais ranking amélioré")
    else:
        print(f"\n Régression (inattendu)")


def analyze_cross_encoder_scores():
    """
    Analyser les scores Cross-Encoder en détail
    """
    print("\n" + "="*80)
    print("🔬 ANALYSE DÉTAILLÉE : Cross-Encoder Scores")
    print("="*80)
    
    retriever = TwoStageRetriever(data_dir='../data/processed')
    
    query = "cna telemetry"
    
    print(f"\n📋 Requête : '{query}'")
    print("\nRécupération de 20 candidats pour analyse...\n")
    
    # Récupérer plus de candidats pour analyse
    results = retriever.search(query, k=20, alpha=0.5, candidates_k=100)
    
    print("Score Cross-Encoder (CE) : -10 à +10 (plus haut = plus pertinent)")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        title_lower = result['title'].lower()
        has_cna = 'cna' in title_lower or 'nursing assistant' in title_lower
        has_tele = 'telemetry' in title_lower or 'tele' in title_lower
        
        if has_cna and has_tele:
            marker = "✅✅"
        elif has_cna or has_tele:
            marker = "⚠️ "
        else:
            marker = "❌"
        
        print(f"\n{marker} #{i:2d} {result['title'][:60]}")
        print(f"      CE Score: {result['ce_score']:6.2f} | "
              f"Hybrid: {result['score']:.3f} | "
              f"Final: {result['final_score']:.3f}")


if __name__ == "__main__":
    # Test 1 : Fonctionnement de base
    test_two_stage()
    
    # Test 2 : Comparaison directe
    compare_stages()
    
    # Test 3 : Analyse des scores
    analyze_cross_encoder_scores()
    
    print("\n" + "="*80)
    print("✅ STEP 2 TERMINÉ")
    print("="*80)
    print("\n📊 Résultats attendus :")
    print("   • CNA Telemetry : 2/2 dans top-5 ✅✅")
    print("   • Scores CE très élevés pour vrais matchs (> 5.0)")
    print("   • Scores CE bas pour faux positifs (< 2.0)")
    print("\n🎯 Prochaine étape : STEP 3 - Intelligent Filtering\n")