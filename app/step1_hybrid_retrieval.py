#!/usr/bin/env python3
"""
STEP 1 : HYBRID RETRIEVAL (SBERT + BM25)

Combine deux approches :
- Dense Retrieval (SBERT) : Comprend la sémantique
- Sparse Retrieval (BM25) : Vérifie la présence des mots-clés

Résultat : Meilleure Precision ET Recall
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
from typing import List, Dict, Tuple
import re


class HybridRetriever:
    """
    Retriever hybride combinant SBERT (sémantique) et BM25 (lexical)
    """
    
    def __init__(self, data_dir='../data/processed'):
        print("\n🔧 Initialisation du HybridRetriever...")
        
        # Charger les données
        print("📂 Chargement des métadonnées...")
        self.jobs_df = pd.read_csv(f'{data_dir}/jobs_metadata.csv')
        print(f"   ✅ {len(self.jobs_df)} jobs chargés")
        
        # Charger SBERT
        print("🧠 Chargement du modèle SBERT...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✅ SBERT chargé")
        
        # Charger FAISS
        print("📊 Chargement de l'index FAISS...")
        self.faiss_index = faiss.read_index(f'{data_dir}/faiss_index.bin')
        print(f"   ✅ FAISS chargé ({self.faiss_index.ntotal} vecteurs)")
        
        # Préparer BM25
        print("📝 Préparation de l'index BM25...")
        self._prepare_bm25()
        print("   ✅ BM25 prêt")
        
        print("\n✅ HybridRetriever initialisé\n")
    
    def _prepare_bm25(self):
        """
        Prépare l'index BM25 sur les titres de jobs
        """
        # Tokenizer simple mais efficace
        tokenized_titles = []
        for title in self.jobs_df['title']:
            # Lowercase + split sur espaces/ponctuation
            tokens = re.findall(r'\b\w+\b', title.lower())
            tokenized_titles.append(tokens)
        
        # Créer l'index BM25
        self.bm25 = BM25Okapi(tokenized_titles)
        self.tokenized_titles = tokenized_titles
    
    def _search_sbert(self, query: str, k: int = 100) -> Tuple[List[int], List[float]]:
        """
        Recherche SBERT (sémantique)
        
        Returns:
            indices: Indices des jobs trouvés
            distances: Distances euclidiennes
        """
        query_embedding = self.sbert_model.encode([query]).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        return indices[0].tolist(), distances[0].tolist()
    
    def _search_bm25(self, query: str, k: int = 100) -> Tuple[List[int], List[float]]:
        """
        Recherche BM25 (lexicale)
        
        Returns:
            indices: Indices des jobs trouvés
            scores: Scores BM25 (plus haut = meilleur)
        """
        # Tokenizer la requête
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Calculer les scores BM25
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Top-K indices
        top_k_indices = np.argsort(bm25_scores)[::-1][:k]
        top_k_scores = bm25_scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()
    
    def search(self, 
               query: str, 
               k: int = 10,
               alpha: float = 0.5,
               retrieve_k: int = 100) -> List[Dict]:
        """
        Recherche hybride SBERT + BM25
        
        Args:
            query: Requête utilisateur
            k: Nombre de résultats finaux
            alpha: Poids SBERT (0.5 = équilibre, 0.7 = plus sémantique, 0.3 = plus lexical)
            retrieve_k: Nombre de candidats à récupérer de chaque méthode
        
        Returns:
            Liste de résultats triés par score hybride
        """
        
        # 1. SBERT Search
        sbert_indices, sbert_distances = self._search_sbert(query, k=retrieve_k)
        
        # Convertir distances en scores (0-1, plus haut = meilleur)
        # Distance L2 → Similarity score
        sbert_scores = 1 / (1 + np.array(sbert_distances))
        
        # 2. BM25 Search
        bm25_indices, bm25_scores_raw = self._search_bm25(query, k=retrieve_k)
        
        # Normaliser scores BM25 (0-1)
        bm25_scores = np.array(bm25_scores_raw)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # 3. Fusion des scores
        # Créer un dictionnaire : job_index → scores
        job_scores = {}
        
        # Ajouter scores SBERT
        for idx, score in zip(sbert_indices, sbert_scores):
            if idx not in job_scores:
                job_scores[idx] = {'sbert': 0, 'bm25': 0}
            job_scores[idx]['sbert'] = float(score)
        
        # Ajouter scores BM25
        for idx, score in zip(bm25_indices, bm25_scores):
            if idx not in job_scores:
                job_scores[idx] = {'sbert': 0, 'bm25': 0}
            job_scores[idx]['bm25'] = float(score)
        
        # Calculer score hybride
        for idx in job_scores:
            job_scores[idx]['hybrid'] = (
                alpha * job_scores[idx]['sbert'] + 
                (1 - alpha) * job_scores[idx]['bm25']
            )
        
        # 4. Trier par score hybride
        sorted_jobs = sorted(
            job_scores.items(), 
            key=lambda x: x[1]['hybrid'], 
            reverse=True
        )
        
        # 5. Créer les résultats finaux
        results = []
        for job_idx, scores in sorted_jobs[:k]:
            job = self.jobs_df.iloc[job_idx]
            
            # Seuil minimal pour filtrer le bruit
            if scores['hybrid'] < 0.3:
                continue
            
            results.append({
                'job_id': int(job['job_id']),
                'title': job['title'],
                'description': job['description'][:200] + '...' if len(job['description']) > 200 else job['description'],
                'location': job['location'],
                'experience_level': job['formatted_experience_level'],
                'score': float(scores['hybrid']),
                'sbert_score': float(scores['sbert']),
                'bm25_score': float(scores['bm25']),
                'retrieval_method': self._get_method_label(scores)
            })
        
        return results
    
    def _get_method_label(self, scores: Dict) -> str:
        """
        Détermine quelle méthode a le plus contribué
        """
        if scores['sbert'] > 0.7 and scores['bm25'] > 0.7:
            return 'both_strong'
        elif scores['sbert'] > scores['bm25'] + 0.2:
            return 'sbert_dominant'
        elif scores['bm25'] > scores['sbert'] + 0.2:
            return 'bm25_dominant'
        else:
            return 'balanced'


def test_hybrid_retriever():
    """
    Test du HybridRetriever sur quelques requêtes
    """
    print("="*80)
    print("🧪 TEST DU HYBRID RETRIEVER")
    print("="*80)
    
    # Initialiser
    retriever = HybridRetriever(data_dir='../data/processed')
    
    # Test queries
    test_queries = [
        ("registered nurse emergency", "SPECIFIC (métier + spécialisation)"),
        ("cna telemetry", "SPECIFIC (problématique actuellement)"),
        ("physical therapist", "MODERATE (métier seul)"),
        ("healthcare job", "GENERIC (très vague)"),
        ("software engineer", "OUT_OF_SCOPE (non-Healthcare)")
    ]
    
    for query, description in test_queries:
        print("\n" + "="*80)
        print(f"📋 Requête : '{query}'")
        print(f"   Type : {description}")
        print("-"*80)
        
        # Recherche avec différents alpha
        for alpha in [0.3, 0.5, 0.7]:
            print(f"\n🔍 Alpha = {alpha} ({'plus BM25' if alpha < 0.5 else 'équilibré' if alpha == 0.5 else 'plus SBERT'})")
            
            results = retriever.search(query, k=5, alpha=alpha)
            
            if not results:
                print("   ❌ Aucun résultat")
                continue
            
            print(f"   ✅ {len(results)} résultats trouvés")
            
            for i, result in enumerate(results, 1):
                print(f"\n   #{i} [{result['score']:.3f}] {result['title'][:60]}")
                print(f"      SBERT: {result['sbert_score']:.3f} | BM25: {result['bm25_score']:.3f} | Method: {result['retrieval_method']}")


def compare_with_sbert_only():
    """
    Comparaison Hybrid vs SBERT seul
    """
    print("\n" + "="*80)
    print("📊 COMPARAISON : Hybrid vs SBERT seul")
    print("="*80)
    
    retriever = HybridRetriever(data_dir='../data/processed')
    
    # Requête problématique
    query = "cna telemetry"
    
    print(f"\n📋 Requête : '{query}'")
    print("   (Actuellement : 0/2 pertinents trouvés avec SBERT seul)\n")
    
    # SBERT seul (alpha = 1.0)
    print("🔵 SBERT SEUL (alpha=1.0) :")
    print("-"*80)
    results_sbert = retriever.search(query, k=10, alpha=1.0)
    
    for i, result in enumerate(results_sbert, 1):
        # Vérifier si pertinent
        title_lower = result['title'].lower()
        is_relevant = 'cna' in title_lower and 'telemetry' in title_lower
        marker = "✅" if is_relevant else "❌"
        
        print(f"{marker} #{i:2d} [{result['score']:.3f}] {result['title'][:70]}")
    
    # Hybrid (alpha = 0.5)
    print("\n🟢 HYBRID (alpha=0.5) :")
    print("-"*80)
    results_hybrid = retriever.search(query, k=10, alpha=0.5)
    
    for i, result in enumerate(results_hybrid, 1):
        # Vérifier si pertinent
        title_lower = result['title'].lower()
        is_relevant = 'cna' in title_lower and 'telemetry' in title_lower
        marker = "✅" if is_relevant else "❌"
        
        print(f"{marker} #{i:2d} [{result['score']:.3f}] {result['title'][:70]}")
        print(f"     SBERT: {result['sbert_score']:.3f} | BM25: {result['bm25_score']:.3f}")


if __name__ == "__main__":
    # Test 1 : Fonctionnement de base
    test_hybrid_retriever()
    
    # Test 2 : Comparaison directe
    compare_with_sbert_only()
    
    print("\n" + "="*80)
    print("✅ STEP 1 TERMINÉ")
    print("="*80)
    print("\n📊 Résultats attendus :")
    print("   • CNA Telemetry : Devrait être dans le top-3 avec Hybrid")
    print("   • Score BM25 élevé pour les matchs exacts")
    print("   • Score SBERT élevé pour les variantes sémantiques")
    print("\n🎯 Prochaine étape : STEP 2 - Cross-Encoder Reranking\n")