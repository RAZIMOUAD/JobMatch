"""
Fonction de recherche sémantique pour JobMatch - VERSION PRODUCTION FINALE
Avec seuil double : strict (0.85) et générique (1.0)
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class JobSearchEngine:
    """
    Moteur de recherche sémantique pour offres d'emploi Healthcare
    """
    
    def __init__(self, data_dir='../data/processed', debug=False):
        """
        Initialiser le moteur de recherche
        
        Args:
            data_dir (str): Chemin vers le dossier contenant les données
            debug (bool): Activer les messages de debug (défaut: False)
        """
        self.data_dir = data_dir
        self.model = None
        self.index = None
        self.jobs_df = None
        self.debug = debug
        
        # Charger automatiquement
        self._load_model()
        self._load_index()
        self._load_metadata()
        
        print(f"✅ JobSearchEngine initialisé avec {len(self.jobs_df)} offres")
    
    def _load_model(self):
        """Charger le modèle SBERT"""
        print("⏳ Chargement du modèle SBERT...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Modèle SBERT chargé")
    
    def _load_index(self):
        """Charger l'index FAISS"""
        index_path = os.path.join(self.data_dir, 'faiss_index.bin')
        print(f"⏳ Chargement de l'index FAISS depuis {index_path}...")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")
        
        self.index = faiss.read_index(index_path)
        print(f"✅ Index FAISS chargé ({self.index.ntotal} vecteurs)")
    
    def _load_metadata(self):
        """Charger les métadonnées des jobs"""
        metadata_path = os.path.join(self.data_dir, 'jobs_metadata.csv')
        print(f"⏳ Chargement des métadonnées depuis {metadata_path}...")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Métadonnées introuvables : {metadata_path}")
        
        self.jobs_df = pd.read_csv(metadata_path)
        print(f"✅ Métadonnées chargées ({len(self.jobs_df)} jobs)")
    
    def search(self, query, k=10, strict_threshold=0.85, generic_threshold=1.0):
        """
        Rechercher des jobs similaires à la requête
        AVEC SEUIL DOUBLE : strict (0.85) et générique (1.0)
        
        Args:
            query (str): Requête en langage naturel
            k (int): Nombre de résultats à retourner (défaut: 10)
            strict_threshold (float): Seuil pour résultats précis (défaut: 0.85)
            generic_threshold (float): Seuil pour résultats génériques (défaut: 1.0)
        
        Returns:
            dict: {
                'results': [...],           # Résultats stricts (distance ≤ 0.85)
                'generic_results': [...],   # Résultats génériques (0.85 < distance ≤ 1.0)
                'query_type': 'specific' / 'generic' / 'mixed' / 'none'
                'strict_count': int,        # Nombre de résultats stricts
                'generic_count': int        # Nombre de résultats génériques
            }
        """
        
        # Validation
        if not query or not query.strip():
            return {
                'results': [],
                'generic_results': [],
                'query_type': 'empty',
                'strict_count': 0,
                'generic_count': 0
            }
        
        if self.debug:
            print(f"\n🔍 Recherche : '{query}'")
            print(f"   Seuil strict    : {strict_threshold}")
            print(f"   Seuil générique : {generic_threshold}")
        
        # Encoder la requête
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Rechercher dans l'index (chercher 3× plus pour avoir des options)
        search_k = min(k * 3, self.index.ntotal)  # Ne pas dépasser le nombre total
        distances, indices = self.index.search(query_embedding, search_k)
        
        if self.debug:
            print(f"   Distance min : {distances[0].min():.4f}")
            print(f"   Distance max : {distances[0].max():.4f}")
            print(f"   Distance moy : {distances[0].mean():.4f}")
        
        # Séparer en 2 catégories
        strict_results = []    # Distance ≤ 0.85 (très pertinent)
        generic_results = []   # 0.85 < Distance ≤ 1.0 (pertinent générique)
        
        for dist, idx in zip(distances[0], indices[0]):
            
            # Arrêter si on a assez de résultats stricts
            if len(strict_results) >= k and len(generic_results) >= k:
                break
            
            job = self.jobs_df.iloc[idx]
            similarity_score = 1 / (1 + dist)
            
            result = {
                'job_id': int(job['job_id']),
                'title': str(job['title']),
                'description': str(job['description'])[:300] + "...",
                'location': str(job['location']),
                'experience_level': str(job['formatted_experience_level']) if pd.notna(job['formatted_experience_level']) else 'Not specified',
                'confidence_level': str(job['confidence_level']),
                'distance': float(dist),
                'similarity_score': float(similarity_score)
            }
            
            # Classifier selon la distance
            if dist <= strict_threshold:
                if len(strict_results) < k:
                    result['rank'] = len(strict_results) + 1
                    result['result_type'] = 'strict'
                    strict_results.append(result)
            
            elif dist <= generic_threshold:
                if len(generic_results) < k:
                    result['rank'] = len(generic_results) + 1
                    result['result_type'] = 'generic'
                    generic_results.append(result)
        
        # Déterminer le type de requête
        if len(strict_results) >= 3:
            query_type = 'specific'  # Assez de résultats précis
        elif len(strict_results) == 0 and len(generic_results) > 0:
            query_type = 'generic'   # Seulement résultats génériques
        elif len(strict_results) > 0 and len(generic_results) > 0:
            query_type = 'mixed'     # Mix des deux
        else:
            query_type = 'none'      # Aucun résultat pertinent
        
        if self.debug:
            print(f"   Type détecté : {query_type}")
            print(f"   Stricts      : {len(strict_results)}")
            print(f"   Génériques   : {len(generic_results)}")
        
        return {
            'results': strict_results,
            'generic_results': generic_results,
            'query_type': query_type,
            'strict_count': len(strict_results),
            'generic_count': len(generic_results)
        }
    
    def get_stats(self):
        """
        Obtenir des statistiques sur le dataset
        
        Returns:
            dict: Statistiques
        """
        return {
            'total_jobs': len(self.jobs_df),
            'locations': self.jobs_df['location'].nunique(),
            'confidence_distribution': self.jobs_df['confidence_level'].value_counts().to_dict()
        }
    
    def analyze_query(self, query, k=10):
        """
        Analyser une requête en détail (mode debug)
        Utile pour comprendre les distances
        
        Args:
            query (str): Requête à analyser
            k (int): Nombre de résultats à analyser
        
        Returns:
            dict: Statistiques détaillées
        """
        
        # Encoder
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Chercher
        distances, indices = self.index.search(query_embedding, k)
        
        # Préparer les résultats détaillés
        results_detail = []
        for dist, idx in zip(distances[0], indices[0]):
            job = self.jobs_df.iloc[idx]
            results_detail.append({
                'title': str(job['title']),
                'distance': float(dist),
                'similarity_score': float(1 / (1 + dist)),
                'in_strict_range': dist <= 0.85,
                'in_generic_range': 0.85 < dist <= 1.0
            })
        
        # Stats globales
        stats = {
            'query': query,
            'min_distance': float(distances[0].min()),
            'max_distance': float(distances[0].max()),
            'mean_distance': float(distances[0].mean()),
            'median_distance': float(np.median(distances[0])),
            'std_distance': float(distances[0].std()),
            'results': results_detail,
            'strict_count': sum(1 for r in results_detail if r['in_strict_range']),
            'generic_count': sum(1 for r in results_detail if r['in_generic_range'])
        }
        
        return stats


# ============================================================================
# TESTS (si exécuté directement)
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🧪 TESTS DU MOTEUR DE RECHERCHE - SEUIL DOUBLE")
    print("="*80)
    
    # Initialiser en mode debug
    engine = JobSearchEngine(data_dir='../data/processed', debug=True)
    
    # ========================================================================
    # TEST 1 : Requête SPÉCIFIQUE (devrait avoir résultats stricts)
    # ========================================================================
    
    print("\n" + "🟢"*40)
    print("TEST 1 : REQUÊTE SPÉCIFIQUE")
    print("🟢"*40)
    
    result = engine.search("emergency room nurse", k=5)
    
    print(f"\n📊 RÉSULTAT :")
    print(f"   Type de requête : {result['query_type']}")
    print(f"   Résultats stricts   : {result['strict_count']}")
    print(f"   Résultats génériques : {result['generic_count']}")
    
    if result['results']:
        print(f"\n✅ TOP 3 RÉSULTATS STRICTS :")
        for i, res in enumerate(result['results'][:3], 1):
            print(f"   {i}. [{res['distance']:.4f}] {res['title']}")
    
    input("\n⏸️  Appuie sur ENTRÉE pour TEST 2...")
    
    # ========================================================================
    # TEST 2 : Requête GÉNÉRIQUE (devrait avoir résultats génériques)
    # ========================================================================
    
    print("\n" + "🟡"*40)
    print("TEST 2 : REQUÊTE GÉNÉRIQUE")
    print("🟡"*40)
    
    result = engine.search("job in health", k=5)
    
    print(f"\n📊 RÉSULTAT :")
    print(f"   Type de requête : {result['query_type']}")
    print(f"   Résultats stricts   : {result['strict_count']}")
    print(f"   Résultats génériques : {result['generic_count']}")
    
    if result['generic_results']:
        print(f"\n✅ TOP 3 RÉSULTATS GÉNÉRIQUES :")
        for i, res in enumerate(result['generic_results'][:3], 1):
            print(f"   {i}. [{res['distance']:.4f}] {res['title']}")
    
    input("\n⏸️  Appuie sur ENTRÉE pour TEST 3...")
    
    # ========================================================================
    # TEST 3 : Requête NON PERTINENTE (devrait n'avoir aucun résultat)
    # ========================================================================
    
    print("\n" + "🔴"*40)
    print("TEST 3 : REQUÊTE NON PERTINENTE")
    print("🔴"*40)
    
    result = engine.search("Salesforce developer", k=5)
    
    print(f"\n📊 RÉSULTAT :")
    print(f"   Type de requête : {result['query_type']}")
    print(f"   Résultats stricts   : {result['strict_count']}")
    print(f"   Résultats génériques : {result['generic_count']}")
    
    if result['query_type'] == 'none':
        print(f"\n✅ AUCUN RÉSULTAT PERTINENT (comme attendu)")
    
    input("\n⏸️  Appuie sur ENTRÉE pour TEST 4...")
    
    # ========================================================================
    # TEST 4 : ANALYSE DÉTAILLÉE
    # ========================================================================
    
    print("\n" + "📊"*40)
    print("TEST 4 : ANALYSE DÉTAILLÉE DES DISTANCES")
    print("📊"*40)
    
    queries = [
        "emergency room nurse",   # Spécifique
        "job in health",          # Générique
        "Salesforce developer"    # Non pertinent
    ]
    
    for query in queries:
        print(f"\n🔍 Analyse : '{query}'")
        print("-" * 80)
        
        analysis = engine.analyze_query(query, k=5)
        
        print(f"   Distance min : {analysis['min_distance']:.4f}")
        print(f"   Distance max : {analysis['max_distance']:.4f}")
        print(f"   Distance moy : {analysis['mean_distance']:.4f}")
        print(f"   Stricts (≤0.85)     : {analysis['strict_count']}")
        print(f"   Génériques (0.85-1.0): {analysis['generic_count']}")
        
        print(f"\n   Top 3 :")
        for i, res in enumerate(analysis['results'][:3], 1):
            status = "✅ STRICT" if res['in_strict_range'] else "🟡 GÉNÉRIQUE" if res['in_generic_range'] else "❌ HORS SEUIL"
            print(f"   {i}. [{res['distance']:.4f}] {status} - {res['title'][:50]}...")
    
    print("\n" + "="*80)
    print("✅ TOUS LES TESTS TERMINÉS")
    print("="*80)
    
    print("\n💡 CONCLUSION :")
    print("   • Requêtes spécifiques → Résultats stricts (distance ≤ 0.85)")
    print("   • Requêtes génériques  → Résultats génériques (0.85 < distance ≤ 1.0)")
    print("   • Requêtes hors sujet  → Aucun résultat (distance > 1.0)")
    print("\n🚀 Le système est prêt pour Flask !")