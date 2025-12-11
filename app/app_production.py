#!/usr/bin/env python3
"""
JOBMATCH - Backend Flask Production
API pour la recherche intelligente de jobs Healthcare

Endpoints:
- GET  /              → Serve landing page
- POST /api/search    → Recherche intelligente
- GET  /api/health    → Health check
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os

# Ajouter le chemin pour importer les modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step3_smart_engine import SmartJobSearchEngine

# ============================================================================
# INITIALISATION
# ============================================================================

app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Initialiser le SmartEngine (au démarrage de l'app)
print("\n" + "="*80)
print("🚀 JOBMATCH - Initialisation du Backend")
print("="*80 + "\n")

engine = SmartJobSearchEngine(data_dir='../data/processed')

print("\n" + "="*80)
print("✅ Backend prêt - SmartEngine chargé")
print("="*80 + "\n")


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve la landing page"""
    return render_template('index.html')


@app.route('/search')
def search_page():
    """Serve la search page"""
    return render_template('search.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'engine': 'ready',
        'dataset_size': len(engine.retriever.hybrid_retriever.jobs_df),
        'version': '3.0-smart'
    })


@app.route('/api/search', methods=['POST'])
def search():
    """
    Endpoint de recherche intelligente
    
    Body:
    {
        "query": "cna telemetry",
        "k": 10,
        "page": 1
    }
    
    Response:
    {
        "status": "success" | "rejected" | "no_results" | "low_confidence",
        "message": "...",
        "results": [...],
        "metadata": {...},
        "pagination": {...}
    }
    """
    
    try:
        # Récupérer les paramètres
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing "query" parameter',
                'results': []
            }), 400
        
        query = data.get('query', '').strip()
        k = data.get('k', 10)
        page = data.get('page', 1)
        
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query cannot be empty',
                'results': []
            }), 400
        
        # Log de la recherche
        print(f"🔍 Recherche : '{query}' (k={k}, page={page})")
        
        # Recherche avec SmartEngine
        response = engine.search(query, k=k*page)  # Récupérer k*page résultats
        
        # Convert numpy types to Python types (FIX)
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            import numpy as np
            import math
            
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                value = float(obj)
                # Handle NaN, Inf, -Inf
                if math.isnan(value) or math.isinf(value):
                    return None
                return value
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, float):
                # Handle Python float NaN
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            else:
                return obj
        
        # Apply conversion to entire response
        response = convert_numpy_types(response)
        
        # Pagination
        start_idx = (page - 1) * k
        end_idx = page * k
        
        if response['results']:
            total_results = len(response['results'])
            paginated_results = response['results'][start_idx:end_idx]
            
            response['results'] = paginated_results
            response['pagination'] = {
                'page': page,
                'per_page': k,
                'total_results': total_results,
                'total_pages': (total_results + k - 1) // k,
                'has_next': end_idx < total_results,
                'has_prev': page > 1
            }
        else:
            response['pagination'] = {
                'page': 1,
                'per_page': k,
                'total_results': 0,
                'total_pages': 0,
                'has_next': False,
                'has_prev': False
            }
        
        # Log du résultat
        print(f"✅ Status: {response['status']} - {len(response.get('results', []))} résultats\n")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Erreur : {str(e)}\n")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}',
            'results': []
        }), 500


@app.route('/api/job/<int:job_id>', methods=['GET'])
def get_job_details(job_id):
    """
    Récupérer les détails complets d'un job
    """
    try:
        jobs_df = engine.retriever.hybrid_retriever.jobs_df
        job = jobs_df[jobs_df['job_id'] == job_id]
        
        if job.empty:
            return jsonify({
                'status': 'error',
                'message': 'Job not found'
            }), 404
        
        job_data = job.iloc[0].to_dict()
        
        # Convert numpy/NaN types (same as search endpoint)
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            import numpy as np
            import math
            
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                value = float(obj)
                if math.isnan(value) or math.isinf(value):
                    return None
                return value
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            else:
                return obj
        
        job_data = convert_numpy_types(job_data)
        
        return jsonify({
            'status': 'success',
            'job': job_data
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Statistiques sur le dataset
    """
    try:
        jobs_df = engine.retriever.hybrid_retriever.jobs_df
        
        # Compter par niveau d'expérience
        experience_counts = jobs_df['formatted_experience_level'].value_counts().to_dict()
        
        stats = {
            'total_jobs': len(jobs_df),
            'experience_levels': experience_counts,
            'healthcare_vocab_size': len(engine.healthcare_vocab),
            'top_keywords': list(engine.healthcare_vocab)[:20]
        }
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🌐 Démarrage du serveur Flask")
    print("="*80)
    print("\n📍 URL : http://localhost:5004")
    print("📡 Endpoints disponibles :")
    print("   GET  /              → Landing page")
    print("   POST /api/search    → Recherche intelligente")
    print("   GET  /api/health    → Health check")
    print("   GET  /api/job/<id>  → Détails d'un job")
    print("   GET  /api/stats     → Statistiques dataset")
    print("\n" + "="*80 + "\n")
    
    # Mode debug pour développement
    app.run(
        host='0.0.0.0',
        port=5004,
        debug=True,
        threaded=True
    )