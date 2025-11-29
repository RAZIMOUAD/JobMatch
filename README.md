# JobMatch – Semantic Job Search Engine

## 🎯 Objectif
Développer un moteur de recherche **sémantique** pour offres d'emploi utilisant **BERT embeddings**.
- Domaine spécialisé : [IT / Healthcare / Finance / Marketing]
- Modèle : Sentence-BERT (all-MiniLM-L6-v2)
- Recherche : Similarité cosinus sur embeddings denses
- Interface : Flask + HTML/CSS minimaliste
- Évaluation : Precision, Recall, Semantic relevance

## 📊 Dataset
- **Source :** Kaggle – Job Descriptions Dataset (123,849 offres)
- **URL :** https://www.kaggle.com/datasets/rashikrahmanbd/job-descriptions-dataset
- **Domaine sélectionné :** [À déterminer après exploration]
- **Taille finale :** ~300-500 offres (niché)

## 🔧 Architecture
```
JobMatch/
├── data/
│   ├── raw/jobs_filtered.csv          # 123,849 offres brutes
│   └── processed/
│       ├── domain_jobs.csv            # Offres filtrées par domaine
│       ├── embeddings.pkl             # SBERT embeddings
│       └── metadata.json              # Job IDs, titles, links
├── notebooks/
│   └── exploration.ipynb              # Analyse + domain selection
├── src/
│   ├── domain_filter.py               # Filtrage par domaine
│   ├── sbert_encoder.py               # Encoding SBERT
│   ├── search.py                      # Fonction de recherche sémantique
│   └── utils.py                       # Helpers
├── app/
│   ├── app.py                         # Flask app
│   └── templates/
│       ├── index.html                 # Interface recherche
│       └── results.html               # Affichage résultats
├── evaluation/
│   └── manual_judgments.csv           # Requêtes test + relevance scores
├── requirements.txt                   # Dépendances
├── DECISIONS.md                       # Choix techniques documentés
└── README.md                          # Ce fichier
```

## 🚀 Stack Technique
- **Python 3.9+**
- **Sentence-BERT** (all-MiniLM-L6-v2)
- **FAISS** (index embeddings)
- **Flask** (API + interface)
- **Pandas** (data processing)
- **NumPy/SciPy** (cosine similarity)

## 📅 Timeline
- **Jour 1 :** Exploration + Domain Selection
- **Jour 2 :** Filtrage + Nettoyage données
- **Jour 3 :** SBERT Encoding + Index
- **Jour 4 :** Fonction de recherche
- **Jour 5 :** Interface Flask
- **Jour 6 :** Evaluation manuelle
- **Jour 7 :** Rapport + Vidéo démo

## 📝 Notes
Module : **SRI & Big Data 2025-2026**
Deadline : **14 décembre 2025**
Approche : **Semantic search** (pas keyword search)
```

---

## 🎯 **LE PROMPT POUR LA PROCHAINE CONVERSATION**

Copie-colle ça au début de ta prochaine demande :
```
# 🚀 JOBMATCH – SEMANTIC JOB SEARCH (SBERT)

## 📊 État du projet
- ✅ Data : 123,849 offres d'emploi (CSV)
- ✅ Architecture : créée (dossiers vides)
- ✅ Technos : décidées (SBERT, Flask, FAISS)
- ⏳ Action : à commencer MAINTENANT

## 🎯 Phase actuelle : EXPLORATION + DOMAIN SELECTION

### Étape 1️⃣ : Explore les données
→ Exécute les tests du Notebook pour voir distribution par domaine

### Étape 2️⃣ : Décide le domaine
→ Basé sur exploration, choisis : A) IT, B) Healthcare, C) Finance, D) Marketing

### Étape 3️⃣ : Filtre & Prépare
→ Script Python : 123k offres → ~300-500 offres (domaine niché)

### Étape 4️⃣ : Encode SBERT
→ Script : offres → embeddings SBERT 384-dim → sauvegarde FAISS

### Étape 5️⃣ : Fonction recherche
→ Implémente : requête utilisateur → embedding → top-10 résultats cosinus

### Étape 6️⃣ : Interface Flask
→ HTML simple : champ recherche + affichage résultats

### Étape 7️⃣ : Evaluation
→ 5-10 requêtes manuelles, calcule scores pertinence

### Étape 8️⃣ : Rapport
→ Documente choix, résultats, améliorations possibles

## 📋 Constraints
- **NO planning/documentation** : action directe
- **Small steps** : 1 étape à la fois, visible output
- **Fast execution** : 30 min par étape max
- **Git commits** : après chaque étape

## 🔗 Ressources
- Code template + explications concises
- Pas de bla-bla théorique
- Just action + résultats

## ⚡ Commencer par
Exécute le Notebook d'exploration pour identifier les domaines dans les données.
Dis-moi le résultat, on décide et on attaque filtrage.