# JobMatch – Moteur de Recherche d'Offres d'Emploi

## 📌 Objectif
Développer une application de recherche d'information (SRI) pour les offres d'emploi.
- Indexation automatique avec TF-IDF
- Recherche vectorielle par similarité cosinus
- Interface Flask simple
- Évaluation (précision, rappel, F1-score)

## 📊 Dataset
- **Source :** Kaggle – Job Descriptions Dataset
- **URL :** https://www.kaggle.com/datasets/rashikrahmanbd/job-descriptions-dataset
- **Taille brute :** 123,849 offres d'emploi
- **Sous-ensemble utilisé :** 1,500 offres (échantillon)

## 🗂️ Structure du Projet
```
JobMatch/
├── data/raw/              # Données brutes (non commitées)
├── data/processed/        # Données nettoyées
├── src/                   # Code Python (indexation, recherche, éval)
├── app/                   # Application Flask
├── report/                # Rapport technique PDF
└── tests/                 # Tests unitaires
```

## 🚀 Prochaines Étapes
- [ ] Jour 1 : Nettoyage du dataset
- [ ] Jour 2 : Indexation TF-IDF
- [ ] Jour 3 : Recherche + classement
- [ ] Jour 4 : Interface Flask
- [ ] Jour 5 : Évaluation
- [ ] Jour 6 : IA optionnelle (Cortex)
- [ ] Jour 7 : Rapport + vidéo

## 🛠️ Stack Technologique
- Python 3.9+
- Flask
- scikit-learn (TF-IDF)
- NLTK / spaCy
- HTML/CSS

## 📝 Notes
Projet réalisé dans le cadre du module **SRI & Big Data 2025-2026**
Deadline : 14 décembre 2025
```

---

### Étape 3️⃣ : Créer un `requirements.txt` initial

**À la racine, crée/remplis `requirements.txt`** (on va l'enrichir progressivement) :
```
pandas==2.0.3
scikit-learn==1.3.0
nltk==3.8.1
spacy==3.7.2
flask==3.0.0
python-dotenv==1.0.0