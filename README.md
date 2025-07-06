# 🍽️ API de Reconnaissance de Plats Traditionnels Africains

Une API Flask intelligente utilisant TensorFlow pour identifier automatiquement les plats traditionnels africains à partir d'images. Le système peut reconnaître 6 plats emblématiques avec des informations culturelles détaillées.

## 📋 Table des Matières

- [Fonctionnalités](#fonctionnalités)
- [Plats Reconnus](#plats-reconnus)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Endpoints de l'API](#endpoints-de-lapi)
- [Structure du Projet](#structure-du-projet)
- [Déploiement](#déploiement)
- [Contribution](#contribution)

## ✨ Fonctionnalités

- **Reconnaissance d'images** : Identification automatique de plats traditionnels africains
- **Authentification utilisateur** : Système de connexion/inscription sécurisé
- **Historique des prédictions** : Suivi des analyses précédentes
- **Informations culturelles** : Détails sur l'origine, les ingrédients et l'histoire de chaque plat
- **Interface web** : Dashboard interactif avec capture photo et upload de fichiers
- **Statistiques utilisateur** : Analyse des habitudes de reconnaissance

## 🥘 Plats Reconnus

L'API peut identifier les plats suivants :

| Plat | Origine | Description |
|------|---------|-------------|
| **Ekwang** | Cameroun | Tubercules de taro râpés enveloppés dans des feuilles |
| **Eru (Okok)** | Cameroun | Feuilles d'eru sauvage avec viande et poisson fumé |
| **Jollof Rice** | Ghana | Riz parfumé dans une sauce tomate épicée |
| **Ndolé** | Cameroun | Plat national aux feuilles amères et arachides |
| **Palm Nut Soup** | Afrique de l'Ouest | Soupe à l'huile de palme rouge |
| **Waakye** | Ghana | Riz aux haricots avec feuilles de millet |

## 🚀 Installation

### Prérequis

- Python 3.8+
- TensorFlow 2.x
- Flask
- Modèle pré-entraîné (`best_food_model.h5`)

### Installation des dépendances

```bash
# Cloner le repository
git clone <your-repo-url>
cd african-food-recognition-api

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Fichier requirements.txt

```txt
Flask==2.3.3
tensorflow==2.13.0
Pillow==10.0.0
numpy==1.24.3
Werkzeug==2.3.7
```

## ⚙️ Configuration

### 1. Modèle de Machine Learning

Placez votre modèle pré-entraîné dans le répertoire racine :
```
best_food_model.h5
```

### 2. Configuration Flask

Modifiez les paramètres dans le fichier principal :

```python
# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'best_food_model.h5'
app.secret_key = 'votre_cle_secrete_ici'  # ⚠️ Changez en production
```

### 3. Structure des dossiers

```
african-food-api/
├── app.py                 # Application principale
├── best_food_model.h5     # Modèle TensorFlow
├── uploads/               # Dossier temporaire (auto-créé)
├── templates/             # Templates HTML
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── plats_traditionnels.html
│   ├── detail_plat.html
│   └── historique.html
├── static/               # Fichiers statiques (CSS, JS)
├── requirements.txt
└── README.md
```

## 🎯 Usage

### Démarrer l'API

```bash
python app.py
```

L'API sera accessible sur `http://localhost:5000`

### Utilisation via Interface Web

1. **Inscription/Connexion** : Créez un compte ou connectez-vous
2. **Dashboard** : Accédez au tableau de bord principal
3. **Prédiction** : 
   - Uploadez une image
   - Ou prenez une photo avec la caméra
4. **Résultats** : Consultez les prédictions avec informations culturelles
5. **Historique** : Visualisez vos analyses précédentes

### Utilisation via API REST

```python
import requests
import base64

# Authentification
session = requests.Session()
login_data = {
    'email': 'user@example.com',
    'password': 'motdepasse'
}
session.post('http://localhost:5000/login', data=login_data)

# Prédiction avec fichier
with open('image_plat.jpg', 'rb') as f:
    files = {'file': f}
    response = session.post('http://localhost:5000/predict', files=files)
    result = response.json()

print(f"Plat prédit: {result['predicted_class']}")
print(f"Confiance: {result['confidence']:.2f}%")
```

## 🔗 Endpoints de l'API

### Authentification

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/login` | POST | Connexion utilisateur |
| `/register` | POST | Inscription utilisateur |
| `/logout` | GET | Déconnexion |

### Prédiction

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/predict` | POST | Analyse d'image (fichier ou base64) |

**Exemple de réponse :**
```json
{
  "predicted_class": "ndole",
  "confidence": 95.67,
  "plat_info": {
    "nom": "Ndolé",
    "description": "Plat national du Cameroun...",
    "origine": "Cameroun",
    "ingredients": ["Feuilles de ndolé", "Arachides grillées", ...]
  },
  "all_predictions": {
    "ekwang": 0.02,
    "eru": 0.01,
    "jollof-ghana": 0.01,
    "ndole": 0.96,
    "palm-nut-soup": 0.00,
    "waakye": 0.00
  }
}
```

### Données Utilisateur

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/historique` | GET | Historique des prédictions |
| `/api/user-stats` | GET | Statistiques utilisateur |

### Informations Culturelles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/plats-traditionnels` | GET | Liste des plats avec infos |
| `/plat/<nom_plat>` | GET | Détails d'un plat spécifique |

## 🏗️ Structure du Projet

### Classe ImagePredictor

```python
class ImagePredictor:
    def __init__(self, model_path, class_names)
    def preprocess_image(self, image_path)
    def preprocess_image_from_pil(self, pil_image)
    def predict_single_image(self, image_path)
    def predict_from_pil_image(self, pil_image)
    def predict_from_folder(self, folder_path)
```

### Base de Données

Le système utilise une base de données en mémoire simple. Pour la production, migrez vers :
- PostgreSQL
- MongoDB
- SQLite avec SQLAlchemy

## 🚀 Déploiement

### Déploiement Local

```bash
# Mode développement
python app.py

# Mode production avec Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Déploiement Cloud

#### Heroku
```bash
# Fichier Procfile
web: gunicorn app:app

# Déploiement
heroku create african-food-api
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🔒 Sécurité

### Recommandations pour la Production

1. **Clé secrète** : Utilisez une clé forte et unique
2. **HTTPS** : Activez SSL/TLS
3. **Base de données** : Migrez vers une DB sécurisée
4. **Validation** : Renforcez la validation des fichiers
5. **Rate limiting** : Limitez les requêtes par utilisateur

```python
# Exemple de configuration sécurisée
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app.secret_key = os.environ.get('SECRET_KEY')
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

## 🤝 Contribution

1. Fork le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🆘 Support

Pour toute question ou problème :
- Ouvrez une issue sur GitHub
- Contactez l'équipe de développement
- Consultez la documentation technique

## 📊 Performances

- **Précision du modèle** : 92%+ sur les plats testés
- **Temps de réponse** : < 2 secondes par prédiction
- **Formats supportés** : JPG, PNG, JPEG, GIF
- **Taille max fichier** : 10MB (configurable)

---

*Développé avec ❤️ pour préserver et célébrer la richesse culinaire africaine*
