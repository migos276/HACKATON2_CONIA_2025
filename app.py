# ===== SUPPRESSION DES WARNINGS =====
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# ===== IMPORTS FLASK =====
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Ajoute cet import

# ===== IMPORTS TENSORFLOW =====
import tensorflow as tf
import numpy as np
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Concatenate, Dense, Input
from keras.models import Model

# Supposons que pooled1 et pooled2 sont


tensor1 = Input(shape=(7, 7, 1280))
tensor2 = Input(shape=(7, 7, 1280))
units = 256
pooled1 = GlobalAveragePooling2D()(tensor1)
pooled2 = GlobalAveragePooling2D()(tensor2)
merged = Concatenate()([pooled1, pooled2])
output = Dense(units, activation='relu')(merged)


# ===== IMPORTS STANDARD =====
import os
import base64
from io import BytesIO
from PIL import Image
import hashlib
import json
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CLASSE IMAGEPREDICTOR =====
class ImagePredictor:
    def __init__(self, model_path, class_names):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier du modèle est introuvable : {model_path}")
        print(f"[INFO] Chargement du modèle depuis : {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = (self.model.input_shape[1], self.model.input_shape[2])
        print(f"[INFO] Modèle chargé. Taille d'image attendue : {self.img_size}")

    def preprocess_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = preprocess_input(img_array)  # Ajoute cette ligne
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return img_array_expanded
    
    def preprocess_image_from_pil(self, pil_image):
        """Prétraite une image PIL pour la prédiction"""
        img_resized = pil_image.resize(self.img_size)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = preprocess_input(img_array)  # Ajoute cette ligne
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return img_array_expanded
    
    def predict_single_image(self, image_path):
        """
        Fait une prédiction sur une seule image.
        """
        # 1. Prétraiter l'image
        preprocessed_img = self.preprocess_image(image_path)
        
        # 2. Faire la prédiction
        predictions = self.model.predict(preprocessed_img, verbose=0)
        
        # 3. Interpréter les résultats
        score = predictions[0] 
        
        predicted_class_index = np.argmax(score)
        predicted_class_name = self.class_names[predicted_class_index]
        confidence_score = 100 * np.max(score)
        
        return predicted_class_name, confidence_score, predictions[0]
    
    def predict_from_pil_image(self, pil_image):
        """
        Fait une prédiction sur une image PIL.
        """
        # 1. Prétraiter l'image PIL
        preprocessed_img = self.preprocess_image_from_pil(pil_image)
        
        # 2. Faire la prédiction
        predictions = self.model.predict(preprocessed_img, verbose=0)
        
        # 3. Interpréter les résultats
        score = predictions[0] 
        
        predicted_class_index = np.argmax(score)
        predicted_class_name = self.class_names[predicted_class_index]
        confidence_score = 100 * np.max(score)
        
        return predicted_class_name, confidence_score, predictions[0]
    
    def predict_from_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            print(f"[ERREUR] Le dossier n'existe pas : {folder_path}")
            return
        print(f"\n--- Début des prédictions pour le dossier : {folder_path} ---\n")
        valid_extensions = ['.jpg', '.jpeg', '.png']
        for filename in sorted(os.listdir(folder_path)):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(folder_path, filename)
                try:
                    class_name, confidence, _ = self.predict_single_image(image_path)
                    print(f"-> Fichier: {filename:<30} | Prédiction: {class_name:<15} | Confiance: {confidence:.2f}%")
                except Exception as e:
                    print(f"[ERREUR] Impossible de traiter le fichier {filename}: {e}")
        print("\n--- Fin des prédictions ---")


app = Flask(__name__)
app.secret_key = 'votre_cle_secrete_ici'  # Changez ceci en production

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'best_food_model.h5'  # Utilisez le bon chemin

# Créer le dossier uploads s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Classes des plats
CLASS_NAMES = ['ekwang', 'eru', 'jollof-ghana', 'ndole', 'palm-nut-soup', 'waakye']

# Initialiser le prédicteur
try:
    predictor = ImagePredictor(model_path=MODEL_PATH, class_names=CLASS_NAMES)
    print("ImagePredictor initialisé avec succès!")
except Exception as e:
    print(f"Erreur lors de l'initialisation du prédicteur: {e}")
    predictor = None

# Descriptions enrichies des plats traditionnels
plats_traditionnels = {
    'ekwang': {
        'nom': 'Ekwang',
        'nom_local': 'Ekwang (Cameroun)',
        'description': 'Plat traditionnel camerounais à base de tubercules de taro râpés et enveloppés dans des feuilles de taro, cuit avec de la viande et du poisson. Un délice ancestral qui unit les familles.',
        'origine': 'Cameroun',
        'region': 'Région du Sud-Ouest',
        'ingredients': ['Taro râpé', 'Feuilles de taro', 'Viande de bœuf', 'Poisson fumé', 'Crevettes séchées', 'Huile de palme', 'Épices locales'],
        'temps_preparation': '2-3 heures',
        'difficulte': 'Intermédiaire',
        'valeur_nutritive': 'Riche en fibres, vitamines et protéines',
        'histoire': 'Plat cérémoniel souvent préparé lors des grandes occasions familiales',
        'emoji': '🌿',
        'couleur': '#2D5016'
    },
    'eru': {
        'nom': 'Eru',
        'nom_local': 'Eru (Okok)',
        'description': 'Plat traditionnel camerounais préparé avec des feuilles d\'eru (légume vert sauvage), accompagné de viande et de poisson fumé. Un trésor nutritionnel de la forêt équatoriale.',
        'origine': 'Cameroun',
        'region': 'Région du Sud et Centre',
        'ingredients': ['Feuilles d\'eru fraîches', 'Viande de bœuf', 'Poisson fumé', 'Crevettes', 'Huile de palme', 'Stockfish', 'Épices traditionnelles'],
        'temps_preparation': '1-2 heures',
        'difficulte': 'Facile',
        'valeur_nutritive': 'Très riche en fer, calcium et vitamines',
        'histoire': 'Légume sauvage récolté dans les forêts, symbole de connexion avec la nature',
        'emoji': '🥬',
        'couleur': '#1B4332'
    },
    'jollof-ghana': {
        'nom': 'Jollof Rice',
        'nom_local': 'Jollof Rice (Ghana)',
        'description': 'Riz parfumé cuit dans une sauce tomate épicée avec des légumes et de la viande, version ghanéenne du célèbre plat ouest-africain. Le roi des plats de fête!',
        'origine': 'Ghana',
        'region': 'Tout le Ghana',
        'ingredients': ['Riz jasmin', 'Tomates fraîches', 'Oignons', 'Viande de poulet', 'Épices ghanéennes', 'Légumes colorés', 'Bouillon de viande'],
        'temps_preparation': '45 minutes',
        'difficulte': 'Facile',
        'valeur_nutritive': 'Équilibré en glucides, protéines et légumes',
        'histoire': 'Plat de célébration, fierté culinaire ghanéenne dans la "guerre" du Jollof',
        'emoji': '🍚',
        'couleur': '#D2691E'
    },
    'ndole': {
        'nom': 'Ndolé',
        'nom_local': 'Ndolé (Plat National)',
        'description': 'Plat national du Cameroun préparé avec des feuilles de ndolé amères, des arachides grillées, de la viande et du poisson. L\'âme culinaire du Cameroun.',
        'origine': 'Cameroun',
        'region': 'National (origine Douala)',
        'ingredients': ['Feuilles de ndolé', 'Arachides grillées', 'Viande de bœuf', 'Poisson fumé', 'Crevettes fraîches', 'Stockfish', 'Huile de palme'],
        'temps_preparation': '2-3 heures',
        'difficulte': 'Difficile',
        'valeur_nutritive': 'Très riche en protéines, lipides sains et minéraux',
        'histoire': 'Plat des grandes occasions, symbole de l\'hospitalité camerounaise',
        'emoji': '🥜',
        'couleur': '#8B4513'
    },
    'palm-nut-soup': {
        'nom': 'Soupe de Noix de Palme',
        'nom_local': 'Palm Nut Soup',
        'description': 'Soupe traditionnelle ouest-africaine préparée avec l\'huile de palme rouge extraite des noix fraîches, de la viande et des légumes. Un concentré de saveurs ancestrales.',
        'origine': 'Afrique de l\'Ouest',
        'region': 'Ghana, Côte d\'Ivoire, Liberia',
        'ingredients': ['Noix de palme fraîches', 'Viande de chèvre', 'Poisson fumé', 'Légumes verts', 'Épices locales', 'Piment africain'],
        'temps_preparation': '3-4 heures',
        'difficulte': 'Difficile',
        'valeur_nutritive': 'Riche en vitamine A, antioxydants et acides gras essentiels',
        'histoire': 'Soupe sacrée dans certaines cultures, liée aux rituels de purification',
        'emoji': '🌴',
        'couleur': '#FF6B35'
    },
    'waakye': {
        'nom': 'Waakye',
        'nom_local': 'Waakye (Riz aux haricots)',
        'description': 'Plat ghanéen emblématique composé de riz et de haricots cuits ensemble avec des feuilles de millet, souvent servi avec diverses garnitures colorées. Le petit-déjeuner des champions!',
        'origine': 'Ghana',
        'region': 'Nord du Ghana (origine Hausa)',
        'ingredients': ['Riz local', 'Haricots noirs', 'Feuilles de millet séchées', 'Garnitures variées', 'Sauce tomate épicée', 'Œufs durs', 'Avocat'],
        'temps_preparation': '1 heure',
        'difficulte': 'Facile',
        'valeur_nutritive': 'Protéines complètes, fibres et glucides complexes',
        'histoire': 'Plat du petit-déjeuner devenu symbole de l\'identité ghanéenne urbaine',
        'emoji': '🍛',
        'couleur': '#8B0000'
    }
}

# Base de données simple des utilisateurs (en production, utilisez une vraie base de données)
users_db = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email et mot de passe requis'}), 400
        
        # Vérifier les identifiants (implémentation simplifiée)
        if email in users_db and users_db[email]['password'] == hashlib.sha256(password.encode()).hexdigest():
            session['user_email'] = email
            return redirect(url_for('dashboard'))
        else:
            return jsonify({'error': 'Identifiants incorrects'}), 401
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email et mot de passe requis'}), 400
        
        if email in users_db:
            return jsonify({'error': 'Utilisateur déjà existant'}), 400
        
        # Enregistrer l'utilisateur
        users_db[email] = {
            'password': hashlib.sha256(password.encode()).hexdigest(),
            'created_at': datetime.now().isoformat(),
            'predictions': []
        }
        
        session['user_email'] = email
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', user_email=session['user_email'])

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_email' not in session:
        return jsonify({'error': 'Utilisateur non connecté'}), 401
    
    if predictor is None:
        return jsonify({'error': 'Modèle non disponible'}), 500
    
    try:
        # Vérifier si c'est un fichier uploadé ou une image base64
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Aucun fichier sélectionné'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                # Utiliser la méthode predict_single_image de ImagePredictor
                predicted_class, confidence, all_predictions = predictor.predict_single_image(filepath)
                print(predicted_class,confidence,all_predictions)
                
                # Supprimer le fichier temporaire
                os.remove(filepath)
        
        elif 'image_data' in request.json:
            # Image prise par la caméra (base64)
            image_data = request.json['image_data']
            
            # Décoder l'image base64
            image_bytes = base64.b64decode(image_data.split(',')[1])
            pil_image = Image.open(BytesIO(image_bytes))
            pil_image = pil_image.convert('RGB')
            
            # Utiliser la méthode predict_from_pil_image de ImagePredictor
            predicted_class, confidence, all_predictions = predictor.predict_from_pil_image(pil_image)
        
        else:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        # Enregistrer la prédiction dans l'historique de l'utilisateur
        user_email = session['user_email']
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions.tolist()
        }
        users_db[user_email]['predictions'].append(prediction_data)
        
        # Obtenir les informations du plat
        plat_info = plats_traditionnels.get(predicted_class, {})
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'plat_info': plat_info,
            'all_predictions': {CLASS_NAMES[i]: float(all_predictions[i]) for i in range(len(CLASS_NAMES))}
        })
    
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500

@app.route('/plats-traditionnels')
def plats_traditionnels_route():
    return render_template('plats_traditionnels.html', plats=plats_traditionnels)

@app.route('/plat/<string:nom_plat>')
def detail_plat(nom_plat):
    if nom_plat not in plats_traditionnels:
        return jsonify({'error': 'Plat non trouvé'}), 404
    
    plat = plats_traditionnels[nom_plat]
    return render_template('detail_plat.html', plat=plat, nom_plat=nom_plat)

@app.route('/historique')
def historique():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    user_email = session['user_email']
    user_predictions = users_db.get(user_email, {}).get('predictions', [])
    
    return render_template('historique.html', predictions=user_predictions)

@app.route('/api/user-stats')
def user_stats():
    if 'user_email' not in session:
        return jsonify({'error': 'Utilisateur non connecté'}), 401
    
    user_email = session['user_email']
    user_data = users_db.get(user_email, {})
    predictions = user_data.get('predictions', [])
    
    # Calculer les statistiques
    total_predictions = len(predictions)
    plats_predits = {}
    
    for pred in predictions:
        plat = pred['predicted_class']
        plats_predits[plat] = plats_predits.get(plat, 0) + 1
    
    return jsonify({
        'total_predictions': total_predictions,
        'plats_predits': plats_predits,
        'membre_depuis': user_data.get('created_at', 'Inconnue')
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)