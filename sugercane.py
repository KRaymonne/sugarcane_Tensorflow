import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

# Configuration de la page
st.set_page_config(
    page_title="Sugarcane Leaf Disease Classifier - TensorFlow",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une interface élégante
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #2E8B57, #3CB371, #8FBC8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2E8B57;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .tensorflow-badge {
        background: linear-gradient(90deg, #FF6B00, #FFA726);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
        margin-left: 10px;
    }
    
    .card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #2E8B57;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.4);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 4px solid #2E8B57;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
    }
    
    .tech-badge {
        background: #FF6B00;
        color: white;
        padding: 3px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Constantes
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Healthy', 'Mosaic', 'Redrot', 'Rust', 'Yellow']
DISEASES_COLORS = ["#28a745", "#dc3545", "#800000", "#fd7e14", "#ffc107"]

# Titre principal avec badge TensorFlow
st.markdown('<h1 class="main-header">🌿 Sugarcane Leaf Disease Classifier <span class="tensorflow-badge">TensorFlow</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555; font-size: 1.2rem;">Système intelligent de diagnostic des maladies foliaires utilisant le Deep Learning avec TensorFlow</p>', unsafe_allow_html=True)

# Sidebar avec informations et navigation
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Modèles TensorFlow")
    
    # Information sur la technologie
    st.markdown("""
    <div style="background: #FF6B0010; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #FF6B00;">
        <strong>🛠️ Technologie :</strong>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <span class="tech-badge">TensorFlow</span>
            <span class="tech-badge" style="background: #4CAF50; margin-left: 5px;">Keras</span>
        </div>
        <small style="color: #666;">Version: {}</small>
    </div>
    """.format(tf.__version__), unsafe_allow_html=True)
    
    # Sélection du modèle
    model_option = st.radio(
        "Choisissez le modèle à utiliser:",
        ["Les deux modèles", "CNN TensorFlow", "DenseNet121 TensorFlow"],
        index=0
    )
    
    # Paramètres de confiance
    confidence_threshold = st.slider(
        "Seuil de confiance minimale (%)",
        min_value=50, max_value=95, value=75, step=5
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Information sur les modèles TensorFlow
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Architecture des Modèles")
    
    st.markdown("""
    <div style="background: #f0f8ff; padding: 10px; border-radius: 8px; margin: 5px 0;">
        <strong>CNN Personnalisé TensorFlow</strong>
        <ul style="margin: 5px 0; padding-left: 20px;">
            <li>Modèle séquentiel Keras</li>
            <li>Couches Conv2D + MaxPooling2D</li>
            <li>Flatten + Dense</li>
            <li>Test Accuracy: 89.20%</li>
        </ul>
    </div>
    
    <div style="background: #f0f8ff; padding: 10px; border-radius: 8px; margin: 5px 0;">
        <strong>DenseNet121 TensorFlow</strong>
        <ul style="margin: 5px 0; padding-left: 20px;">
            <li>Transfer Learning avec TensorFlow</li>
            <li>Base pré-entraînée sur ImageNet</li>
            <li>Fine-tuning avec Keras</li>
            <li>Test Accuracy: 83.03%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Fonction pour charger les modèles TensorFlow (mis en cache)
@st.cache_resource
def load_tensorflow_models():
    """Charge les modèles TensorFlow/Keras"""
    models = {}
    
    with st.spinner("🔧 Chargement des modèles TensorFlow..."):
        progress_bar = st.progress(0)
        
        try:
            # Afficher la version de TensorFlow
            st.info(f"TensorFlow Version: {tf.__version__}")
            
            
            # Charger CNN Simple TensorFlow
            models['cnn_tensorflow'] = tf.keras.models.load_model('CNN_Simple_tensorflow.h5')
            
            # Afficher l'architecture
            with st.expander("📐 Architecture du CNN TensorFlow"):
                # Créer un buffer pour capturer le résumé
                from io import StringIO
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    models['cnn_tensorflow'].summary(print_fn=lambda x: print(x))
                    summary_str = sys.stdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                st.text(summary_str)
            
            progress_bar.progress(50)
            
            # Charger DenseNet121 TensorFlow
            st.info("Chargement de DenseNet121 TensorFlow Transfert learning(best_model_tensorflow.h5)...")
            models['densenet_tensorflow'] = tf.keras.models.load_model('best_model_tensorflow.h5')
            
            with st.expander("📐 Architecture de DenseNet121"):
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    models['densenet_tensorflow'].summary(print_fn=lambda x: print(x))
                    summary_str = sys.stdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                st.text(summary_str)
            
            progress_bar.progress(100)
            
            st.success("✅ Modèles TensorFlow chargés avec succès!")
            
        except Exception as e:
            models = {
                'cnn_tensorflow': None,
                'densenet_tensorflow': None
            }
    
    return models

# Fonction de prétraitement TensorFlow
def preprocess_image_tensorflow(image, img_size=IMG_SIZE):
    """Prétraite une image pour TensorFlow"""
    # Redimensionner
    image = image.resize(img_size)
    
    # Convertir en array
    img_array = np.array(image)
    
    # Vérifier si l'image est en RGB (3 canaux)
    if len(img_array.shape) == 2:  # Image en niveaux de gris
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # Image RGBA
        img_array = img_array[:, :, :3]
    
    # Normaliser les valeurs des pixels (0-1) pour TensorFlow
    img_array = img_array / 255.0
    
    # Ajouter une dimension de batch (format TensorFlow)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Fonction de prédiction TensorFlow
def predict_with_tensorflow_model(model, image_array, model_name="Modèle"):
    """Effectue une prédiction avec un modèle TensorFlow"""
    if model is None:
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 0
    
    try:
        # Prédiction avec TensorFlow
        start_time = time.time()
        predictions = model.predict(image_array, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # en millisecondes
        
        # Formater les prédictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            pred = predictions[0]
        else:
            pred = tf.nn.softmax(predictions[0]).numpy()
        
        return pred, inference_time
    except Exception as e:
        st.error(f"Erreur TensorFlow avec {model_name}: {e}")
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 0

# Charger les modèles TensorFlow
models = load_tensorflow_models()

# Zone principale de l'application
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyse d'Image", "📊 Comparaison", "📚 Guide", "ℹ️ À Propos"])

with tab1:
    st.markdown('<h3 class="sub-header">🔍 Analyse d\'Image avec TensorFlow</h3>', unsafe_allow_html=True)
    
    # Section de téléchargement d'image
    uploaded_file = st.file_uploader(
        "Téléchargez une image de feuille de canne à sucre",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formats acceptés: JPG, JPEG, PNG, BMP - Compatible TensorFlow"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Feuille analysée", width='stretch')
            
            # Bouton d'analyse
            if st.button("🚀 Lancer l'Analyse TensorFlow", width='stretch'):
                with st.spinner("Analyse en cours avec TensorFlow..."):
                    # Prétraitement TensorFlow
                    image_array = preprocess_image_tensorflow(image)
                    
                    # Prédictions
                    predictions = {}
                    inference_times = {}
                    model_results = []
                    
                    if model_option in ["Les deux modèles", "CNN TensorFlow"]:
                        pred, inference_time = predict_with_tensorflow_model(
                            models['cnn_tensorflow'], 
                            image_array, 
                            "CNN TensorFlow"
                        )
                        predictions['cnn'] = pred
                        inference_times['cnn'] = inference_time
                        model_results.append(('CNN TensorFlow', pred, inference_time))
                    
                    if model_option in ["Les deux modèles", "DenseNet121 TensorFlow"]:
                        pred, inference_time = predict_with_tensorflow_model(
                            models['densenet_tensorflow'], 
                            image_array, 
                            "DenseNet121 TensorFlow"
                        )
                        predictions['densenet'] = pred
                        inference_times['densenet'] = inference_time
                        model_results.append(('DenseNet121 TensorFlow', pred, inference_time))
                    
                    # Afficher les résultats
                    st.markdown("---")
                    
                    for model_name, pred, inference_time in model_results:
                        st.markdown(f'<h4 class="sub-header">🧠 {model_name}</h4>', unsafe_allow_html=True)
                        
                        # Métriques
                        class_idx = np.argmax(pred)
                        confidence = pred[class_idx] * 100
                        
                        col1_metric, col2_metric, col3_metric = st.columns(3)
                        
                        with col1_metric:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; color: {DISEASES_COLORS[class_idx]}">
                                    {CLASS_NAMES[class_idx][0]}
                                </div>
                                <div style="font-size: 1.2rem; font-weight: bold;">
                                    {CLASS_NAMES[class_idx]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2_metric:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; color: {DISEASES_COLORS[class_idx]}">
                                    {confidence:.1f}%
                                </div>
                                <div>Confiance TensorFlow</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3_metric:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; color: #FF6B00">
                                    {inference_time:.1f}ms
                                </div>
                                <div>Temps d'inférence</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Graphique des probabilités
                        fig = go.Figure(data=[
                            go.Bar(
                                x=CLASS_NAMES,
                                y=pred * 100,
                                marker_color=DISEASES_COLORS,
                                text=[f"{p*100:.1f}%" for p in pred],
                                textposition='auto',
                            )
                        ])
                        
                        fig.update_layout(
                            title=f"Distribution des Probabilités - {model_name}",
                            xaxis_title="Maladies",
                            yaxis_title="Probabilité (%)",
                            height=400,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Conclusion et recommandations
                    if len(predictions) == 2:
                        st.markdown("---")
                        st.markdown('<h4 class="sub-header">💡 Analyse Comparative</h4>', unsafe_allow_html=True)
                        
                        # Comparer les prédictions des deux modèles
                        cnn_class = np.argmax(predictions['cnn'])
                        densenet_class = np.argmax(predictions['densenet'])
                        cnn_confidence = predictions['cnn'][cnn_class] * 100
                        densenet_confidence = predictions['densenet'][densenet_class] * 100
                        
                        if cnn_class == densenet_class:
                            st.markdown(f"""
                            <div class="success-box">
                                <h5>✅ Accord entre les modèles TensorFlow</h5>
                                <p>Les deux modèles TensorFlow s'accordent sur le diagnostic : <strong>{CLASS_NAMES[cnn_class]}</strong></p>
                                <ul>
                                    <li>CNN TensorFlow: {cnn_confidence:.1f}% de confiance</li>
                                    <li>DenseNet121 TensorFlow: {densenet_confidence:.1f}% de confiance</li>
                                </ul>
                                <p>Le diagnostic est considéré comme fiable.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-box">
                                <h5>⚠️ Désaccord entre les modèles TensorFlow</h5>
                                <p>Les modèles TensorFlow proposent des diagnostics différents :</p>
                                <ul>
                                    <li><strong>CNN TensorFlow</strong> : {CLASS_NAMES[cnn_class]} ({cnn_confidence:.1f}%)</li>
                                    <li><strong>DenseNet121 TensorFlow</strong> : {CLASS_NAMES[densenet_class]} ({densenet_confidence:.1f}%)</li>
                                </ul>
                                <p><strong>Recommandation :</strong> Considérer le diagnostic du CNN TensorFlow (89.20% de précision) et consulter un expert si nécessaire.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Boutons d'action
                    st.markdown("---")
                    st.markdown("### 📋 Actions")
                    
                    col_act1, col_act2, col_act3 = st.columns(3)
                    
                    # Générer un rapport
                    if st.session_state.get('generate_report', False):
                        # Créer un rapport textuel
                        report_content = f"""
                        RAPPORT D'ANALYSE - SUGARCANE LEAF DISEASE CLASSIFIER
                        {'='*60}
                        
                        Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
                        Image: {uploaded_file.name}
                        Dimensions: {image.width} x {image.height}
                        Modèle(s) utilisé(s): {model_option}
                        
                        RÉSULTATS TENSORFLOW:
                        """
                        
                        for model_name, pred, inference_time in model_results:
                            class_idx = np.argmax(pred)
                            confidence = pred[class_idx] * 100
                            report_content += f"\n\n{model_name}:"
                            report_content += f"\n  - Diagnostic: {CLASS_NAMES[class_idx]}"
                            report_content += f"\n  - Confiance: {confidence:.1f}%"
                            report_content += f"\n  - Temps d'inférence: {inference_time:.1f}ms"
                            report_content += f"\n  - Distribution des probabilités:"
                            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, pred)):
                                report_content += f"\n    * {cls}: {prob*100:.1f}%"
                        
                        report_content += f"\n\n{'='*60}"
                        report_content += "\nINFORMATIONS TECHNIQUES:"
                        report_content += f"\n- TensorFlow Version: {tf.__version__}"
                        report_content += f"\n- GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}"
                        
                        # Créer un bouton de téléchargement
                        st.download_button(
                            label="📥 Télécharger le Rapport",
                            data=report_content,
                            file_name=f"diagnostic_tensorflow_{uploaded_file.name.split('.')[0]}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
        
        with col2:
            st.markdown("### 📋 Informations Techniques")
            st.info(f"**TensorFlow Version:** {tf.__version__}")
            st.info("**Backend:** Keras Sequential API")
            st.info("**Format d'entrée:** 224x224 RGB")
            st.info("**Normalisation:** [0, 1]")
            
            # Information sur l'image
            st.markdown("### 🖼️ Détails de l'Image")
            st.metric("Dimensions", f"{image.width} × {image.height}")
            st.metric("Mode", image.mode)
            st.metric("Taille", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Conseils
            st.markdown("### 🎯 Conseils")
            st.info("""
            Pour de meilleurs résultats TensorFlow :
            1. Image bien éclairée
            2. Feuille centrée
            3. Format RGB
            4. Résolution minimale: 224x224
            """)
    
    else:
        # Section d'exemple quand aucune image n'est téléchargée
        st.markdown("""
        <div class="card">
            <h4>📋 Comment utiliser cette application :</h4>
            <ol>
                <li><strong>Téléchargez</strong> une image de feuille de canne à sucre</li>
                <li><strong>Sélectionnez</strong> le(s) modèle(s) TensorFlow</li>
                <li><strong>Cliquez</strong> sur "🚀 Lancer l'Analyse TensorFlow"</li>
                <li><strong>Consultez</strong> les résultats détaillés</li>
            </ol>
            
            <h4>🎯 Modèles TensorFlow disponibles :</h4>
            <ul>
                <li><strong>CNN TensorFlow</strong> : Architecture personnalisée (89.20% accuracy)</li>
                <li><strong>DenseNet121 TensorFlow</strong> : Transfer Learning (83.03% accuracy)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown('<h3 class="sub-header">📊 Comparaison des Modèles TensorFlow</h3>', unsafe_allow_html=True)
    
    # Données de comparaison
    st.markdown("### 📈 Performances Comparées")
    
    models_comparison = ['CNN TensorFlow', 'DenseNet121 TensorFlow']
    accuracy = [89.20, 83.03]
    f1_scores = [85.11, 83.09]
    inference_time = [45.0, 28.0]  # Temps d'inférence estimés en ms
    
    # Graphique comparatif
    fig = go.Figure(data=[
        go.Bar(name='Test Accuracy (%)', x=models_comparison, y=accuracy, marker_color='#2E8B57'),
        go.Bar(name='F1 Macro Score (%)', x=models_comparison, y=f1_scores, marker_color='#3CB371'),
        go.Bar(name='Temps Inférence (ms)', x=models_comparison, y=inference_time, marker_color='#FF6B00')
    ])
    
    fig.update_layout(
        barmode='group',
        title='Comparaison des Performances des Modèles TensorFlow',
        xaxis_title="Modèles",
        yaxis_title="Valeurs",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau de comparaison
    st.markdown("### 📋 Caractéristiques Techniques")
    
    comparison_data = {
        "Modèle": ["CNN TensorFlow", "DenseNet121 TensorFlow"],
        "Test Accuracy": ["89.20%", "83.03%"],
        "F1 Macro": ["85.11%", "83.09%"],
        "Architecture": ["Conv2D + MaxPooling2D", "DenseNet121 pré-entraîné"],
        "Framework": ["TensorFlow/Keras", "TensorFlow/Keras"],
        "Format": [".h5", ".h5"],
        "Recommandé": ["🏆", "✅"]
    }
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    # Code d'implémentation TensorFlow
    st.markdown("### 🛠️ Implémentation TensorFlow")
    
    code_tab1, code_tab2 = st.tabs(["CNN TensorFlow", "DenseNet121 TensorFlow"])
    
    with code_tab1:
        st.code("""
# Architecture CNN avec TensorFlow
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', 
                       input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
        """, language='python')
    
    with code_tab2:
        st.code("""
# Transfer Learning avec DenseNet121
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121

# Charger DenseNet121 pré-entraîné
base_model = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
        """, language='python')

with tab3:
    st.markdown('<h3 class="sub-header">📚 Guide des Maladies</h3>', unsafe_allow_html=True)
    
    diseases_info = [
        {
            "name": "Healthy",
            "color": "#28a745",
            "description": "Feuille saine sans signe de maladie",
            "symptoms": ["Couleur verte uniforme", "Texture lisse", "Pas de taches", "Croissance normale"],
            "prevention": "Maintenir de bonnes pratiques agricoles",
            "treatment": "Aucun traitement nécessaire"
        },
        {
            "name": "Mosaic",
            "color": "#dc3545",
            "description": "Maladie virale causée par le Sugarcane Mosaic Virus",
            "symptoms": ["Mosaïque de taches vert clair et foncé", "Feuilles striées", "Rachitisme"],
            "prevention": "Variétés résistantes, contrôle des pucerons",
            "treatment": "Élimination des plantes infectées"
        },
        {
            "name": "Redrot",
            "color": "#800000",
            "description": "Maladie fongique causée par Colletotrichum falcatum",
            "symptoms": ["Lésions rougeâtres", "Pourriture de la tige", "Odeur caractéristique"],
            "prevention": "Drainage approprié, rotation des cultures",
            "treatment": "Fongicides (Carbendazim, Mancozeb)"
        },
        {
            "name": "Rust",
            "color": "#fd7e14",
            "description": "Maladie fongique causée par Puccinia melanocephala",
            "symptoms": ["Pustules orange-rouille", "Feuilles jaunissantes", "Chute des feuilles"],
            "prevention": "Variétés résistantes, éviter excès d'azote",
            "treatment": "Fongicides à base de triazole"
        },
        {
            "name": "Yellow",
            "color": "#ffc107",
            "description": "Syndrome des feuilles jaunes causé par un phytoplasme",
            "symptoms": ["Jaunissement généralisé", "Rachitisme", "Réduction de croissance"],
            "prevention": "Contrôle des insectes vecteurs",
            "treatment": "Élimination des plantes infectées"
        }
    ]
    
    for disease in diseases_info:
        with st.expander(f"**{disease['name']}**"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"""
                <div style="width: 100px; height: 100px; background-color: {disease['color']}; 
                            border-radius: 50%; display: flex; align-items: center; 
                            justify-content: center; margin: auto;">
                    <span style="color: white; font-size: 2rem; font-weight: bold;">
                        {disease['name'][0]}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Description:** {disease['description']}")
                st.markdown("**Symptômes:**")
                for symptom in disease['symptoms']:
                    st.markdown(f"- {symptom}")
                st.markdown(f"**Prévention:** {disease['prevention']}")
                st.markdown(f"**Traitement:** {disease['treatment']}")

with tab4:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">ℹ️ À Propos - Implementation TensorFlow</h3>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card">
            <h4>🚀 Stack Technologique</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 15px 0;">
                <div style="background: #FF6B00; color: white; padding: 10px; border-radius: 8px; text-align: center;">
                    <strong>TensorFlow</strong><br>v{tf.__version__}
                </div>
                <div style="background: #4CAF50; color: white; padding: 10px; border-radius: 8px; text-align: center;">
                    <strong>Keras API</strong><br>High-level API
                </div>
                <div style="background: #2196F3; color: white; padding: 10px; border-radius: 8px; text-align: center;">
                    <strong>Streamlit</strong><br>v{st.__version__}
                </div>
                <div style="background: #FF9800; color: white; padding: 10px; border-radius: 8px; text-align: center;">
                    <strong>NumPy</strong><br>v{np.__version__}
                </div>
            </div>
            
            <h5>📊 Modèles Entraînés avec TensorFlow</h5>
            <ul>
                <li><strong>CNN TensorFlow</strong> : Modèle convolutionnel personnalisé (89.20% accuracy)</li>
                <li><strong>DenseNet121 TensorFlow</strong> : Transfer Learning avec fine-tuning (83.03% accuracy)</li>
            </ul>
        </div>
        """)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>📁 Fichiers TensorFlow</h4>
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px;">
                <strong>Modèles .h5 :</strong>
                <ul style="margin-top: 10px;">
                    <li>CNN_Simple_tensorflow.h5</li>
                    <li>best_model_tensorflow.h5</li>
                </ul>
                <p style="margin-top: 10px; font-size: 0.9rem;">
                    Format: HDF5 (TensorFlow/Keras)
                </p>
            </div>
        </div>
        
        <div class="card">
            <h4>⚙️ Configuration</h4>
            <ul>
                <li>Framework: TensorFlow 2.x</li>
                <li>Backend: Keras</li>
                <li>Image Size: 224×224</li>
                <li>Batch Size: 32</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Pied de page
st.markdown("---")
col1_footer, col2_footer, col3_footer = st.columns([1, 2, 1])

with col2_footer:
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        <p>🌿 <strong>Sugarcane Leaf Disease Classifier</strong> - TensorFlow Implementation</p>
        <p>TensorFlow {tf.__version__} • Modèles .h5 • Accuracy: 89.20%</p>
        <p>© 2024 - Développé avec TensorFlow et Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Initialisation de la session
if 'generate_report' not in st.session_state:
    st.session_state.generate_report = False

# Message de bienvenue
st.toast("Application TensorFlow prête !", icon="✅")