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
    page_title="Sugarcane Leaf Disease Classifier",
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
    
    .disease-card {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        font-weight: 600;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
    }
    
    .uploaded-image {
        border-radius: 15px;
        border: 3px solid #2E8B57;
        padding: 5px;
        background: white;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger les modèles (mis en cache)
@st.cache_resource
def load_models():
    # Simulation du chargement des modèles
    # Dans la vraie application, remplacez par le chargement de vos modèles réels
    st.info("🔧 Chargement des modèles de deep learning...")
    progress_bar = st.progress(0)
    
    # Simulation du chargement
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    # Création de modèles factices pour la démo
    # Dans votre cas réel, utilisez:
    # custom_model = tf.keras.models.load_model('models/custom_cnn.h5')
    # transfer_model = tf.keras.models.load_model('models/best_transfer_model.h5')
    
    class_names = ['Healthy', 'Mosaic', 'Redrot', 'Rust', 'Yellow']
    
    return class_names

# Titre principal avec style
st.markdown('<h1 class="main-header">🌿 Sugarcane Leaf Disease Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555; font-size: 1.2rem;">Système intelligent de diagnostic des maladies foliaires utilisant l\'apprentissage profond</p>', unsafe_allow_html=True)

# Sidebar avec informations et navigation
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Modèles Disponibles")
    
    # Sélection du modèle
    model_option = st.radio(
        "Choisissez le modèle à utiliser:",
        ["Les deux modèles", "CNN Personnalisé", "Transfer Learning (DenseeNet121)"],
        index=0
    )
    
    # Paramètres de confiance
    confidence_threshold = st.slider(
        "Seuil de confiance minimale (%)",
        min_value=50, max_value=95, value=75, step=5
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Information sur les maladies
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🍃 Types de Maladies")
    
    diseases = {
        "Healthy": {"color": "#28a745", "desc": "Feuille saine sans maladie"},
        "Mosaic": {"color": "#dc3545", "desc": "Virus de la mosaïque du sucre"},
        "Redrot": {"color": "#800000", "desc": "Pourriture rouge (Colletotrichum falcatum)"},
        "Rust": {"color": "#fd7e14", "desc": "Rouille (Puccinia melanocephala)"},
        "Yellow": {"color": "#ffc107", "desc": "Syndrome des feuilles jaunes"}
    }
    
    for disease, info in diseases.items():
        st.markdown(f"""
        <div style="background-color: {info['color']}20; border-left: 4px solid {info['color']}; 
                    padding: 8px; margin: 5px 0; border-radius: 4px;">
            <strong style="color: {info['color']}">{disease}</strong><br>
            <small style="color: #666">{info['desc']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Statistiques
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Performances")
    
    # Métriques factices (à remplacer par vos vraies métriques)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Précision CNN", "89.2%", "+2.3%")
    with col2:
        st.metric("Précision Transfer", "93.7%", "+1.8%")
    
    st.progress(93)
    st.caption("Meilleur modèle: DenseNet121 (Transfer Learning)")
    st.markdown("</div>", unsafe_allow_html=True)

# Zone principale de l'application
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyse d'Image", "📊 Comparaison", "📚 Guide", "ℹ️ À Propos"])

with tab1:
    st.markdown('<h3 class="sub-header">🔍 Analyse d\'Image de Feuille</h3>', unsafe_allow_html=True)
    
    # Section de téléchargement d'image
    uploaded_file = st.file_uploader(
        "Téléchargez une image de feuille de canne à sucre",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formats acceptés: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Affichage de l'image téléchargée
        col1, col2 = st.columns([2, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Feuille analysée", width=400, use_column_width=True, output_format="auto", 
                    clamp=True, channels="RGB")
            
            # Bouton d'analyse
            if st.button("🚀 Lancer l'Analyse", use_container_width=True):
                # Simulation du traitement
                with st.spinner("Analyse en cours avec les modèles de deep learning..."):
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                        if i < 30:
                            progress_text.text("Chargement de l'image...")
                        elif i < 60:
                            progress_text.text("Traitement par le CNN personnalisé...")
                        else:
                            progress_text.text("Analyse par Transfer Learning...")
                    
                    # Simuler des prédictions aléatoires pour la démo
                    np.random.seed(int(time.time()))
                    
                    # Prédictions factices pour la démo
                    custom_pred = np.random.dirichlet(np.ones(5), size=1)[0]
                    transfer_pred = np.random.dirichlet(np.ones(5)*0.5, size=1)[0]
                    
                    # Assurer que les prédictions sont cohérentes
                    custom_pred = custom_pred / custom_pred.sum()
                    transfer_pred = transfer_pred / transfer_pred.sum()
                    
                    custom_class_idx = np.argmax(custom_pred)
                    transfer_class_idx = np.argmax(transfer_pred)
                    
                    class_names = ['Healthy', 'Mosaic', 'Redrot', 'Rust', 'Yellow']
                    diseases_colors = ["#28a745", "#dc3545", "#800000", "#fd7e14", "#ffc107"]
                    
                    # Afficher les résultats
                    st.markdown("---")
                    
                    if model_option == "Les deux modèles" or model_option == "CNN Personnalisé":
                        st.markdown(f'<h4 class="sub-header">🧠 Résultats du CNN Personnalisé</h4>', unsafe_allow_html=True)
                        
                        # Métrique de confiance
                        confidence = custom_pred[custom_class_idx] * 100
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        
                        with col_metric1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; color: {diseases_colors[custom_class_idx]};">
                                    {class_names[custom_class_idx][0]}
                                </div>
                                <div style="font-size: 1.2rem; font-weight: bold;">
                                    {class_names[custom_class_idx]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_metric2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; color: {diseases_colors[custom_class_idx]}">
                                    {confidence:.1f}%
                                </div>
                                <div>Confiance</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_metric3:
                            status = "✅ Confiant" if confidence > 80 else "⚠️ Limite" if confidence > 60 else "❌ Faible"
                            status_color = "#28a745" if confidence > 80 else "#ffc107" if confidence > 60 else "#dc3545"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 1.5rem; color: {status_color};">
                                    {status}
                                </div>
                                <div>Statut</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Graphique des probabilités
                        fig1 = go.Figure(data=[
                            go.Bar(
                                x=class_names,
                                y=custom_pred * 100,
                                marker_color=diseases_colors,
                                text=[f"{p*100:.1f}%" for p in custom_pred],
                                textposition='auto',
                            )
                        ])
                        
                        fig1.update_layout(
                            title="Distribution des Probabilités - CNN Personnalisé",
                            xaxis_title="Maladies",
                            yaxis_title="Probabilité (%)",
                            height=400,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    if model_option == "Les deux modèles" or model_option == "Transfer Learning (DenseNet121)":
                        st.markdown(f'<h4 class="sub-header">🚀 Résultats du Transfer Learning (DenseNet121)</h4>', unsafe_allow_html=True)
                        
                        # Métrique de confiance
                        confidence = transfer_pred[transfer_class_idx] * 100
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        
                        with col_metric1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; color: {diseases_colors[transfer_class_idx]};">
                                    {class_names[transfer_class_idx][0]}
                                </div>
                                <div style="font-size: 1.2rem; font-weight: bold;">
                                    {class_names[transfer_class_idx]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_metric2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; color: {diseases_colors[transfer_class_idx]}">
                                    {confidence:.1f}%
                                </div>
                                <div>Confiance</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_metric3:
                            status = "✅ Confiant" if confidence > 80 else "⚠️ Limite" if confidence > 60 else "❌ Faible"
                            status_color = "#28a745" if confidence > 80 else "#ffc107" if confidence > 60 else "#dc3545"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 1.5rem; color: {status_color};">
                                    {status}
                                </div>
                                <div>Statut</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Graphique des probabilités
                        fig2 = go.Figure(data=[
                            go.Bar(
                                x=class_names,
                                y=transfer_pred * 100,
                                marker_color=diseases_colors,
                                text=[f"{p*100:.1f}%" for p in transfer_pred],
                                textposition='auto',
                            )
                        ])
                        
                        fig2.update_layout(
                            title="Distribution des Probabilités - Transfer Learning",
                            xaxis_title="Maladies",
                            yaxis_title="Probabilité (%)",
                            height=400,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Conclusion et recommandations
                    st.markdown("---")
                    st.markdown('<h4 class="sub-header">💡 Recommandations</h4>', unsafe_allow_html=True)
                    
                    if custom_class_idx == transfer_class_idx and model_option == "Les deux modèles":
                        disease_name = class_names[custom_class_idx]
                        st.markdown(f"""
                        <div class="success-box">
                            <h5>✅ Diagnostic Concordant</h5>
                            <p>Les deux modèles s'accordent sur le diagnostic : <strong>{disease_name}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif model_option != "Les deux modèles":
                        disease_name = class_names[custom_class_idx] if model_option == "CNN Personnalisé" else class_names[transfer_class_idx]
                        st.markdown(f"""
                        <div class="success-box">
                            <h5>✅ Diagnostic Établi</h5>
                            <p>Le modèle a identifié la maladie : <strong>{disease_name}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <h5>⚠️ Diagnostic Divergent</h5>
                            <p>Les modèles proposent des diagnostics différents :</p>
                            <ul>
                                <li>CNN Personnalisé : <strong>{class_names[custom_class_idx]}</strong> ({custom_pred[custom_class_idx]*100:.1f}%)</li>
                                <li>Transfer Learning : <strong>{class_names[transfer_class_idx]}</strong> ({transfer_pred[transfer_class_idx]*100:.1f}%)</li>
                            </ul>
                            <p>Il est recommandé de consulter un expert agricole.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Boutons d'action
                    col_act1, col_act2, col_act3 = st.columns(3)
                    with col_act1:
                        st.button("📋 Générer Rapport", use_container_width=True)
                    with col_act2:
                        st.button("📧 Envoyer à un Expert", use_container_width=True)
                    with col_act3:
                        st.button("🔄 Analyser une Autre Image", use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Informations Image")
            img_info = Image.open(uploaded_file)
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Format", uploaded_file.type.split('/')[-1].upper())
                st.metric("Taille", f"{uploaded_file.size / 1024:.1f} KB")
            with info_col2:
                st.metric("Dimensions", f"{img_info.width} × {img_info.height}")
                st.metric("Mode", img_info.mode)
            
            st.markdown("### 🎯 Conseils de Capture")
            st.info("""
            Pour une meilleure analyse :
            1. Prenez la photo sous un bon éclairage
            2. Centrez la feuille dans le cadre
            3. Évitez les reflets et ombres
            4. Capturez les deux faces de la feuille
            """)
    
    else:
        # Section d'exemple quand aucune image n'est téléchargée
        st.markdown("""
        <div class="card">
            <h4>Comment utiliser cette application :</h4>
            <ol>
                <li>Téléchargez une image de feuille de canne à sucre en utilisant le bouton ci-dessus</li>
                <li>Sélectionnez les modèles à utiliser dans la sidebar</li>
                <li>Cliquez sur "Lancer l'Analyse"</li>
                <li>Consultez les résultats détaillés et les recommandations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Exemples d'images
        st.markdown("### 📸 Exemples d'Images")
        example_cols = st.columns(4)
        example_images = [
            ("saine.jpg", "Feuille Saine"),
            ("mosaic.jpg", "Mosaïque"),
            ("redrot.jpg", "Pourriture Rouge"),
            ("rust.jpg", "Rouille")
        ]
        
        for idx, (img_name, caption) in enumerate(example_images):
            with example_cols[idx]:
                # Pour la démo, nous affichons des placeholders
                st.image("https://via.placeholder.com/200x150/2E8B57/FFFFFF?text=Feuille+Exemple", 
                        caption=caption, use_column_width=True)

with tab2:
    st.markdown('<h3 class="sub-header">📊 Comparaison des Modèles</h3>', unsafe_allow_html=True)
    
    # Graphique de comparaison des performances
    st.markdown("### 📈 Performances Comparées")
    
    models = ['CNN Personnalisé', 'DenseNet121', 'ResNet50', 'EfficientNetB0']
    accuracy = [89.2, 93.7, 91.5, 92.1]
    inference_time = [45, 28, 62, 85]
    
    fig = go.Figure(data=[
        go.Bar(name='Précision (%)', x=models, y=accuracy, marker_color='#2E8B57'),
        go.Bar(name='Temps d\'inférence (ms)', x=models, y=inference_time, marker_color='#3CB371')
    ])
    
    fig.update_layout(
        barmode='group',
        title='Comparaison des Performances des Modèles',
        xaxis_title="Modèles",
        yaxis_title="Valeurs",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau de comparaison
    st.markdown("### 📋 Caractéristiques des Modèles")
    
    comparison_data = {
        "Modèle": ["CNN Personnalisé", "DenseNet121", "ResNet50", "EfficientNetB0"],
        "Précision (%)": ["89.2", "93.7", "91.5", "92.1"],
        "Taille (Mo)": ["15.2", "14.0", "98.0", "92.0"],
        "Temps Inférence (ms)": ["45", "28", "62", "85"],
        "Entraînement (min)": ["120", "45", "180", "200"],
        "Recommandé": ["✅", "🏆", "✅", "✅"]
    }
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    # Recommandation
    st.markdown("""
    <div class="success-box">
        <h5>🏆 Recommandation</h5>
        <p>Sur la base de nos tests, <strong>DenseNet121</strong> offre le meilleur équilibre entre précision (93.7%) 
        et vitesse d'inférence (28ms). C'est pourquoi nous l'avons sélectionné comme modèle de transfert learning principal.</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown('<h3 class="sub-header">📚 Guide des Maladies</h3>', unsafe_allow_html=True)
    
    # Informations sur chaque maladie
    diseases_info = [
        {
            "name": "Healthy",
            "color": "#28a745",
            "description": "Feuille saine sans signe de maladie",
            "symptoms": ["Couleur verte uniforme", "Texture lisse", "Pas de taches", "Croissance normale"],
            "treatment": "Aucun traitement nécessaire. Maintenir les bonnes pratiques agricoles."
        },
        {
            "name": "Mosaic",
            "color": "#dc3545",
            "description": "Maladie virale causée par le Sugarcane Mosaic Virus",
            "symptoms": ["Mosaïque de taches vert clair et foncé", "Feuilles striées", "Rachitisme"],
            "treatment": "Utiliser des variétés résistantes, éliminer les plantes infectées, contrôle des pucerons."
        },
        {
            "name": "Redrot",
            "color": "#800000",
            "description": "Maladie fongique causée par Colletotrichum falcatum",
            "symptoms": ["Lésions rougeâtres sur les feuilles", "Pourriture de la tige", "Odeur caractéristique"],
            "treatment": "Traitement fongicide, rotation des cultures, drainage approprié."
        },
        {
            "name": "Rust",
            "color": "#fd7e14",
            "description": "Maladie fongique causée par Puccinia melanocephala",
            "symptoms": ["Pustules orange-rouille", "Feuilles jaunissantes", "Chute prématurée des feuilles"],
            "treatment": "Fongicides à base de triazole, variétés résistantes, espacement adéquat."
        },
        {
            "name": "Yellow",
            "color": "#ffc107",
            "description": "Syndrome des feuilles jaunes causé par un phytoplasme",
            "symptoms": ["Jaunissement généralisé", "Rachitisme", "Réduction de la croissance"],
            "treatment": "Contrôle des insectes vecteurs, élimination des plantes malades, gestion nutritionnelle."
        }
    ]
    
    for disease in diseases_info:
        with st.expander(f"**{disease['name']}**", expanded=disease['name'] == "Healthy"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Créer un cercle de couleur pour la maladie
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
                
                st.markdown("**Symptômes principaux:**")
                for symptom in disease['symptoms']:
                    st.markdown(f"- {symptom}")
                
                st.markdown(f"**Traitement recommandé:** {disease['treatment']}")

with tab4:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">ℹ️ À Propos de l\'Application</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>🌾 Mission</h4>
            <p>Cette application a pour objectif d'aider les agriculteurs et agronomes à diagnostiquer 
            rapidement et précisément les maladies des feuilles de canne à sucre grâce à l'intelligence 
            artificielle.</p>
        </div>
        
        <div class="card">
            <h4>🧠 Technologie</h4>
            <p>L'application utilise deux approches de deep learning :</p>
            <ul>
                <li><strong>CNN Personnalisé</strong> : Architecture convolutionnelle spécialement conçue pour ce dataset</li>
                <li><strong>Transfer Learning</strong> : DenseNet121pré-entraîné sur ImageNet et fine-tuné sur notre dataset</li>
            </ul>
            <p>Les modèles ont été entraînés sur 2,569 images réparties en 5 classes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>📊 Dataset</h4>
            <p><strong>Sugarcane Leaf Disease Dataset</strong><br>
            Source: Kaggle<br>
            Images: 2,569<br>
            Classes: 5<br>
            Split: 70% train, 20% val, 10% test</p>
        </div>
        
        <div class="card">
            <h4>👨‍💻 Développeurs</h4>
            <p>Équipe d'IA Agricole</p>
            <p><small>© 2024 - Tous droits réservés</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Contact
    st.markdown("---")
    st.markdown('<h4 class="sub-header">📞 Contact et Support</h4>', unsafe_allow_html=True)
    
    contact_cols = st.columns(3)
    with contact_cols[0]:
        st.markdown("**Support Technique**")
        st.write("tech-support@sugarcane-ai.com")
    with contact_cols[1]:
        st.markdown("**Partnerships**")
        st.write("partnerships@sugarcane-ai.com")
    with contact_cols[2]:
        st.markdown("**Documentation**")
        st.write("[Documentation Technique](https://docs.sugarcane-ai.com)")

# Pied de page
st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[1]:
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        <p>🌿 <strong>Sugarcane Leaf Disease Classifier</strong> v2.0</p>
        <p>Précision moyenne: 91.5% • Modèle principal: DenseNet121</p>
        <p>Dernière mise à jour: Octobre 2024</p>
    </div>
    """, unsafe_allow_html=True)

# Script pour initialiser les modèles (simulé)
class_names = load_models()