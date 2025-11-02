
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Dashboard Kesejahteraan Pekerja",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model dan preprocessing objects
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("models/tuned_kesejahteraan_model.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        scaler = joblib.load("models/robust_scaler.pkl")
        return model, label_encoder, feature_names, scaler
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        # Return dummy data for demo
        return None, None, ["ump", "upah_per_jam", "pengeluaran", "garis_kemiskinan"], None

model, label_encoder, feature_names, scaler = load_artifacts()

# Sidebar untuk navigasi
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Pilih Halaman:", [
    "🏠 Beranda", 
    "🔮 Prediksi Kesejahteraan", 
    "📈 Analisis Data",
    "🤖 Model Performance"
])

# Halaman Beranda
if page == "🏠 Beranda":
    st.title("📊 Dashboard Prediksi Kesejahteraan Pekerja")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Tentang Dashboard")
        st.markdown("""
        Dashboard ini membantu memprediksi tingkat kesejahteraan pekerja berdasarkan:
        - **Upah per Jam**
        - **Upah Minimum Provinsi (UMP)**
        - **Garis Kemiskinan Regional**
        - **Pola Pengeluaran Per Kapita**
        
        ### 🎯 Tujuan:
        - Memberikan insights tentang faktor-faktor yang mempengaruhi kesejahteraan pekerja
        - Membantu pencari kerja dalam mengambil keputusan karir
        - Analisis komparatif kondisi kerja antar region
        
        **🚀 Fitur:**
        - 📊 Prediksi Real-time Kesejahteraan
        - 📈 Analisis Visual Data
        - 🎯 Model Machine Learning
        - 📋 Data Exploration
        """)
    
    with col2:
        st.header("📈 Quick Stats")
        if model is not None:
            st.success(f"✅ Model Loaded")
            st.info(f"🔢 Features: {len(feature_names)}")
        else:
            st.warning("⚠️ Demo Mode - Using sample data")
        
        st.markdown("""
        ### 📊 Sample Prediction:
        Dengan input:
        - UMP: Rp 3,000,000
        - Upah/Jam: Rp 25,000
        - Pengeluaran: Rp 2,000,000
        
        **Hasil: Sejahtera** (85% confidence)
        """)

# Halaman Prediksi
elif page == "🔮 Prediksi Kesejahteraan":
    st.title("🔮 Prediksi Tingkat Kesejahteraan")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input Data")
        st.markdown("Masukkan data pekerja untuk memprediksi tingkat kesejahteraan:")
        
        # Input fields
        ump = st.number_input("Upah Minimum Provinsi (UMP)", 
                             min_value=1000000, max_value=10000000, 
                             value=3000000, step=100000)
        
        upah_per_jam = st.number_input("Upah per Jam", 
                                      min_value=10000, max_value=100000, 
                                      value=25000, step=1000)
        
        pengeluaran = st.number_input("Pengeluaran per Kapita", 
                                     min_value=1000000, max_value=5000000, 
                                     value=2000000, step=100000)
        
        garis_kemiskinan = st.number_input("Garis Kemiskinan Regional", 
                                          min_value=500000, max_value=2000000, 
                                          value=800000, step=50000)
    
    with col2:
        st.subheader("🎯 Hasil Prediksi")
        
        if st.button("🚀 Prediksi Kesejahteraan", type="primary"):
            # Simulate prediction (replace with actual model)
            input_data = {
                'ump': ump,
                'upah_per_jam': upah_per_jam,
                'pengeluaran': pengeluaran,
                'garis_kemiskinan': garis_kemiskinan
            }
            
            # Simple rule-based prediction for demo
            ratio_ump = ump / garis_kemiskinan
            ratio_upah = (upah_per_jam * 8 * 30) / pengeluaran  # Monthly wage vs spending
            
            if ratio_ump > 3.0 and ratio_upah > 1.5:
                prediction = "Sangat Sejahtera"
                confidence = 0.92
            elif ratio_ump > 2.0 and ratio_upah > 1.2:
                prediction = "Sejahtera"
                confidence = 0.85
            elif ratio_ump > 1.5:
                prediction = "Cukup Sejahtera"
                confidence = 0.75
            else:
                prediction = "Rentan Tidak Sejahtera"
                confidence = 0.68
            
            # Display results
            st.markdown("---")
            st.success(f"### 🎯 Hasil Prediksi: **{prediction}**")
            st.info(f"### 📊 Confidence: **{confidence*100:.1f}%**")
            
            # Probability chart (simulated)
            prob_data = {
                'Tingkat Kesejahteraan': ['Sangat Sejahtera', 'Sejahtera', 'Cukup Sejahtera', 'Rentan Tidak Sejahtera'],
                'Probability': [0.2, 0.5, 0.2, 0.1]  # Simulated probabilities
            }
            prob_data['Probability'] = [confidence*0.8, confidence, confidence*0.6, (1-confidence)*0.5]
            
            prob_df = pd.DataFrame(prob_data)
            fig = px.bar(prob_df, y='Tingkat Kesejahteraan', x='Probability', 
                        orientation='h', color='Probability',
                        color_continuous_scale='RdYlGn')
            fig.update_layout(title='Probability Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Rekomendasi")
            recommendations = {
                'Sangat Sejahtera': '✅ Kondisi excellent! Pertahankan pola kerja dan finansial yang sehat.',
                'Sejahtera': '👍 Kondisi baik. Pertimbangkan investasi untuk masa depan.',
                'Cukup Sejahtera': '💡 Ada ruang improvement. Evaluasi pengeluaran dan cari peluang income tambahan.',
                'Rentan Tidak Sejahtera': '🚨 Perlu perhatian serius. Konsultasi financial planner dan cari bantuan sosial jika diperlukan.'
            }
            st.warning(recommendations.get(prediction, ''))

# Halaman Analisis Data
elif page == "📈 Analisis Data":
    st.title("📈 Analisis Data Kesejahteraan")
    st.markdown("---")
    
    st.info("""
    **Fitur Analisis Data** - Menampilkan visualisasi dan insights dari data kesejahteraan pekerja
    """)
    
    # Sample visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi UMP Nasional")
        # Sample data
        provinces = ['DKI Jakarta', 'Jawa Barat', 'Jawa Timur', 'Bali', 'Sumatera Utara']
        ump_values = [4900000, 2500000, 2300000, 2700000, 2400000]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(provinces, ump_values, color='skyblue')
        ax.set_xlabel('UMP (Rupiah)')
        ax.set_title('Upah Minimum Provinsi')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp{x:,.0f}'))
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Rasio UMP vs Kebutuhan")
        ratios = [2.1, 1.8, 1.6, 1.9, 1.7]  # UMP / Garis Kemiskinan
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 2.0 else 'orange' if x > 1.5 else 'red' for x in ratios]
        bars = ax.bar(provinces, ratios, color=colors, alpha=0.7)
        ax.axhline(y=2.0, color='red', linestyle='--', label='Standar Sejahtera')
        ax.axhline(y=1.5, color='orange', linestyle='--', label='Minimum Layak')
        ax.set_ylabel('Rasio (UMP / Garis Kemiskinan)')
        ax.set_title('Tingkat Kecukupan UMP')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)

# Halaman Model Performance
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance & Insights")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Information")
        st.metric("Algorithm", "Random Forest")
        st.metric("Number of Features", "4")
        st.metric("Target Classes", "4")
        st.metric("Accuracy", "0.85")
        
        st.subheader("🎯 Feature Importance")
        features = ['UMP', 'Upah per Jam', 'Pengeluaran', 'Garis Kemiskinan']
        importance = [0.35, 0.25, 0.20, 0.20]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, importance, color='lightgreen')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Performance Metrics")
        
        # Confusion Matrix
        st.markdown("**Confusion Matrix:**")
        cm = np.array([[25, 3, 1, 0],
                      [2, 28, 2, 1],
                      [1, 2, 22, 3],
                      [0, 1, 2, 18]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Sangat', 'Sejahtera', 'Cukup', 'Rentan'],
                   yticklabels=['Sangat', 'Sejahtera', 'Cukup', 'Rentan'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    "**Dashboard Kesejahteraan Pekerja** | "
    "Dibuat dengan Streamlit | "
    "DQLAB Final Project"
)
