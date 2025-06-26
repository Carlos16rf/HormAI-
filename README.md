# HormAI – Predicción Personalizada en Reproducción Asistida mediante IA

HormAI es un sistema basado en inteligencia artificial diseñado para asistir al personal clínico en la toma de decisiones durante tratamientos de fecundación in vitro (FIV). A través de modelos de machine learning, predice de forma secuencial variables clave como:

- Número de ovocitos aspirados  
- Número de ovocitos maduros (MII)  
- Calidad ovocitaria grupal  
- Probabilidad de implantación embrionaria  

El proyecto incluye un Jupyter Notebook con el desarrollo completo y una aplicación web funcional (frontend + backend) que permite realizar predicciones desde una interfaz sencilla e intuitiva.

---

## 📁 Estructura del proyecto

HormAI/
├── README.md
├── requirements.txt
├── HormAI.ipynb # Jupyter Notebook con desarrollo y modelos
├── backend/
│ ├── app.py # Backend en Flask
│ └── modelos/
│ ├── modelo_xgb_residuos.pkl
│ ├── modelo_CatBoostClassifier.pkl
│ └── model_calidad_grupo.pkl
├── frontend/
│ └── index.html # Interfaz web (HTML)
