#!/bin/bash

# Crear entorno virtual con Python
python -m venv env

# Activar el entorno virtual (para el resto del script)
source env/bin/activate

# Instalar dependencias desde requirements.txt
pip install -r requirements.txt

# Ejecutar Streamlit
streamlit run interfaz.py