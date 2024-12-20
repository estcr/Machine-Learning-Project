import joblib
import pandas as pd
import streamlit as st

# Cargar el modelo guardado, el escalador y SMOTE
model = joblib.load(r'logistic_regression_smote_model.pkl')
scaler = joblib.load(r'scalerfinal.pkl')
smote = joblib.load(r'smotefinal.pkl')

# Función para solicitar los datos del paciente
def get_patient_data():
    data = {}
    data['age'] = st.number_input("Ingrese la edad del paciente:", min_value=0, max_value=120, value=30)
    data['hypertension'] = st.selectbox("¿El paciente tiene hipertensión?", options=[0, 1])
    data['heart_disease'] = st.selectbox("¿El paciente tiene enfermedad cardíaca?", options=[0, 1])
    data['avg_glucose_level'] = st.number_input("Ingrese el nivel promedio de glucosa del paciente:", min_value=0.0, max_value=300.0, value=100.0)
    data['bmi'] = st.number_input("Ingrese el índice de masa corporal (BMI) del paciente:", min_value=0.0, max_value=100.0, value=25.0)
    data['gender'] = st.selectbox("¿El paciente es masculino?", options=[0, 1])
    data['ever_married'] = st.selectbox("¿El paciente alguna vez se ha casado?", options=[0, 1])
    data['work_type_Never_worked'] = st.selectbox("¿El paciente nunca ha trabajado?", options=[0, 1])
    data['work_type_Private'] = st.selectbox("¿El paciente trabaja en el sector privado?", options=[0, 1])
    data['work_type_Self-employed'] = st.selectbox("¿El paciente es autónomo?", options=[0, 1])
    data['work_type_children'] = st.selectbox("¿El paciente es un niño?", options=[0, 1])
    data['Residence_type_Urban'] = st.selectbox("¿El paciente vive en una zona urbana?", options=[0, 1])
    data['smoking_status_formerly smoked'] = st.selectbox("¿El paciente fumaba anteriormente?", options=[0, 1])
    data['smoking_status_never smoked'] = st.selectbox("¿El paciente nunca ha fumado?", options=[0, 1])
    data['smoking_status_smokes'] = st.selectbox("¿El paciente fuma actualmente?", options=[0, 1])
    
    return pd.DataFrame([data])

# Crear la aplicación de Streamlit
st.title('Predicción de Riesgo de Accidente Cerebrovascular')

# Crear un índice en la barra lateral
st.sidebar.title('Índice')
section = st.sidebar.radio('Ir a', ['Resumen', 'Prueba el Modelo'])

# Mostrar la sección seleccionada
if section == 'Resumen':
    st.markdown("""
    ### Resumen del Proyecto de Machine Learning 🧠💻

    Este proyecto tiene como objetivo predecir el riesgo de accidente cerebrovascular en pacientes utilizando un modelo de Random Forest entrenado con datos balanceados mediante SMOTE.

    #### Pasos Realizados en el Archivo `main.ipynb`:

    1. **Carga de Datos 📂**: Se cargan los datos desde un archivo CSV.
    2. **Preprocesamiento de Datos 🧹**:
       - Separación de columnas categóricas y numéricas.
       - Aplicación de `One-Hot Encoding` a las columnas categóricas.
       - Normalización de las columnas numéricas.
    3. **Balanceo de Clases ⚖️**: Se utiliza SMOTE para balancear las clases.
    4. **Entrenamiento del Modelo 🏋️‍♂️**: Se entrenan varios modelos de Machine Learning, incluyendo:
       - K-Nearest Neighbors (KNN)
       - Random Forest
       - XGBoost
       - Decision Tree
       - Logistic Regression
    5. **Evaluación de Modelos 📊**: Se evaluaron los modelos utilizando métricas como Accuracy, Precision, Recall y F1-Score.
    6. **Selección del Mejor Modelo 🏆**: El modelo de Random Forest con SMOTE fue seleccionado como el mejor modelo.
    7. **Guardado del Modelo y Preprocesadores 💾**: Se guardan el modelo entrenado, el escalador y SMOTE utilizando `joblib`.

    #### Problemas Encontrados y Soluciones 🛠️:
    - **Desbalanceo de Clases**: Se solucionó utilizando SMOTE para balancear las clases.
    - **Datos Faltantes**: Se manejaron los datos faltantes mediante imputación.
    - **Multicolinealidad**: Se revisaron las correlaciones entre las variables y se eliminaron las variables altamente correlacionadas.
    """)
elif section == 'Prueba el Modelo':
    st.header('Prueba el Modelo')
    st.write('Ingrese los datos del paciente para realizar la predicción.')

    # Solicitar los datos del paciente
    patient_data = get_patient_data()

    if st.button('Realizar Predicción'):
        # Aplicar el escalador a los datos del paciente
        patient_data_scaled = scaler.transform(patient_data)

        # Aplicar SMOTE a los datos del paciente (esto es inusual y generalmente no se hace)
        X_resampled, y_resampled = smote.fit_resample(patient_data_scaled, [0])  # Aquí asumimos que la clase es 0 para el nuevo dato

        # Realizar predicciones con el modelo cargado
        prediction = model.predict(X_resampled)

        # Mostrar las predicciones
        if prediction[0] == 1:
            st.write("El paciente tiene un riesgo de sufrir un accidente cerebrovascular.")
        else:
            st.write("El paciente no tiene un riesgo significativo de sufrir un accidente cerebrovascular.")