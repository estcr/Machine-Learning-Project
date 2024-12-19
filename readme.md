# 🧠 Predicción de Accidentes Cerebrovasculares

Este proyecto tiene como objetivo predecir la probabilidad de que un paciente sufra un accidente cerebrovascular utilizando varios algoritmos de Machine Learning y técnicas de balanceo de clases.

## 📊 Descripción del Proyecto

El principal desafío de este proyecto fue el desbalanceo de clases, ya que los casos de accidentes cerebrovasculares (Stroke) eran significativamente menores en comparación con los casos negativos. Para abordar este problema, utilizamos técnicas de balanceo de clases como SMOTE y Oversampling.

## 🧹 Limpieza de Datos

La limpieza de datos es un paso crucial en cualquier proyecto de Machine Learning. En este proyecto, realizamos las siguientes tareas de limpieza de datos:

1. **Manejo de Valores Nulos**: Imputamos valores nulos en las columnas `bmi` y `smoking_status` utilizando la mediana y la moda, respectivamente.
2. **Codificación de Variables Categóricas**: Convertimos variables categóricas como `gender`, `ever_married`, `work_type`, `Residence_type` y `smoking_status` en variables dummy utilizando `pd.get_dummies()`.
3. **Normalización de Datos**: Normalizamos las características numéricas como `age`, `avg_glucose_level` y `bmi` para asegurar que todas las características estén en la misma escala.

## 🚀 Algoritmos Evaluados

Evaluamos varios algoritmos de Machine Learning, incluyendo:

- **Random Forest**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**
- **Regresión Logística**

## ⚖️ Técnicas de Balanceo de Clases

Para mejorar la predicción de la clase minoritaria (Stroke), utilizamos las siguientes técnicas de balanceo de clases:

- **SMOTE (Synthetic Minority Over-sampling Technique)**
- **Oversampling**

## 📈 Resultados Generales

A pesar de nuestros esfuerzos, los modelos no mostraron consistencia en la predicción de la clase minoritaria. La precisión y el recall para la clase minoritaria siguieron siendo bajos.

### Análisis de los Modelos

| Modelo                                | Accuracy | Precision | Recall | F1-Score |
|---------------------------------------|----------|-----------|--------|----------|
| KNN (Oversampling)                    | 0.916830 | 0.111111  | 0.10   | 0.105263 |
| Random Forest (Oversampling)          | 0.942270 | 0.000000  | 0.00   | 0.000000 |
| XGBoost (Oversampling)                | 0.926614 | 0.142857  | 0.10   | 0.117647 |
| Decision Tree (Oversampling)          | 0.911937 | 0.166667  | 0.20   | 0.181818 |
| Logistic Regression (Oversampling)    | 0.721135 | 0.126984  | 0.80   | 0.219178 |
| KNN (SMOTE)                           | 0.886497 | 0.107143  | 0.18   | 0.134328 |
| Random Forest (SMOTE)                 | 0.913894 | 0.120000  | 0.12   | 0.120000 |
| XGBoost (SMOTE)                       | 0.920744 | 0.170213  | 0.16   | 0.164948 |
| Decision Tree (SMOTE)                 | 0.877691 | 0.087912  | 0.16   | 0.113475 |
| Logistic Regression (SMOTE)           | 0.844423 | 0.165644  | 0.54   | 0.253521 |

## 🏆 Mejor Modelo

El modelo **Logistic Regression (SMOTE)** fue seleccionado como el mejor modelo debido a su balance entre precisión, recall y F1-Score. Aunque su precisión general es más baja, su alto recall y F1-Score lo hacen más efectivo para identificar la clase minoritaria (Stroke).

## 📦 Preparación del Modelo para Despliegue

El modelo entrenado se ha guardado en un archivo `logistic_regression_smote_model.pkl` utilizando la biblioteca `joblib`. Este archivo puede ser cargado y utilizado para realizar predicciones en un entorno de producción.

## 🔍 Conclusiones

Aunque los modelos evaluados tienen un buen rendimiento general, su capacidad para predecir la clase minoritaria (Stroke) fue limitada. Esto se refleja en la baja precisión, recall y F1-score para la clase 1 (Stroke) en todos los modelos evaluados.

## 📅 Próximos Pasos

Para mejorar el rendimiento en la clase minoritaria, se pueden considerar las siguientes estrategias:

- **Ajustar el umbral de decisión**: Modificar el umbral de probabilidad para clasificar una instancia como clase 1.
- **Técnicas de balanceo de clases adicionales**: Explorar técnicas como SMOTE combinado con Tomek Links o SMOTEENN para generar más ejemplos sintéticos de la clase minoritaria y eliminar ejemplos ruidosos.
- **Modelos más avanzados**: Probar modelos más avanzados como redes neuronales profundas o ensamblajes de modelos.

En resumen, aunque los modelos evaluados tienen un buen rendimiento general, es necesario abordar el desbalanceo de clases para mejorar la predicción de la clase minoritaria (Stroke). Debido a esta falta de consistencia en la predicción de la clase minoritaria, no recomendamos continuar con este proyecto ni llevarlo a producción en su estado actual. Sin embargo, continuaremos trabajando en este proyecto con fines educativos, explorando nuevas técnicas y enfoques para mejorar el rendimiento de los modelos.

## 📚 Recursos

- [Documentación de Scikit-Learn](https://scikit-learn.org/stable/documentation.html)
- [Documentación de Imbalanced-Learn](https://imbalanced-learn.org/stable/)
- [Documentación de Joblib](https://joblib.readthedocs.io/en/latest/)

## ✨ Contribuciones

¡Las contribuciones son bienvenidas! Si tienes alguna idea o mejora, no dudes en abrir un issue o enviar un pull request.

## 👨‍💼 Autor

- Esteban Cristos Muzzupappa - [LinkedIn](https://www·linkedin·com/in/esteban-cristos-muzzupappa/)
- Gerardo Jimenez - [LinkedIn](https://www·linkedin·com/in/gerardo-jimenez/) 


👋 **Gracias por visitar este proyecto!** Si tienes preguntas o sugerencias, no dudes en abrir un issue o contactarnos directamente.