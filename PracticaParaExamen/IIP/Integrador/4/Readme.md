🏠 EXAMEN FINAL INTEGRADOR - OPCIÓN B (Regresión)

Catedrática: Ing. Nicole Rodríguez Tema: Feature Engineering Geográfico + Ensambles de Regresión

Contexto del Negocio: Una agencia de bienes raíces en California quiere modernizar su sistema de valuación. Tienen datos históricos (habitaciones, antigüedad, ingresos de la zona, latitud, longitud). El problema es que la Ubicación (Latitud/Longitud) es difícil de entender para un modelo lineal simple. Tu misión es:

    Usar Clustering para agrupar las casas en "Vecindarios" basados en sus coordenadas.

    Usar esos "Vecindarios" como una nueva característica para entrenar un modelo de Ensamble (Stacking) que prediga el precio exacto de la vivienda.

Dataset:

    Fuente: California Housing Prices (Kaggle)

    Archivo: housing.csv

    Target: median_house_value

Instrucciones Técnicas (housing_master.py):

    Limpieza y EDA:

        Carga con argparse.

        Manejo de nulos (si los hay).

        Codifica la columna ocean_proximity (OneHotEncoder).

    Ingeniería de Características (El Truco Pro):

        Antes de dividir los datos, usa K-Means sobre las columnas latitude y longitude para crear, por ejemplo, 10 clusters (Vecindarios).

        Agrega la columna cluster_location al dataframe. ¡Ahora el modelo sabrá en qué "zona" está la casa!

    Modelado (Stacking Regressor):

        Queremos predecir el precio numérico.

        Implementa un StackingRegressor:

            Nivel 1: SVR (Kernel RBF) y Random Forest Regressor.

            Nivel 2 (Final): Linear Regression (o Ridge).

    Evaluación:

        Calcula el RMSE (Raíz del Error Cuadrático Medio) y R2 Score.

        Grafica: Valores Reales vs. Valores Predichos (Un scatter plot donde una línea diagonal perfecta significaría predicción perfecta).