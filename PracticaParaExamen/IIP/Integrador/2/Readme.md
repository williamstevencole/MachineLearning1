📋 EXAMEN PARCIAL - PRÁCTICO

Catedrática: Ing. Nicole Rodríguez Asignatura: Aprendizaje de Máquina Tema: Reducción de Dimensionalidad y Clasificación (PCA + SVM)

Contexto del Problema: Una startup de tecnología móvil, TechMobile S.A., está diseñando su estrategia de precios para el próximo año. Cuentan con una base de datos de competidores que incluye especificaciones técnicas (RAM, Batería, Núcleos, etc.) y el rango de precio en el que se venden.

El Director de Producto tiene una hipótesis: "Las características técnicas definen el precio de forma tan clara que no necesitamos analizar las 20 especificaciones, sino solo las 3 dimensiones principales".

Su tarea es demostrar esta hipótesis visualmente y construir un modelo predictivo eficiente.

Dataset:

    Fuente: Mobile Price Classification (Kaggle)

    Archivo: train.csv

    Variable Objetivo: price_range (0, 1, 2, 3)

Instrucciones Técnicas:

Desarrolle un script en Python (mobile_analysis.py) modular y profesional que ejecute el siguiente flujo de trabajo secuencial:

    Carga y Argumentos:

        Implemente argparse para recibir la ruta del dataset desde la terminal.

        Realice una carga segura de los datos y separe las características (X) de la variable objetivo (y).

    Fase 1: Reducción y Visualización (Proceso Manual):

        Aplique StandardScaler a los datos originales para normalizar las escalas.

        Utilice PCA (Análisis de Componentes Principales) para reducir las 20 características originales a solo 3 Componentes.

        Calcule e imprima el porcentaje de Varianza Explicada Acumulada (¿cuánta información se conservó?).

        Genere un Gráfico de Dispersión 3D (Scatter Plot) utilizando los 3 componentes.

            Requisito: Los puntos deben estar coloreados según su price_range real. Esto servirá para validar visualmente si las clases son separables en este nuevo espacio reducido.

    Fase 2: Modelado con Vectores de Soporte (SVM):

        Utilice la matriz reducida obtenida en el paso anterior (el dataset de solo 3 columnas) como entrada para el modelo.

        Divida estos datos reducidos en conjuntos de Entrenamiento y Prueba (80/20) utilizando estratificación.

        Entrene un clasificador SVM (SVC).

        Utilice GridSearchCV para encontrar los mejores hiperparámetros (C, kernel, gamma).

    Evaluación:

        Reporte el Accuracy final del modelo.

        Grafique la Matriz de Confusión para analizar qué rangos de precio se confunden más entre sí.