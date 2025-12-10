📋 EXAMEN PARCIAL - PRÁCTICO (Simulacro)

Catedrática: Ing. Nicole Rodríguez Tema: Clustering (K-Means & Segmentación)

Contexto del Negocio: Eres el Data Scientist principal de un centro comercial. El departamento de marketing tiene una base de datos de clientes con su "Ingreso Anual" y un "Puntaje de Gastos" (asignado por el centro comercial basado en comportamiento). Ellos no saben qué grupos de clientes existen. Tu misión es utilizar un algoritmo no supervisado para encontrar los segmentos de mercado y explicarle a marketing en qué se diferencian.

Dataset: Descarga el archivo Mall_Customers.csv desde Kaggle: 🔗 Link: Mall Customer Segmentation Data

Instrucciones Técnicas (Requisitos de Código):

Desarrolla un script en Python (clustering_exam.py) que cumpla con:

    Estructura Modular:

        El código debe estar organizado en funciones (def).

        Debe tener un bloque if __main__ == "__main__":.

        Debe usar argparse para recibir la ruta del archivo .csv.

    EDA (Análisis Exploratorio) - OBLIGATORIO:

        Muestra las primeras 5 filas y el .describe().

        Genera gráficos de distribución (histogramas) para Annual Income y Spending Score.

    Preprocesamiento (Pipelines):

        Construye un Pipeline de Scikit-Learn.

        Crucial: Dado que K-Means usa distancias euclidianas, debes incluir un StandardScaler dentro del pipeline o preprocesamiento.

    Modelado (Elbow Method & K-Means):

        Implementa una función para graficar el Método del Codo (Inercia) probando de 1 a 10 clusters.

        Entrena el modelo final con el K que consideres óptimo (usualmente 5 para este dataset).

    Evaluación y Visualización:

        Calcula el Silhouette Score para validar matemáticamente la separación.

        Genera un Scatter Plot:

            Eje X: Ingresos.

            Eje Y: Score de Gastos.

            Color: Cluster asignado.

            Extra: Dibuja los centroides (puntos centrales) en el gráfico.
