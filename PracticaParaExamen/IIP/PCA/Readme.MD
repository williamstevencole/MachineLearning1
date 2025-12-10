📋 EXAMEN PARCIAL - PRÁCTICO (PCA)

Catedrática: Ing. Nicole Rodríguez Tema: Reducción de Dimensionalidad (PCA)

Contexto del Negocio: Un laboratorio médico tiene un dataset de Cáncer de Mama con 30 características numéricas extraídas de imágenes digitales (radio, textura, perímetro, área, suavidad, etc.). Los médicos no pueden visualizar gráficos de 30 dimensiones. Te contratan para reducir esta complejidad a 2 Dimensiones para poder ver en una pantalla plana si los tumores Malignos y Benignos se separan visualmente.

Dataset: Descarga el dataset Breast Cancer Wisconsin (Diagnostic): 🔗 Link Kaggle: Breast Cancer Wisconsin (Archivo: data.csv)

Instrucciones Técnicas:

Desarrolla el script pca_exam.py modular:

    Limpieza y EDA:

        Carga los datos con argparse.

        Elimina la columna id y Unnamed: 32 (si existen).

        La columna diagnosis es tu target (M/B). Sepárala en y. El resto es X.

        Muestra un Heatmap de Correlación de las 30 variables (verás que es un caos y hay mucha redundancia, justificando el uso de PCA).

    Pipeline de PCA:

        Crea un Pipeline con:

            StandardScaler (Obligatorio).

            PCA(n_components=2).

        Aplica el pipeline a X.

    Análisis de Varianza:

        Accede al objeto PCA dentro del pipeline.

        Imprime: "Varianza explicada por cada componente".

        Imprime: "Varianza total acumulada" (La suma). ¿Es suficiente información? (Usualmente > 80% es bueno, > 60% es aceptable para visualización).

    Visualización 2D:

        Crea un DataFrame con los resultados del PCA (Componente 1, Componente 2).

        Agrega la columna diagnosis a ese DataFrame.

        Genera un Scatter Plot:

            Eje X: Componente 1.

            Eje Y: Componente 2.

            Color (hue): Diagnosis (M vs B).

        Objetivo: Deberías ver dos grupos claramente separados, demostrando que PCA funcionó.
