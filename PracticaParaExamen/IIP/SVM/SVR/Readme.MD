📋 EXAMEN PARCIAL - PRÁCTICO (SVR)

Catedrática: Ing. Nicole Rodríguez Tema: Regresión con Vectores de Soporte (SVR)

Contexto del Negocio: Una compañía de seguros de salud te ha contratado. Quieren un modelo capaz de predecir cuánto cobrarle a un cliente (charges) basándose en su edad, género, índice de masa corporal (bmi), número de hijos, si fuma y su región. El modelo debe ser muy preciso porque un error grande significa perder dinero.

Dataset: Descarga el dataset "Medical Cost Personal Datasets": 🔗 Link Kaggle: Medical Cost Personal Datasets (Archivo: insurance.csv)

Instrucciones Técnicas:

Desarrolla el script svr_exam.py modular:

    EDA (Adaptado a Regresión):

        Muestra .info() y .describe().

        Grafica la distribución de la variable objetivo charges (Histograma).

        Grafica un Scatter Plot: Eje X=bmi, Eje Y=charges, Color=smoker (Esto te dará pistas visuales importantes).

    Preprocesamiento (Pipeline Avanzado):

        Define X (features) e y (target: charges).

        Identifica:

            Columnas Numéricas: age, bmi, children.

            Columnas Categóricas: sex, smoker, region.

        Construye el ColumnTransformer:

            Numéricas → StandardScaler (¡SVR lo exige!).

            Categóricas → OneHotEncoder.

    Modelado (GridSearchCV):

        Usa SVR().

        Grid de Parámetros:

            C: [1, 10, 100, 1000] (En regresión los valores de C suelen ser más altos).

            kernel: ['linear', 'rbf'].

            gamma: ['scale', 'auto'].

        Scoring: Usa r2 (R cuadrado).

    Evaluación:

        Imprime los mejores hiperparámetros.

        Calcula e imprime en el Test Set:

            MAE (Mean Absolute Error).

            RMSE (Root Mean Squared Error).

            R2 Score (¿Qué tan bien se ajusta el modelo? 1.0 es perfecto).
