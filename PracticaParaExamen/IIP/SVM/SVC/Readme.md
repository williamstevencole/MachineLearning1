📋 EXAMEN PARCIAL - PRÁCTICO (Simulacro)

Catedrática: Ing. Nicole Rodríguez Tema: Support Vector Classification (SVC)

Contexto del Negocio: El Instituto Nacional de Diabetes necesita un modelo para predecir si un paciente tiene diabetes o no, basándose en medidas diagnósticas (embarazos, glucosa, presión, insulina, BMI, etc.). Se requiere un modelo de Alta Precisión utilizando Máquinas de Vectores de Soporte.

Dataset: Descarga el dataset "Pima Indians Diabetes": 🔗 Link Kaggle: Pima Indians Diabetes Database (Archivo: diabetes.csv)

Instrucciones Técnicas:

Desarrolla el script svc_exam.py modular:

    Carga y EDA:

        Carga el CSV.

        Muestra la correlación de las variables con el target Outcome (usando un mapa de calor o lista ordenada).

        Grafica la distribución de la variable Outcome (¿Está balanceado el dataset?).

    Preprocesamiento (Pipeline):

        Define X (features) e y (target: Outcome).

        Divide en Train/Test (80/20) con stratify=y (importante en salud).

        Crea un Pipeline con:

            StandardScaler (Crucial).

            SVC() (El clasificador vacío).

    Modelado (GridSearchCV):

        Configura un diccionario de hiperparámetros para probar:

            C: [0.1, 1, 10, 100] (Qué tanto penalizamos errores).

            kernel: ['linear', 'rbf'] (El truco matemático).

            gamma: ['scale', 'auto'] (Solo afecta al rbf).

        Ejecuta GridSearchCV con 5 folds (cv=5).

    Evaluación:

        Imprime los mejores parámetros encontrados.

        Evalúa el mejor modelo en el Test Set.

        Muestra: Accuracy, Classification Report y la Confusion Matrix (graficada con Seaborn).