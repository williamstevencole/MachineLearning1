📋 EXAMEN FINAL - PRÁCTICO (Ensambles)

Catedrática: Ing. Nicole Rodríguez Tema: Métodos de Ensamble (Bagging, Boosting y Stacking)

Contexto del Negocio: Una empresa de Telecomunicaciones está perdiendo clientes ("Churn"). Quieren predecir quién se va a ir para ofrecerle una promoción antes de que sea tarde. Han probado un Árbol de Decisión simple y falló. Ahora te piden usar Métodos de Ensamble para mejorar la precisión.

Dataset: Descarga el famoso "Telco Customer Churn": 🔗 Link Kaggle: Telco Customer Churn (Archivo: WA*Fn-UseC*-Telco-Customer-Churn.csv)

Instrucciones Técnicas (Nivel Experto):

Desarrolla el script ensemble_battle.py:

    Limpieza y EDA Rápido:

        Carga datos con argparse.

        OJO: La columna TotalCharges tiene espacios vacíos que parecen texto. Conviértela a numérico (pd.to_numeric(..., errors='coerce')) y llena los nulos.

        Target: Churn (Yes/No). Conviértelo a 1/0.

        Elimina customerID.

    Preprocessing (Pipeline Robusto):

        Tienes columnas numéricas y categóricas.

        Usa ColumnTransformer:

            Numéricas → StandardScaler.

            Categóricas → OneHotEncoder.

    La Batalla de Modelos (Entrenamiento): Entrena y compara estos 3 modelos usando el mismo X_train:

        Competidor A (Bagging): RandomForestClassifier

            n_estimators=100, random_state=42.

        Competidor B (Boosting): GradientBoostingClassifier

            n_estimators=100, learning_rate=0.1.

        Competidor C (Stacking): StackingClassifier

            Estimadores base: Un RandomForest y un SVC.

            Estimador final: LogisticRegression.

    Evaluación Final:

        Crea un bucle que recorra los 3 modelos.

        Para cada uno imprime: Accuracy y F1-Score.

        ¿Cuál ganó? (Normalmente Boosting o Stacking ganan por poco).
