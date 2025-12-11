🎓 EXAMEN FINAL INTEGRADOR (Simulacro)

Catedrática: Ing. Nicole Rodríguez Tema: Pipeline Completo de Machine Learning

Contexto del Negocio: Eres el Lead Data Scientist de una nueva compañía de teléfonos móviles. Quieren lanzar una nueva línea de celulares y necesitan un sistema para estimar el rango de precio (0=Barato, 1=Medio, 2=Alto, 3=Muy Alto) basándose en sus especificaciones técnicas (RAM, Batería, Megapixeles, Peso, etc.).

Además, el CEO tiene una duda: "¿Nuestros teléfonos se agrupan naturalmente por sus características o el precio es arbitrario?". Debes usar técnicas no supervisadas para responderle y técnicas supervisadas para crear el predictor de precios.

Dataset: Descarga el dataset "Mobile Price Classification": 🔗 Link Kaggle: Mobile Price Classification (Usa el archivo train.csv)

Instrucciones del Reto (Script final_exam.py):

    Fase 1: Análisis Visual Avanzado (PCA + Clustering)

        El dataset tiene 20 features. Imposible de ver.

        Reduce la dimensionalidad a 3 Componentes usando PCA.

        Aplica K-Means (con K=4, ya que hay 4 rangos de precio) sobre estos componentes para ver si la IA encuentra los grupos sola.

        Grafica en 3D: Los puntos coloreados por el Cluster que encontró K-Means. (Esto responde al CEO si hay grupos naturales).

    Fase 2: El Torneo de Modelos (Supervisado)

        Ahora usa las etiquetas reales (price_range).

        Divide en Train/Test (Estratificado).

        Modelo A (El Matemático): Pipeline con StandardScaler + SVC. Usa GridSearchCV para optimizar Kernel y C.

        Modelo B (El Fuerza Bruta): RandomForestClassifier.

        Evaluación: Compara ambos modelos con classification_report y accuracy. Decide el ganador.
