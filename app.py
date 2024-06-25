from flask import Flask, request, render_template, jsonify
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Ruta completa del archivo CSV
csv_path = 'concrete_data.csv'

# Intentar cargar datos desde el CSV
try:
    data = pd.read_csv(csv_path)
    app.logger.debug('Datos cargados correctamente.')

    # Eliminar filas con valores NaN en las columnas seleccionadas
    data = data[['Cement', 'Blast Furnace Slag', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Age', 'Strength']].dropna()
    app.logger.debug('Datos limpios, NaN eliminados.')

except FileNotFoundError as e:
    app.logger.error(f'Error al cargar los datos: {str(e)}')
    data = None

# Definir límites para cada característica
feature_limits = {
    'Cement': (102, 540),
    'Blast Furnace Slag': (0, 359),
    'Water': (122, 247),
    'Superplasticizer': (0, 32.2),
    'Coarse Aggregate': (801, 1150),
    'Age': (1, 365)
}

# Función para validar los límites de las características
def validate_input(features):
    for feature, (min_val, max_val) in feature_limits.items():
        if feature in features:
            value = features[feature]
            if value < min_val or value > max_val:
                return False
    return True

# Preparar los datos para el modelo
if data is not None:
    X = data[['Cement', 'Blast Furnace Slag', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Age']]
    y = data['Strength']

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un modelo de regresión (ejemplo: RandomForestRegressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

# Ruta principal para la página web
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtener los datos del formulario
        cement = float(request.form['cement'])
        slag = float(request.form['slag'])
        water = float(request.form['water'])
        superplasticizer = float(request.form['superplasticizer'])
        coarse_agg = float(request.form['coarse_agg'])
        age = int(request.form['age'])

        # Validar los límites de los datos de entrada
        input_features = {
            'Cement': cement,
            'Blast Furnace Slag': slag,
            'Water': water,
            'Superplasticizer': superplasticizer,
            'Coarse Aggregate': coarse_agg,
            'Age': age
        }

        if not validate_input(input_features):
            error_message = 'Los valores ingresados están fuera de los límites permitidos.'
            return render_template('index.html', error=error_message)

        # Preparar los datos de entrada para la predicción
        input_data = np.array([[cement, slag, water, superplasticizer, coarse_agg, age]])

        # Realizar la predicción
        prediction = model.predict(input_data)
        predicted_strength = prediction[0]

        return render_template('index.html', prediction=predicted_strength)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
