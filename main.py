from flask import Flask, request, jsonify
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier  # используем KNN вместо логистической регрессии
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Метод для создания пустой модели
@app.route('/createModel', methods=['POST'])
def create_model():
    model = KNeighborsClassifier(n_neighbors=1)  # создаем пустую модель KNN
    model_path = os.path.join(MODELS_DIR, f"model_{len(os.listdir(MODELS_DIR))}.pkl")
    joblib.dump(model, model_path)
    return jsonify({"model_path": model_path})


# Метод для обучения модели
@app.route('/learnModel', methods=['POST'])
def learn_model():
    data = request.json
    model_path = data.get('model_path')
    image_path = data.get('image_path')
    image_type = data.get('image_type')

    if not os.path.exists(model_path):
        return jsonify({"error": "Model path does not exist."}), 400

    if not os.path.exists(image_path):
        return jsonify({"error": "Image path does not exist."}), 400

    # Загружаем модель
    model = joblib.load(model_path)

    # Открываем и обрабатываем изображение
    image = Image.open(image_path)
    image_array = np.array(image).flatten()

    # Проверяем, если у модели нет обучающих данных, создаем их
    if hasattr(model, 'classes_'):
        X = np.vstack([model._fit_X, image_array])
        y = np.concatenate([model._y, [image_type]])
    else:
        X = [image_array]
        y = [image_type]

    # Обучаем модель
    model.fit(X, y)

    # Сохраняем модель
    joblib.dump(model, model_path)

    return jsonify({"message": "Model trained successfully."})


# Метод для предсказания типа изображения
@app.route('/predictModel', methods=['POST'])
def predict_model():
    if 'file' not in request.files:
        return 'Файл не найден', 400
    file = request.files['file']

    if file.filename == '':
        return 'Имя файла не указано', 400

    if file and file.filename.endswith('.jpg'):
        # Построение безопасного имени файла
        filename = secure_filename(file.filename)

        # Построение полного пути к файлу
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Сохранение файла
        file.save(full_path)

        # Загрузка модели
        model_path = "models/model_0.pkl"
        model = joblib.load(model_path)

        # Открываем и обрабатываем изображение
        image = Image.open(full_path)
        image_array = np.array(image).flatten()

        # Предсказание
        prediction = model.predict([image_array])[0]

        return jsonify({"predicted_type": int(prediction)})
    else:
        return 'Неверный формат файла', 400 #  data = request.json
 #  model_path = data.get('model_path')
 #  image_path = data.get('image_path')

 #  if not os.path.exists(model_path):
 #      return jsonify({"error": "Model path does not exist."}), 400

 #  if not os.path.exists(image_path):
 #      return jsonify({"error": "Image path does not exist."}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
