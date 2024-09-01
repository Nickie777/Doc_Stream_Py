import cv2
from flask import Flask, request, jsonify
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier  # используем KNN вместо логистической регрессии
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from pyzbar.pyzbar import decode

app = Flask(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model_path = "models/model_0.pkl"
# Задаем целевой размер для всех изображений
TARGET_IMAGE_SIZE = (3508, 2479)


def preprocess_image(image_path):
    image = Image.open(image_path)

    # Приводим изображение к целевому размеру с сохранением пропорций
    image = image.convert('RGB')
    image = image.resize(TARGET_IMAGE_SIZE)

    return image

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
    #model_path = data.get('model_path')
    image_path = data.get('image_path')
    image_type = data.get('image_type')

    if not os.path.exists(model_path):
        return jsonify({"error": "Model path does not exist."}), 400

    if not os.path.exists(image_path):
        return jsonify({"error": "Image path does not exist."}), 400

    # Загружаем модель
    model = joblib.load(model_path)

    # Открываем и обрабатываем изображение
    image = preprocess_image(image_path)

    # Приводим изображение к целевому размеру
    #image = image.resize(TARGET_IMAGE_SIZE)

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

    # Возвращаем успешный ответ
    return jsonify({"message": "Model trained successfully."})

# Метод для предсказания типа изображения
@app.route('/predictModel', methods=['POST'])
def predict_model():
    data = request.json
    image_path = data.get('image_path')

    if not os.path.exists(image_path):
        return jsonify({"error": "Image path does not exist."}), 400

    # Загрузка модели
    if not os.path.exists(model_path):
        return jsonify({"error": "Model path does not exist."}), 400
    model = joblib.load(model_path)

    # Открываем и обрабатываем изображение
    image = preprocess_image(image_path)

    # Приводим изображение к целевому размеру
    #image = image.resize(TARGET_IMAGE_SIZE)

    image_array = np.array(image).flatten()

    # Предсказание
    prediction = model.predict([image_array])[0]

    return jsonify({"predicted_type": int(prediction)})

# Новый метод для чтения QR-кода из изображения
@app.route('/getQR', methods=['POST'])
def get_qr():
    data = request.json
    image_path = data.get('image_path')

    if not os.path.exists(image_path):
        return jsonify({"error": "Image path does not exist."}), 400

    # Открываем изображение и конвертируем его в формат OpenCV
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Failed to open image."}), 400

    # Декодируем QR-коды на изображении
    qr_codes = decode(image)

    if not qr_codes:
        return jsonify({"error": "No QR code found in the image."}), 400

    # Извлекаем данные из первого QR-кода
    qr_data = qr_codes[0].data.decode('utf-8')

    return jsonify({"qr_code": qr_data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
