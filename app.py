from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

# Carregar os modelos
model_path_1 = 'models/Modelo_S_Data_Augmentation.keras'
model_path_2 = 'models/Modelo_S.keras'

model_1 = load_model(model_path_1)
model_2 = load_model(model_path_2)

models = {
    'Modelo_S_Data_Augmentation': model_1,
    'Modelo_S': model_2
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Ler a imagem diretamente da memória
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((32, 32))  # Ajustando para 32x32 pixels
        
        # Pré-processar a imagem
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        print("Imagem pré-processada:", img_array.shape, img_array[0, 0, 0, :])  # Depuração

        # Selecionar o modelo
        model_name = request.form.get('model')
        model = models.get(model_name)

        if model:
            # Classificar a imagem
            prediction = model.predict(img_array)
            print("Predição bruta:", prediction)  # Depuração

            # Aqui você pode ajustar a forma como o resultado é retornado
            result = np.argmax(prediction, axis=1)[0]
            return jsonify({'result': int(result)})

    return jsonify({'error': 'Failed to classify'})

if __name__ == '__main__':
    app.run(debug=True)
