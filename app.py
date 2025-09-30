# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64

app = Flask(__name__)

print("Cargando modelo...")
model = tf.keras.models.load_model("model.h5")
print("✅ Modelo cargado")

def preprocess_base64_image(data_url):
    """
    Devuelve un array listo para el modelo: shape (1, 784), valores en [0,1].
    Hace: decodifica base64 -> gris -> blur -> invertir -> umbral Otsu ->
    crop por contorno mayor -> resize 20x? manteniendo aspecto -> pad 28x28 ->
    centrar por centro de masa -> normalizar -> aplanar.
    """
    # decodificar
    encoded = data_url.split(",")[1]
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # normalmente 280x280 desde el canvas

    # suavizar ligermente para quitar ruido
    img = cv2.GaussianBlur(img, (3,3), 0)

    # invertir colores: (canvas negro en fondo blanco) -> MNIST blanco sobre negro
    img = cv2.bitwise_not(img)

    # binarizar con Otsu
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # encontrar contornos y usar el contorno de área mayor
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # Si no encuentra contornos, usar la imagen entera redimensionada
        roi = thresh
    else:
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        # añadir un pequeño padding para no cortar trazos
        pad = 8
        x0 = max(x-pad, 0)
        y0 = max(y-pad, 0)
        x1 = min(x+ w + pad, thresh.shape[1])
        y1 = min(y+ h + pad, thresh.shape[0])
        roi = thresh[y0:y1, x0:x1]

    # redimensionar manteniendo aspecto a 20x? y luego centrar en 28x28
    h, w = roi.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w * new_h) / h)))
    else:
        new_w = 20
        new_h = max(1, int(round((h * new_w) / w)))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # crear imagen 28x28 y pegar el resized centrado
    canvas = np.zeros((28,28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    # centrar por centro de masa (como hace MNIST)
    m = cv2.moments(canvas)
    if m["m00"] != 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        shiftx = int(round(14 - cx))
        shifty = int(round(14 - cy))
        M = np.float32([[1,0,shiftx],[0,1,shifty]])
        canvas = cv2.warpAffine(canvas, M, (28,28))
    # normalizar a [0,1]
    canvas = canvas.astype("float32") / 255.0

    # aplanar a 784 (porque tu modelo lo espera así)
    arr = canvas.reshape(1, 784)
    return arr

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("image", None)
        if not data:
            return jsonify({"error": "No se recibió imagen"}), 400

        x = preprocess_base64_image(data)
        preds = model.predict(x)
        digit = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        return jsonify({"prediction": digit, "confidence": conf})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render asigna el puerto
    app.run(host="0.0.0.0", port=port, debug=False)

