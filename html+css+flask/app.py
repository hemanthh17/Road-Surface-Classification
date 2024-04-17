from flask import Flask, request, render_template
from image_process import predict_class
import io

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict_surface', methods=['POST'])
def predict_road_surface():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename=='':
        return 'No selected file',400
    if file:
        img_bytes=file.read()
        output=predict_class(img_bytes)
        output=predict_class(img_bytes) 
        return render_template('results.html',result=output)
    else:
        return 'Invalid file',400

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
