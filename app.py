import __main__
from flask import Flask, request, jsonify, render_template
import joblib
import os
class OnlineLearningModel:
    pass

__main__.OnlineLearningModel = OnlineLearningModel
app = Flask(__name__)
model = joblib.load('iris_model.pkl')

@app.route("/")
def start():
    return "The ML Server is Running"

# @app.route("/index")
# def index():
#     return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sepal_length = data['Sepal length']
        sepal_width = data['Sepal width']
        petal_length = data['Petal length']
        petal_width = data['Petal width']
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        return jsonify({'predicted_species': prediction})
    except Exception as e:
        return jsonify({'error': 'Prediction error: ' + str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 9000))
#     app.run(host='0.0.0.0', port=port)
