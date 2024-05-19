from flask import Flask, request, jsonify,render_template
import joblib
# import psycopg2 
# import json

app = Flask(__name__)
model = joblib.load('iris_model.pkl')



# GET Data
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Body POST
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
    app.run(debug=True , port = 9000)