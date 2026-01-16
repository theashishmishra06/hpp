from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    prediction = model.predict([[area, bhk, bath]])[0]

    # Graph
    data = pd.read_csv("house_data.csv")
    plt.figure()
    plt.scatter(data['area'], data['price'])
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Area vs Price")
    plt.savefig("static/graph.png")
    plt.close()

    return render_template("index.html", price=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
