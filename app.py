import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# load model
with open('model/rf_pkl', 'rb') as file:
  model = pickle.load(file)

features_encode = {
    "JENIS KELAMIN": {
        "P": 1,
        "L": 0
    },
    "FOTO TORAKS": {
        "Positif": 1,
        "Negatif": 0
    },
    "STATUS HIV": {
        "Negatif": 0,
        "Positif": 1
    },
    "RIWAYAT DIABETES": {
        "Tidak": 0,
        "Ya": 1
    },
    "HASIL TCM": {
        "Rif Sensitif": 1,
        "Negatif": 0,
        "Rif resisten": 2
    }
}

labels_encode = {
    1: "Paru",
    0: "Ekstra paru",
}

style_encode = {
    1: ["bg-yellow-600 text-yellow-50", "bg-yellow-100 text-yellow-700 hover:bg-yellow-200"],
    0: ["bg-red-600 text-red-50", "bg-red-100 text-red-700 hover:bg-red-200"],
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    jenis_kelamin = features_encode["JENIS KELAMIN"][request.form['jenis_kelamin']]
    foto_toraks = features_encode["FOTO TORAKS"][request.form['foto_toraks']]
    status_hiv = features_encode["STATUS HIV"][request.form['status_hiv']]
    riwayat_diabetes = features_encode["RIWAYAT DIABETES"][request.form['riwayat_diabetes']]
    hasil_tcm = features_encode["HASIL TCM"][request.form['hasil_tcm']]
    
    list_data = [jenis_kelamin, foto_toraks, status_hiv, riwayat_diabetes, hasil_tcm]

    prediction = model.predict([list_data])
    label_class = labels_encode[prediction[0]]
    style_class = style_encode[prediction[0]]
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)