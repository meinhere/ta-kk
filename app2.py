# Data Processing
import pandas as pd
import numpy as np
import pickle

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder

# Visualization
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Home", "Analysis", "Preprocessing", "Modeling","Testing"])

    if page == "Home":
        show_home()
    elif page == "Analysis":
        show_analysis()
    elif page == "Preprocessing":
        show_preprocessing()
    elif page == "Modeling":
        show_model()
    elif page == "Testing":
        show_testing()

def show_home():
    st.title("Pengolahan Data TBC dengan Metode Random Forest")

    # Explain what is Random Forest
    st.header("Apa itu Random Forest?")
    st.write("Random Forest adalah salah satu algoritma yang terbaik dalam machine learning. Random Forest adalah kumpulan dari decision tree atau pohon keputusan. Algoritma ini merupakan kombinasi masing-masing tree dari decision tree yang kemudian digabungkan menjadi satu model. Biasanya, Random Forest dipakai untuk masalah regresi dan klasifikasi dengan kumpulan data yang berukuran besar.")

    # Explain the purpose of this website
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memberikan pemahaman mengenai tahapan proses pengolahan data TBC dengan menggunakan metode Random Forest.")

    # Explain the data
    st.header("Data")
    st.write("Data yang digunakan diambil dari dosen mata kuliah yang berisi informasi terkait Tuberculosis yang ada di Kabupaten Bangkalan.")

    # Explain the process of Random Forest
    st.header("Tahapan Proses Random Forest")
    st.write("1. **Data Preparation**")
    st.write("2. **Preprocessing Data**")
    st.write("3. **Modeling**")
    st.write("4. **Evaluation**")
    st.write("5. **Deploy App**")

def show_analysis():
    st.title("Analysis Data")

    # --------------- Load Data -----------------
    df = pd.DataFrame(pd.read_excel("dataset/tuberculosis.xlsx"))

    st.write("Data yang tersedia:")
    st.write(df.head())

    # --------------- Missing value -----------------
    st.write("### Mengecek Missing Value")

    umur = len(df[df['UMUR'] == '0'])
    tcm = len(df[df['HASIL TCM'] == 'Tidak dilakukan'])
    toraks = len(df[df['FOTO TORAKS'] == 'Tidak dilakukan'])
    hiv = len(df[df['STATUS HIV'] == 'Tidak diketahui'])
    diabet = len(df[df['RIWAYAT DIABETES'] == 'Tidak diketahui'])

    st.write("\nData dengan 'UMUR' == '0':")
    st.write(umur)

    st.write("\nData dengan 'HASIL TCM' == 'Tidak dilakukan':")
    st.write(tcm)

    st.write("\nData dengan 'FOTO TORAKS' == 'Tidak dilakukan':")
    st.write(toraks)

    st.write("\nData dengan 'STATUS HIV' == 'Tidak diketahui':")
    st.write(hiv)

    st.write("\nData dengan 'RIWAYAT DIABETES' == 'Tidak diketahui':")
    st.write(diabet)

    # --------------- Duplicate -----------------
    st.write("### Mengecek Data Duplicate")

    duplicate = df.duplicated().sum()

    st.write("\nData Duplicate:")
    st.write(duplicate)

def show_preprocessing():
    st.title("Preprocessing Data")

    # --------------- Load Data -----------------
    df = pd.DataFrame(pd.read_excel("dataset/tuberculosis.xlsx"))

    st.write("Data yang tersedia:")
    st.write(df.head())

    # --------------- Penghapusan Data Duplicate -----------------
    st.write("### Penghapusan Data Duplicate")

    df = df.drop_duplicates()
    duplicate = df.duplicated().sum()

    st.write("\nData Duplicate:")
    st.write(duplicate)

    # --------------- Pengisian Data Missing Value -----------------
    st.write("### Pengisian Data Missing Value")

    # Tampilkan hasil
    st.write("Mengisi nilai yang hilang")
    df['FOTO TORAKS'] = df['FOTO TORAKS'].replace('Tidak dilakukan', 'Positif')
    df['STATUS HIV'] = df['STATUS HIV'].replace('Tidak diketahui', 'Negatif')
    df['RIWAYAT DIABETES'] = df['RIWAYAT DIABETES'].replace('Tidak diketahui', 'Tidak')
    df['HASIL TCM'] = df['HASIL TCM'].replace('Tidak dilakukan', 'Rif Sensitif')
    st.write(df.head())

    # --------------- Encoding Data -----------------
    st.write("### Encoding Data")

    col1, col2 = st.columns(2,vertical_alignment='top')

    # --------------- Sebelum Encoding -----------------
    with col1:
        st.write("#### Sebelum Encoding")
        st.write("JENIS KELAMIN:", df['JENIS KELAMIN'].unique())
        st.write("FOTO TORAKS:", df['FOTO TORAKS'].unique())
        st.write("STATUS HIV:", df['STATUS HIV'].unique())
        st.write("RIWAYAT DIABETES:", df['RIWAYAT DIABETES'].unique())
        st.write("HASIL TCM:", df['HASIL TCM'].unique())

    with col2:
        st.write("#### Setelah Encoding")

        le = LabelEncoder()
        df = df.apply(lambda x: le.fit_transform(x))
        
        # --------------- Setelah Encoding -----------------
        st.write("JENIS KELAMIN:", df['JENIS KELAMIN'].unique())
        st.write("FOTO TORAKS:", df['FOTO TORAKS'].unique())
        st.write("STATUS HIV:", df['STATUS HIV'].unique())
        st.write("RIWAYAT DIABETES:", df['RIWAYAT DIABETES'].unique())
        st.write("HASIL TCM:", df['HASIL TCM'].unique())

    st.write("#### Tipe Data")
    # --------------- Type Data -----------------
    st.write(df.dtypes)

    # --------------- Final data -----------------
    st.write("### Data Akhir")
    st.write(df.head())
    st.session_state['preprocessed_data'] = df

def show_model():
    st.title("Modeling")
    
    if 'preprocessed_data' not in st.session_state:
        st.write("Silakan lakukan preprocessing data terlebih dahulu.")
        return
    
    # load data
    df = st.session_state['preprocessed_data']
    df = pd.concat([df[df.columns[0:2]], df[df.columns[3:]]], axis=1)
    
    st.write("Data yang telah dipreproses:")
    st.write(df.head())

    # Memisahkan fitur dan label
    features_df = df.iloc[:, :-1]
    labels_df = df.iloc[:, -1]

    # Membagi data menjadi data latih dan data uji
    train_features, test_features, train_labels, test_labels = train_test_split(features_df, labels_df, test_size = 0.25, random_state=1)

    # Membuat model Decision Tree dengan kriteria 'entropy' untuk C4.5
    rf = RandomForestClassifier()

    # Melatih model
    rf.fit(train_features, train_labels)

    # Memprediksi data uji
    predictions = rf.predict(test_features)

    # Evaluasi model
    st.subheader("Akurasi")
    st.write(f"{accuracy_score(test_labels, predictions)*100:.2f}%")

    cr = classification_report(test_labels, predictions)
    st.subheader("Classification Report")
    st.text(cr)

    # Menampilkan Confusion Matrix sebagai tabel
    cm = confusion_matrix(test_labels, predictions)
    cm_df = pd.DataFrame(cm, 
                                    index=[f'Actual {i}' for i in range(len(cm))], 
                                    columns=[f'Predicted {i}' for i in range(len(cm))])
    st.subheader("Confusion Matrix")
    st.table(cm_df)

    # Menampilkan pohon keputusan
    for i in range(3):
        # Pick one tree from the forest, e.g., the first tree (index 0)
        tree_to_plot = rf.estimators_[i]

        name_class = [str(c) for c in tree_to_plot.classes_]

        # Plot the decision tree
        plt.figure(figsize=(30, 20))
        plot_tree(tree_to_plot, feature_names=features_df.columns, class_names=name_class, filled=True, rounded=True, fontsize=10)
        plt.title("Decision Tree from Random Forest")
        plt.show()
        st.subheader("Random Forest Tree")
        st.pyplot(plt)

def show_testing():
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

    st.title("Prediksi Lokasi Anatomi TBC")

    umur = st.text_input("Umur", 1)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["P", "L"])
    foto_toraks = st.selectbox("Foto Toraks", ["Positif", "Negatif"])
    status_hiv = st.selectbox("Status HIV", ["Negatif", "Positif"])
    riwayat_diabetes = st.selectbox("Riwayat Diabetes", ["Tidak", "Ya"])
    hasil_tcm = st.selectbox("Hasil TCM", ["Rif Sensitif", "Negatif", "Rif resisten"])

    jk_encoded = features_encode["JENIS KELAMIN"][jenis_kelamin]
    foto_encoded = features_encode["FOTO TORAKS"][foto_toraks]
    hiv_encoded = features_encode["STATUS HIV"][status_hiv]
    diabetes_encoded = features_encode["RIWAYAT DIABETES"][riwayat_diabetes]
    tcm_encoded = features_encode["HASIL TCM"][hasil_tcm]

    # Create a feature vector
    input_data = pd.DataFrame({
        "UMUR": [umur],
        "JENIS KELAMIN": [jk_encoded],
        "FOTO TORAKS": [foto_encoded],
        "STATUS HIV": [hiv_encoded],
        "RIWAYAT DIABETES": [diabetes_encoded],
        "HASIL TCM": [tcm_encoded]
    })

    check= st.button("Prediksi")

    file = open("model/rf_pkl",'rb')
    model = pickle.load(file)

    prediction = model.predict(input_data)

    if  check:
        st.write(f"##### Hasil prediksi : :red[{labels_encode[int(prediction[0])]}]")

if __name__ == "__main__":
    st.set_page_config(page_title="Random Forest Classificication", page_icon="🌳")
    main()