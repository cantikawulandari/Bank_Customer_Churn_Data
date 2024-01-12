import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

# Mengimport model 
nb = pickle.load(open('NB.pkl','rb'))

# Menload dataset
data = pd.read_csv('Bank Customer.csv')


st.title('Aplikasi Bank Customer')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Costumer Churn</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1, unsafe_allow_html=True)
activities = [' ', 'MultinomialNB']
option = st.sidebar.selectbox('Model Name',activities)
st.sidebar.header('Data Churn Costumer Bank')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p> Dataset Bank Costumer Churn merupakan dataset yang menampilkan kostumer bank yang mengalami naik turun</p>
    """
    st.markdown(html_layout2, unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe Dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

# Proses training test split
X = data.drop('churn',axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    # Input untuk fitur numerik
    Skor_Kredit     = st.sidebar.slider('credit_score', 0, 20, 1)
    Age             = st.sidebar.slider('age', 0, 140, 40)
    Tenure          = st.sidebar.slider('tenure', 0, 100, 25)
    Jumlah_Produk   = st.sidebar.slider('products_number', 0, 1000, 120)
    Estimasi_Gaji   = st.sidebar.slider('estimated_salary', 0, 80, 25)
    customer_id     = st.sidebar.slider('customer_id', 0, 5, 10000)

    # Input untuk fitur kategorikal
    credit_card     = st.sidebar.checkbox('Has Credit Card')
    active_member   = st.sidebar.checkbox('Active Member')
    churn           = st.sidebar.radio('Churn', ['Yes', 'No'])
    
    # Mengonversi gender dan churn ke nilai float
    churn = 0.0 if churn == 'No' else 1.0

    # Membuat DataFrame
    user_report_data = {
        'customer_id': customer_id,
        'credit_card': credit_card,
        'active_member': active_member,
        'churn': churn,
        'credit_score': Skor_Kredit,
        'age': Age,
        'tenure': Tenure,
        'products_number': Jumlah_Produk,
        'estimated_salary': Estimasi_Gaji,
    }
        
    report_data = pd.DataFrame(user_report_data, index=[0])

    return report_data


#Data Pasion
user_data = user_report()
st.subheader('Data churn Kostumer Bank')
st.write(user_data)

user_result = nb.predict(user_data)
nb_score = accuracy_score(y_test,nb.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Churn Kostumer Aman'
else:
    output ='Churn Kostumer Rendah'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(nb_score*100)+'%')