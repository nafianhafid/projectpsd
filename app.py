import streamlit as st
import pickle as pkle
import os.path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score 
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

data_trainfull = pd.read_csv('train_full2.csv')
data_orders = pd.read_csv('orders.csv')
data_trainfull = data_trainfull.reset_index()
# create a button in the side bar that will move to the next page/radio button choice
datasetsatu = data_trainfull[['customer_id', 'gender','location_type','id','OpeningTime',
                            'language','vendor_rating','serving_distance','vendor_tag_name',
                            'delivery_charge']]
datasetsatu.rename(columns = {'vendor_rating':'mean_rating'},inplace=True)

#gabungkan dengan id yang sama
kolom = ['id','customer_id']
datasetsatu['semua'] = datasetsatu[kolom].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
#hilangkan duplikasi data id dan customer id
datasetsatu.drop_duplicates(['semua'],inplace=True)
#dataset order
datasetdua = data_orders[['akeed_order_id','customer_id','vendor_id','item_count',
                        'grand_total','vendor_rating']][:]
datasetdua.rename(columns = {'vendor_id':'id'}, inplace =True)

kolom = ['id','customer_id']
datasetdua['semua'] = datasetdua[kolom].apply(lambda row: '_'.join(row.values.astype(str)),axis=1)

#gabung dataset
datasetgabung = pd.merge(datasetsatu,datasetdua, on='semua', how='inner')
datasetgabung.head()
#rename kolom
datasetgabung.rename(columns = {'id_x':'vendor_id'}, inplace = True)
datasetgabung.rename(columns = {'customer_id_x' : 'customer_id'}, inplace =True)

#hapus kolom yang sama
datasetgabung.drop(['customer_id_y', 'id_y'], axis=1,inplace=True)

##dataset vendor
dataset2 = datasetsatu[['customer_id','id','vendor_tag_name']]
dataset2.rename(columns={'id':'vendor_id'}, inplace =True)

#cleaning data
kolom = ['delivery_charge','item_count','serving_distance','grand_total','vendor_rating']

#hapus kolom language dan hapus null pada kolom gender
datasetgabung.drop(['language'] ,axis=1,inplace=True)
datasetgabung = datasetgabung[datasetgabung['gender'].notnull()].reset_index(drop=True)

sex=pd.get_dummies(datasetgabung['gender'], columns = ['gender'],prefix="sex",drop_first=True)

datasetgabung=pd.concat([datasetgabung,sex],axis=1)

datasetgabung.drop(['gender'],axis=1,inplace=True)
#dataset untuk training
dataset1_training = datasetgabung[:]
train_for = datasetgabung[:]
dataset_train = train_for[['customer_id', 'vendor_id','OpeningTime','vendor_tag_name']]


# cek rasio null kolom vendor tag name

dataset_train = dataset_train[dataset_train['vendor_tag_name'].notnull()].reset_index(drop=True)

#cleaning vendor_tag_name
dataset_train['vendor_tag_name'] = dataset_train['vendor_tag_name'].apply(lambda x:x.lower())

# string to list
dataset_train['vendor_tag'] = dataset_train['vendor_tag_name'].str.split(',')


#Modelling
#Collaborative Filtering
customer_vendor_ratings = dataset1_training[['customer_id','vendor_id','vendor_rating']]
# hitung rata-rata rating tiap rating yang valid(kecuali miss data)
rating_not_null = []
for i in range(0, customer_vendor_ratings.shape[0]-1):
  if pd.isnull(customer_vendor_ratings.iloc[i][2]) == False and customer_vendor_ratings.iloc[i][2] !=0:
    rating_not_null.append(customer_vendor_ratings.iloc[i][2])

valid_rating_mean = np.mean(np.array(rating_not_null))

# mengisi data kosong dengan rata-rata valid rating
def rating_kosong(x):
  if pd.isnull(x) == True or x == 0:
    return valid_rating_mean
  else:
    return x
customer_vendor_ratings['rating'] = customer_vendor_ratings['vendor_rating'].apply(rating_kosong)

customer_vendor_ratings = customer_vendor_ratings[['customer_id', 'vendor_id','rating']]
# integrasi ke dalam tiap rating grup berdasarkan ratarata
customer_vendor_ratings_mean = customer_vendor_ratings.groupby(['customer_id','vendor_id']).mean()

#reset index
data_customer_vendor_rating_mean = customer_vendor_ratings_mean.reset_index()
# membuat full matriks
rating_full_matriks = data_customer_vendor_rating_mean.pivot(index='customer_id',
                                                             columns='vendor_id',
                                                             values='rating')
# Menghitung similarity semua pasangan dari customer dari full matrix
rating_matriks_dummy = rating_full_matriks.copy().fillna(0)

customer_similarity = cosine_similarity(rating_matriks_dummy,rating_matriks_dummy)

customer_similarity = pd.DataFrame(customer_similarity, index=rating_full_matriks.index,columns=rating_full_matriks.index)

# fungsi yang menghitung evaluasi regresi RMSE, MSE, MAPE, MAE AUC
def eval(y_true, y_pred):
  mae = mean_absolute_error(y_true,y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = r2_score(y_true,y_pred)

  return mae,mse,rmse

# K-Nearest Neighbour skor
def knn_skor(model, neighbor_size=0):
  id_pair = zip(data_customer_vendor_rating_mean['customer_id'],data_customer_vendor_rating_mean['vendor_id'])
  
  y_pred = np.array([model(customer, vendor, neighbor_size) for (customer, vendor) in id_pair])
  y_true = np.array(data_customer_vendor_rating_mean['rating'])
  return eval(y_true, y_pred)

# CF Model
def cf_knn(customer_id,vendor_id,neighbor_size=0):
  if vendor_id in rating_full_matriks:
    sim_scores = customer_similarity[customer_id].copy()
    vendor_ratings = rating_full_matriks[vendor_id].copy()
    none_rating_idx = vendor_ratings[vendor_ratings.isnull()].index

    vendor_ratings = vendor_ratings.drop(none_rating_idx)
    sim_scores = sim_scores.drop(none_rating_idx)

    if neighbor_size == 0:
      mean_rating = np.dot(sim_scores, vendor_ratings)/sim_scores.sum()
    else:
      if len(sim_scores > 1):
        neighbor_size = min(neighbor_size, len(sim_scores))
        sim_scores = np.array(sim_scores)
        vendor_ratings = np.array(vendor_ratings)
        customer_idx = np.argsort(sim_scores)
        sim_scores = sim_scores[customer_idx][-neighbor_size:]
        vendor_ratings = vendor_ratings[customer_idx][-neighbor_size:]
        mean_rating = np.dot(sim_scores, vendor_ratings)/sim_scores.sum()
      else:
        mean_rating = valid_rating_mean
  else:
    mean_rating = valid_rating_mean
  
  return mean_rating

#knn_skor(cf_knn, neighbor_size=20)

def cf_rekom_vendor(customer_id, n_items, neighbor_size=0):
  customer_vendor = rating_full_matriks.loc[customer_id].copy()

  for vendor in rating_full_matriks:
    if pd.notnull(customer_vendor.loc[vendor]):
      customer_vendor.loc[vendor] = 0
    else:
      customer_vendor.loc[vendor] = cf_knn(customer_id, vendor, neighbor_size)
  
  vendor_sort = customer_vendor.sort_values(ascending=False)[:n_items]
  rekom_vendor_temp = dataset1_training.loc[vendor_sort.index]
  rekom_vendor_temp2 = rekom_vendor_temp[['vendor_id','mean_rating','vendor_tag_name']]
  rekom_vendors = rekom_vendor_temp2.reset_index(drop=True)

  return rekom_vendors

#contoh rekomendasi list

#cf_rekom_vendor(customer_id='00FQ1U9',n_items=5, neighbor_size=30)

##Modelling2
data_content_for_analy = dataset_train[:]
data_content_for_analy['vendor_tag'] = data_content_for_analy['vendor_tag_name'].str.split(',')

#strip
data_content_for_analy['vendor_tag'] = data_content_for_analy['vendor_tag'].apply(lambda x:[str.lower(i.replace(' ',''))for i in x])
#cek kata yang sama

def similar(a, b):
  rasio = SequenceMatcher(None, a, b).ratio()
  
  return print("Persamaan dari {} dan {} : {}".format(a,b,rasio))

similar('pasta','pastas')
similar('pasta','pastry')
similar('pizza','pizzas')
similar('soups','shuwa')
similar('shawarma','shuwa')
similar('thali','thai')
similar('milkshakes','mishkak')

data_content_for_analy['vendor_tag'] = data_content_for_analy['vendor_tag'].apply(lambda x:[i.replace('pastas','pasta')for i in x])
data_content_for_analy['vendor_tag'] = data_content_for_analy['vendor_tag'].apply(lambda x:[i.replace('pizzas','pizza')for i in x])
data_content_for_analy['vendor_tag'] = data_content_for_analy['vendor_tag'].apply(lambda x:[i.replace('thali','thai')for i in x])
data_content_for_analy['vendor_tag1'] = data_content_for_analy['vendor_tag'].apply(lambda x:' '.join(x))

prak = data_content_for_analy.drop_duplicates('vendor_id',keep='first',inplace=False)

prak['vendor_id'] = prak['vendor_id'].astype(str)
prak1 = prak[:]

#TF IDF Model
prak.set_index('vendor_id', inplace=True)

#vectorizer untuk mendapat indeks makanan

vectorizer = TfidfVectorizer()
count_matriks = vectorizer.fit_transform(prak['vendor_tag1'])

indices = pd.Series(prak.index)

#untuk melihat similarity
cosine_sim = cosine_similarity(count_matriks,count_matriks)
prak = prak.reset_index()
listmakanan = vectorizer.vocabulary_

def get_rekomendasi(id, cosine_sim=cosine_sim):
  indices = pd.Series(prak.index, index = prak['vendor_id']).drop_duplicates()
  idx = indices[id]
  listrekomendasi = pd.DataFrame(columns=['Nama Makanan', 'Persentase Rekomendasi'])
  sim_scores = enumerate(cosine_sim[idx])
  sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:11]
  for x in list(sim_scores):
    for a in listmakanan:
      if x[0] == listmakanan[a]:
        listrekomendasilen = len(listrekomendasi)
        listrekomendasi.loc[listrekomendasilen] = [a, x[1]]  
  return listrekomendasi
#menampilkan rekomendasi makanan untuk vendor dengan id 113
#get_rekomendasi('113')
next = st.sidebar.button('Next on list')

# will use this list and next button to increment page, MUST BE in the SAME order
# as the list passed to the radio button
new_choice = ['Home','Exploratory Data Analysis','Modelling']


if os.path.isfile('next.p'):
    next_clicked = pkle.load(open('next.p', 'rb'))
    # check if you are at the end of the list of pages
    if next_clicked == len(new_choice):
        next_clicked = 0 # go back to the beginning i.e. homepage
else:
    next_clicked = 0 #the start

# this is the second tricky bit, check to see if the person has clicked the
# next button and increment our index tracker (next_clicked)
if next:
    #increment value to get to the next page
    next_clicked = next_clicked +1

    # check if you are at the end of the list of pages again
    if next_clicked == len(new_choice):
        next_clicked = 0 # go back to the beginning i.e. homepage

# create your radio button with the index that we loaded
choice = st.sidebar.radio("Menu",('Home','Exploratory Data Analysis', 'Modelling'), index=next_clicked)

# pickle the index associated with the value, to keep track if the radio button has been used
pkle.dump(new_choice.index(choice), open('next.p', 'wb'))

# finally get to whats on each page
if choice == 'Home':
    st.title("Projek Akhir Kelompok 11")
    st.subheader("Anggota:")
    st.text("M Farid Muzayyani 195150207111040")
    st.text("M Hafidh Ilmi Nafi'an 195150200111042")
    st.text("M Fachri Pratama 185150200111047")
    expander_label = st.expander(label='Latar Belakang')
    expander_metode = st.expander(label='Metode')
    expander_dataset = st.expander(label = 'Dataset')
    expander_label.write('Sering kali kita berkumpul dengan keluarga atau teman ketika datang waktu makan. Dengan adanya aplikasi rekomendasi, orang akan lebih tertarik mengenai bagaimana kita akan menyukai restoran. Di masa lalu, orang memperoleh saran untuk restoran dari teman- teman. Meskipun metode ini digunakan, namun memiliki beberapa keterbatasan. Pertama, rekomendasi dari teman terbatas pada tempat-tempat yang telah mereka kunjungi sebelumnya. Dengan demikian, pengguna tidak dapat memperoleh informasi tentang tempat-tempat yang belum dikunjungi oleh teman-teman mereka. Selain itu, ada kemungkinan pengguna tidak menyukai tempat yang direkomendasikan oleh teman-teman mereka. Pada dataset Restaurant Recommendation Challenge ini bertujuan untuk menentukan rekomendasi makanan apa yang akan ditawarkan oleh sebuah restaurant kepada customer berdasarkan makanan yang paling sering dipesan oleh customer.')
    expander_metode.write('Metode yang digunakan untuk prediksi adalah KNN dan Cosine Similarity')
    expander_dataset.image('dataset.png')
elif choice == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis')
    st.subheader('Melihat 10 Data Pertama')

    st.text('Dataframe data_trainfull')
    st.dataframe(data_trainfull.head(10))
    expander_info1 = st.expander(label = 'Data Info')
    buffer = io.StringIO()
    data_trainfull.info(buf=buffer)
    s = buffer.getvalue()
    expander_info1.text(s)

    st.text('Dataframe data_orders')
    st.dataframe(data_orders.head(10))
    expander_info2 = st.expander(label = 'Data Info')
    buffer2 = io.StringIO()
    data_orders.info(buf=buffer2)
    s2 = buffer2.getvalue()
    expander_info2.text(s2)
    #plot
    expander_plot1 = st.expander(label='Persebaran Gender')
    fig = plt.figure(figsize=(10,4))
    sns.countplot('gender', data=data_trainfull)
    expander_plot1.pyplot(fig)
    #plot
    expander_plot2 = st.expander(label='Persebaran Gender berdasarkan Lokasi Pemesanan')
    fig = plt.figure(figsize=(10,4))
    sns.countplot(data_trainfull['gender'], hue=data_trainfull['location_type'])
    expander_plot2.pyplot(fig)
elif choice == 'Modelling':
    st.title('Modelling')

    #st.subheader("Data Customer")
    
    #col1,col2 = st.columns(2)
    
    st.subheader("Data Customer")
    customer_unique = pd.DataFrame(data_customer_vendor_rating_mean['customer_id'].unique())
    st.dataframe(customer_unique[:10])
    

    st.subheader('Evaluasi Rating Based Model') 
    mae,mse,rmse = knn_skor(cf_knn, neighbor_size=20)    
    st.write('Mean Absolute Error     : ', mae)
    st.write('Mean Squared Error      : ', mse)
    st.write('Root Mean Squared Error: ', rmse)

    st.subheader('Rekomendasi Restaurant dan Makanan')
    textbyuser = st.text_input("Masukkan Inputan Id Customer")
    if st.button('Prediksi'):
      restoran = cf_rekom_vendor(textbyuser,n_items=5, neighbor_size=30)
      restoranrekom = restoran.iloc[0]['vendor_id']
      st.write('Restaurant Rekomendasi untuk ID Customer ',textbyuser,' adalah Restaurant dengan ID ',restoranrekom ,'dengan makanan')
      st.dataframe(get_rekomendasi(str(restoranrekom)))
