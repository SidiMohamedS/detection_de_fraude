import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import json
from io import StringIO



req = requests.get("http://127.0.0.1:5000/streamlit")
print(req)
resultat = req.json()
print(resultat)

# data = pd.read_json(resultat)
data = pd.read_json(StringIO(resultat))

st.title('Analyse et Visualisation des Données pour la Détection de Fraude ')


section=st.sidebar.radio('Plan',
                 ('Jeu de données', 'Nettoyage des données',
                 'Analyse exploratoire','Visualisation', 'prédiction'))

if section=='Jeu de données':
    st.markdown('### Descriptions de l\'ensembles des données')
    st.write('L\'ensemble de données que nous utilisons vise à détecter les transactions frauduleuses dans le domaine financier afin de permettre aux institutions financières de bloquer avec précision les transactions frauduleuses ')
    st.write('Nous avons un grand nombre d\'enregistrement soit:')
    st.write('- 26 401 lignes ')
    st.write('- 12 colonnes')
    st.markdown('#### Les types de variables ')
    st.write('- dix (09) variables quantitatives continues.')
    st.write('- deux (2)variables quantitatives discrète.')
    if st.button('voir le jeu de données'):
      st.dataframe(data)
      
if section=='Nettoyage des données':
     st.markdown('#### Prétraitement des données')
     st.write('Lors de l\'étape de nettoyage des données, nous avons examiné attentivement la structure des données, vérifié la présence de données, et identifié les éventuelles valeurs manquantes. La présence de doublons n \'a pas d\'impact sur notre traitement, car nous travaillons avec des données de transactions.')     
     if st.button('Nombre de valeur manquante'):
        st.write (data.isnull().sum())
     if st.button('Nombre de doublons'):
        nombre_de_doublons = data.duplicated().sum()
        st.write(f"Nombre de doublons dans les données : {nombre_de_doublons}")
     st.markdown("Informations sur les données :")
     if st.button('Info'):
       data_info = data.info
       st.text(data_info) 

if section=='Analyse exploratoire':
   st.markdown('###### Analyse exploratoire des données')
   st.write('Au cours de l\'analyse exploratoire:')
   st.write('- Nous avons réalisé une analyse descriptive de nos données. Nous avons noté que la moyenne des montants de transactions est de [valeur] avec un montant minimum de 0. Pour améliorer la pertinence du modèle, nous avons éliminé les variables non significatives telles que l\'index, nameOrig et nameDest.')
   if st.button('statistique'):
     desc=data.describe()
     st.write(desc)    
   st.write('- Nous avons transformé la variable step (représentant les jours) en heures pour une analyse plus détaillée. Cette transformation nous a permis d\'obtenir des insights plus précis dans notre exploration des données.')  
   st.write('- Un constat intéressant a émergé lors de l\'examen des catégories des variables discrètes. Nous avons observé que seules les transactions de type CASH_OUT ou TRANSFER semblent correspondre le plus aux moyens de transfert associés aux transactions frauduleuses.')    
   st.write('- Les modifications et les constatations au cours de l\'analyse exploratoire ont renforcé notre compréhension des données')
if section=='Visualisation':
      st.markdown('###### Description des types de transferts')
      st.write('- Transaction par retrait -> 0')
      st.write('- Transaction par transfère -> 1')
      if st.button('Voir les analyses graphiques'):
         st.markdown("#### Proportions des transactions frauduleuses")
         fraud_counts = data['vraiValeur'].value_counts()
         st.bar_chart(fraud_counts, color="#1f77b4")
         
         selected_type = st.selectbox("Choisir le type de transfère", data['type_TRANSFER'].unique())
   # Filtrer les données en fonction du type sélectionné
         filtered_data = data[data['type_TRANSFER'] == selected_type]
         average_amount_per_hour = filtered_data.groupby('step')['amount'].mean().reset_index()

   # Créer une figure avec Matplotlib
         fig, ax = plt.subplots(figsize=(12, 6))
   # Ajouter un titre au graphe
         ax.plot(average_amount_per_hour['step'], average_amount_per_hour['amount'])
         ax.set_title(f"L'évolution du montant moyen par heure en fonction du type de transaction {selected_type}")
   # Afficher le graphique avec Streamlit
         st.pyplot(fig)
      total_amount_by_type = data.groupby('type_TRANSFER')['amount'].sum()
      max_amount_type = total_amount_by_type.idxmax()
      fig, ax = plt.subplots(figsize=(6, 6))
      ax.pie(total_amount_by_type,labels=total_amount_by_type.index, autopct='%1.1f%%', startangle=90)
      ax.set_title("Répartition des montants par type de transfert")
      st.pyplot(fig)

#  n_bins = st.number_input(
#       label="Choisir un nombre de bins",
#       min_value=1, value=20)
#       # Créer une figure avec Matplotlib
#       fig, ax = plt.subplots(figsize=(12, 8))
#       sns.countplot(x='vraiValeur', data=data, bins=n_bins, ax=ax)
#       ax.set_title("Proportion des transactions frauduleuses")
#       st.pyplot(fig)

# fig, ax = plt.subplots()
# n_bins = st.number_input(
#     label="Choisir un nombre de bins",
#     min_value=10,
#     value=20
# )

# ax.hist(data.id, bins=n_bins)

# title=st.text_input(label="Saisir le titre du graphe")
# st.title(title)
# st.pyplot(fig)

# data = [1,2,3]
# df= pd.DataFrame(data)
# st.area_chart(df)