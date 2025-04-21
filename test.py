import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
from shapely.geometry import Point
import joblib
from datetime import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

df_usagers = pd.read_csv(r"C:\Users\macha\Desktop\Test_py\all_data_usagers_2019_2023.csv")
df_caracteristiques = pd.read_csv(r"C:\Users\macha\Desktop\Test_py\all_data_carac_2019_2023.csv")
df_lieux= pd.read_csv(r"C:\Users\macha\\Desktop\Test_py\all_data_lieux_2019_2023.csv")
df_vehicules=pd.read_csv(r"C:\Users\macha\Desktop\Test_py\all_data_vehicules_2019_2023.csv")
df_total_final = pd.read_csv(r"C:\Users\macha\Desktop\Test_py\df_total_final.csv")
df_machine_learning =pd.read_csv(r"C:\Users\macha\Desktop\Test_py\df_machine_learning.csv")
df_usagfinal = pd.read_csv(r"C:\Users\macha\Desktop\Test_py\df_usagfinal.csv")
df_total_final = pd.read_csv(r"C:\Users\macha\Desktop\Test_py\df_total_final.csv")
df_lieux_final = pd.read_csv(r"C:\Users\macha\Desktop\Test_py\df_lieux_final.csv")
df_carac_lieux = pd.read_csv(r"C:\Users\macha\Desktop\Test_py\df_carac_lieux.csv")
df_merged_usag_veh_final = pd.read_csv(r"C:\Users\macha\Desktop\Test_py\df_merged_usag_veh_final.csv")
df  = pd.read_csv(r"C:\Users\macha\Desktop\Test_py\df_analyse_exploratoire.csv")
def afficher_infos_dataframe(df):
    st.write(f"**Nombre de lignes :** {df.shape[0]}")
    st.write(f"**Nombre de colonnes :** {df.shape[1]}")


    # Afficher le nombre de valeurs manquantes
    st.write("Valeurs manquantes")
    valeurs_manquantes = df.isnull().sum()
    valeurs_manquantes = valeurs_manquantes[valeurs_manquantes > 0]  # Garde uniquement les colonnes avec valeurs manquantes


    if not valeurs_manquantes.empty:
        st.dataframe(valeurs_manquantes)  # Affiche uniquement si des valeurs manquantes existent
    else:
        st.write("Aucune valeur manquante dans ce dataset")  # Affichage textuel sinon
    
class CyclicalFeatures(BaseEstimator, TransformerMixin):
        def __init__(self, period=24):
                    self.period = period

        def fit(self, X, y=None):
                    return self

        def transform(self, X):
                    X = X.astype(float)  # Assurer que les valeurs sont numériques
                    X_sin = np.sin(2 * np.pi * X / self.period)
                    X_cos = np.cos(2 * np.pi * X / self.period)
                    return np.c_[X_sin, X_cos]  # Retourne un array 2D avec sin et cos




st.sidebar.title("Projet accidents de la route en France")
pages=["Présentation du projet", "Exploration des données", "DataVizualization", "Modélisation","Conclusion"]

if "page" not in st.session_state:
    st.session_state.page = "Présentation du projet"
if "show_exploration" not in st.session_state:
    st.session_state.show_exploration = False
if "show_visualisation" not in st.session_state:
    st.session_state.show_visualisation = False
if "show_modelisation" not in st.session_state:
    st.session_state.show_modelisation = False
if "show_conclusion" not in st.session_state:
    st.session_state.show_conclusion = False

# Fonction pour changer de page
def switch_page(new_page):
    st.session_state.page = new_page

# Fonction pour basculer l'affichage des sections
def toggle_section(section):
    # Réinitialiser toutes les sections sauf celle cliquée
    if section == "exploration":
        st.session_state.show_exploration = not st.session_state.show_exploration
        st.session_state.show_visualisation = False
        st.session_state.show_modelisation = False
        st.session_state.show_conclusion = False
    elif section == "visualisation":
        st.session_state.show_visualisation = not st.session_state.show_visualisation
        st.session_state.show_exploration = False
        st.session_state.show_modelisation = False
        st.session_state.show_conclusion = False
    elif section == "modelisation":
        st.session_state.show_modelisation = not st.session_state.show_modelisation
        st.session_state.show_exploration = False
        st.session_state.show_visualisation = False
        st.session_state.show_conclusion = False
    elif section == "conclusion":
        st.session_state.show_conclusion = not st.session_state.show_conclusion
        st.session_state.show_exploration = False
        st.session_state.show_visualisation = False
        st.session_state.show_modelisation = False

# Gestion de l'état de la page avec session_state
if "page" not in st.session_state:
    st.session_state.page = "Présentation du projet"
if "show_exploration" not in st.session_state:
    st.session_state.show_exploration = False
if "show_visualisation" not in st.session_state:
    st.session_state.show_visualisation = False
if "show_modelisation" not in st.session_state:
    st.session_state.show_modelisation = False
if "show_conclusion" not in st.session_state:
    st.session_state.show_conclusion = False

############################################################ SIDEBAR ##########################################################################################################################################################

st.sidebar.button("Présentation du projet", on_click=switch_page, args=("Présentation du projet",))


######## Exploration des données
# Exploration des données avec effet toggle
if st.sidebar.button("Exploration des données", on_click=toggle_section, args=("exploration",)):
    switch_page("Exploration des données")

# Affichage des sous-sections sous "Exploration des données" avec checkboxes
if st.session_state.show_exploration:
    presentation_donnees_checkbox = st.sidebar.checkbox("Préparation et nettoyage des données")
    elaboration_dataset_checkbox = st.sidebar.checkbox("Dataset final")
  
else:
    presentation_donnees_checkbox = None
    elaboration_dataset_checkbox = None
   
 

############ Data visualisation
# Data Visualisation avec effet toggle
if st.sidebar.button("Data Visualisation", on_click=toggle_section, args=("visualisation",)):
    switch_page("Data Visualisation")

# Affichage des sous-sections sous "Data Visualisation" avec checkboxes
if st.session_state.show_visualisation:
    distribution_gravité_checkbox = st.sidebar.checkbox("Distribution de la variable gravité")
    conditions_accident_checkbox = st.sidebar.checkbox("Contexte temporel")
    localisation_checkbox = st.sidebar.checkbox("Facteurs géographiques et infrastructurels")
    usagers_checkbox = st.sidebar.checkbox("Profils et statuts des usagers") 
    vehicules_checkbox = st.sidebar.checkbox("Typologie et points d'impact des véhicules")
    matrices_corrélation_checkbox = st.sidebar.checkbox("Matrices de corrélation")
else: 
    distribution_gravité_checkbox = None
    conditions_accident_checkbox = None
    usagers_checkbox = None
    localisation_checkbox = None
    vehicules_checkbox = None
    matrices_corrélation_checkbox = None

############## Modélisation  
# Modélisation
if st.sidebar.button("Modélisation", on_click=toggle_section, args=("modelisation",)):
    switch_page("Modélisation")

# Affichage des sous-sections sous "Modélisation" avec checkboxes
if st.session_state.show_modelisation:
    Méthodologie_et_résultats_checkbox = st.sidebar.checkbox("Méthodologie et résultats")
    Prédictions_checkbox = st.sidebar.checkbox("Prédictions")

############## Conclusion

# Affichage des sous-sections sous "Conclusion" avec checkboxes
if st.sidebar.button("Conclusion", on_click=toggle_section, args=("conclusion",)):
    switch_page("Conclusion")




######################################################################################################################################################################################################################""

############################################################################################################# Page présentation du projet #########################################################################################################
if st.session_state.page == "Présentation du projet":
    
    tab1, tab2 = st.tabs(["Projet", " Equipe"])

# Contenu du premier onglet
    with tab1:
        st.header("Projet")
        st.markdown("""
<div style="background-color: #e8f5e9; border: 2px solid #81c784; padding: 20px; border-radius: 10px;">

Ce projet a pour objectif de <strong>prédire la gravité des accidents de la route en France</strong>, à partir des données historiques recueillies entre <strong>2005 et 2023</strong>.

Après une analyse collective des besoins, nous avons défini une finalité concrète :  
Développer un modèle de machine learning destiné à <strong>assister un centre d’appel d’urgence</strong> dans l’évaluation <strong>en temps réel</strong> de la gravité d’un accident signalé.  
Cela permettra de <strong>mobiliser rapidement les secours appropriés</strong>.

La <strong>variable cible</strong> de notre modèle est la <strong>gravité de l’accident</strong>.

<hr style="border:1px solid #81c784;">

<h4>Choix méthodologiques</h4>

Pour répondre à cet objectif, deux décisions clés ont été prises :

<ol>
<li><strong>Limiter la période d’étude aux années 2019 à 2023</strong><br>
Ces données sont <em>plus récentes</em>, <em>mieux renseignées</em> et présentent une <em>structure homogène</em>.  
Ce choix nous permet de <strong>fusionner les quatre bases de données</strong> disponibles (caractéristiques, lieux, usagers, véhicules) de manière cohérente autour de la variable cible, tout en conservant un nombre d’observations suffisant (<strong>273 226 lignes</strong>) pour entraîner efficacement notre modèle.</li><br>

<li><strong>Supprimer les variables jugées non pertinentes pour la prédiction de la gravité</strong><br>
Afin d’alléger le jeu de données et d’améliorer la performance du modèle, nous avons écarté les variables ne présentant pas de lien clair avec notre problématique.</li>
</ol>

<hr style="border:1px solid #81c784;">

Ces choix nous permettent d’aborder la modélisation avec un jeu de données à la fois <strong>riche</strong>, <strong>cohérent</strong> et <strong>pertinent</strong>.

</div>
""", unsafe_allow_html=True)



# Contenu du deuxième onglet
    with tab2:
        st.subheader("Equipe")
        st.markdown("""
<div style="background-color: #eaf8ec; border: 2px solid #81c784; padding: 15px; border-radius: 10px;">
        <li>Appolinaire Allarassem</li>
        <li>Juliette Meunier</li>
        <li>Macha Lagune</li>
    </ul>
</div>
""", unsafe_allow_html=True)

#################################### Page exploration de données et ses sous catégories ####################################
elif st.session_state.page == "Exploration des données":
    st.header("Exploration des données")
    st.markdown("""
<div style="background-color: #eaf8ec; border: 2px solid #81c784; padding: 15px; border-radius: 10px;">


L'objectif initial est de <strong>fusionner les quatre jeux de données</strong> mis à disposition, en <strong>conservant un maximum d'informations pertinentes</strong> pour notre analyse. Cette étape inclut un premier travail de nettoyage et de préparation des données, comprenant les actions suivantes :

<ul>
<li><strong>Regroupement de certaines valeurs similaires</strong> dans des catégories plus larges,</li>
<li><strong>Traitement des valeurs aberrantes</strong> (remplacement ou suppression),</li>
<li><strong>Élimination des variables jugées non pertinentes</strong> pour notre problématique.</li>
</ul>

Afin de simplifier le jeu de données et de le recentrer sur notre objectif — la prédiction de la gravité des accidents — nous avons choisi de <strong>supprimer les variables suivantes</strong>, considérées comme peu informatives ou non exploitables dans notre contexte :

<pre>'lum', 'com', 'adr', 'voie', 'v1', 'v2', 'circ', 'vosp', 'prof', 'pr', 'pr1', 'plan', 'lartpc', 'larrout', 'infra', 'situ', 'id_vehicule', 'num_veh', 'senc', 'occutc', 'id_usager', 'trajet', 'locp', 'actp', 'etatp'</pre>

</div>
""", unsafe_allow_html=True)

#"Préparation des données"
    if presentation_donnees_checkbox:
        st.subheader('Préparation et nettoyage des données')
        # Définition des options
        rubriques = [
    "Rubrique USAGERS","Rubrique VÉHICULES", "Rubrique LIEUX","Rubrique CARACTÉRISTIQUES"]

# Stocker les rubriques sélectionnées
        rubriques_selectionnees = [rubrique for rubrique in rubriques if st.checkbox(rubrique)]

        if "Rubrique USAGERS" in rubriques_selectionnees:
            st.write("### Usagers")
            st.write("La première étape est le travail sur la table Usagers qui est celle avec le niveau d'agrégation le plus fin (1 ligne par usager, total 619971 lignes)")
            st.dataframe(df_usagers)
        # Checkboxes pour afficher les infos
            if st.checkbox("Afficher infos et valeurs manquantes (USAGERS)"):
                afficher_infos_dataframe(df_usagers)
        
            st.write (" ##### Variables transformées et nettoyées dans ce dataset")
            options = ['']+['age']
            choix_usagers = st.selectbox('sélectionner une variable', options)
        
            if 'age' in choix_usagers:
                st.write("##### Age")
                df_usagers["annee"]=df_usagers["Num_Acc"].apply(lambda x: str(x)[:4])   # je rajoute une colonne "année"
                df_usagers["annee"]=df_usagers["annee"].astype(int)
                df_usagers["age"]=df_usagers['annee']-df_usagers['an_nais']# je rajoute une colonne année de naissance de l'usager
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(data=df_usagers, x='age', palette="viridis", ax=ax)
                # Ajout du titre
                plt.title("Distribution de l'âge des usagers accidentés")
                st.pyplot(fig)
                age_median = df_usagers['age'].median()
                st.write("L'age médian est")
                st.write(age_median)
                Q1 = df_usagers['age'].quantile(0.25)  # Premier quartile (25e percentile)
                Q3 = df_usagers['age'].quantile(0.75)  # Troisième quartile (75e percentile)
                IQR = Q3 - Q1                  # IQR (Interquartile Range)
                # Définir les limites pour identifier les outliers
                upper_limit = Q3 + 1.5 * IQR
                st.write ('la upper limite est ')
                st.write(upper_limit)
                mode_value = df_usagers['age'].mode()[0]
                st.write('le mode de la variable age est')
                st.write(mode_value)
                st.write("Nous créons ensuite des tranches d'âge , dont une supérieure à la upper limit de 96 ans, pour pourvoir supprimer cette tranche après encodage sans supprimer les autres informations des accidents concernés. Nous remplacerons les valeurs manquantes par le mode de la variable age ")
                st.write("Les tranches d'âge sont ['0-17'], ['18-35'], ['35-61'],['61-95'], ['>96] ")
       
            st.write("Nous obtenons le  dataframe usager final suivant, qui compte désormais une ligne par véhicule")
            st.dataframe(df_usagfinal)
            if st.checkbox("Afficher infos et valeurs manquantes df_usagfinal)"):
                afficher_infos_dataframe(df_usagfinal)
           
        
        if "Rubrique VÉHICULES" in rubriques_selectionnees:  
            st.write("### Véhicules")
            st.write("La 2ème étape consiste à travailler sur la table véhicules afin de pouvoir la merger à la table USAGERS précédente qui compte 1 ligne par véhicule")
            st.dataframe(df_vehicules)
            if st.checkbox("Afficher infos et valeurs manquantes (VEHICULES)"):
                afficher_infos_dataframe(df_vehicules)
            
            st.write (" ##### Variables transformées et nettoyées dans ce dataset")
            options = [''] + ['catv', 'choc']
            choix_véhicules = st.selectbox ('sélectionner une variable', options)
        
            if 'catv' in choix_véhicules:
                st.write("##### Catégorie de véhicule")
                catv_counts = df_vehicules.catv.value_counts()
                fig, ax = plt.subplots(figsize=(4, 3))

                # Tracer un graphique en barres
                ax.bar(catv_counts.index, catv_counts.values)

                # Ajouter un titre et des labels
                ax.set_title("Distribution des valeurs de la variable 'catv'")
                ax.set_xlabel("catv")
                ax.set_ylabel("Nombre")
                # Afficher la figure dans Streamlit
                st.pyplot(fig)
                st.write("Nous décidons de regrouper cette variables en 5 catégories: ")
                def replace_catv(catv):
                    if catv in [7,10]:
                        return 1
                    if catv in [2,30,31,32,33,34,35,36,41,42,43]:
                        return 2
                    if catv in [13,14,15]:
                        return 3
                    if catv in [37,38]:
                        return 4
                    else:
                        return 5


                df_vehicules['new_catv'] = df_vehicules['catv'].apply(replace_catv)
                fig, ax = plt.subplots(figsize=(8, 6))

                sns.countplot(data=df_vehicules, x='new_catv',ax = ax,palette="viridis")
                plt.xticks(ticks=[0,1, 2, 3, 4], labels=["VL/VU", "2-3roues&quad", "PL", "Bus/car", "Velo/EDP/autre"],rotation=45);
                plt.title("Distribution du nombre d'accident par type de véhicule regroupés en 5 catégories")
                st.pyplot(fig)
            
            if 'choc' in choix_véhicules:
                st.write("##### Point de choc")
                valeurs_uniques = df_vehicules['choc'].unique()
                st.write ("Les valeurs uniques de la variable 'catr' sont")
                st.write(*valeurs_uniques)
                st.write("-1 – Non renseigné", "0 – Aucun","1 – Avant","2 – Avant droit","3 – Avant gauche","4 – Arrière","5 – Arrière droit","6 – Arrière gauche","7 – Côté droit","8 – Côté gauche","9 – Chocs multiples (tonneaux)")
                st.write("Nous les regroupons en 5 catégories: ' aucun choc', 'choc_AV, 'choc_AR', 'choc_tonneaux','choc_coté")
        
            st.write("Nous mergeons ensuite cette table à la table usagers précédente, et on obtient le dataframe df_usag_veh suivant, qui compte désormais une ligne par accident, soit 273226  lignes")
            st.dataframe(df_merged_usag_veh_final)
            if st.checkbox("Afficher infos et valeurs manquantes df_merged_usag_veh_final)"):
                afficher_infos_dataframe(df_merged_usag_veh_final)


        if "Rubrique LIEUX" in rubriques_selectionnees:
            st.write("### Lieux")
            st.write("La 3ème étape est le travail sur la table lieux")
            st.dataframe(df_lieux)
            if st.checkbox("Afficher infos et valeurs manquantes (LIEUX)"):
                afficher_infos_dataframe(df_lieux)
            
            st.write (" ##### Variables transformées et nettoyées dans ce dataset")
            options = [''] + ['nbv','vma','catr', 'circ', 'surf']
            choix_lieux = st.selectbox ('sélectionner une variable', options)
        
            if 'nbv' in choix_lieux:
                st.write("##### Nombre de voies")
                mode_value = df_total_final['nbv'].mode()[0]
                st.write ('Le mode de la variable nbv est')
                st.write(mode_value)
                df_total_final['nbv']= df_total_final['nbv'].replace([-1, 11,12, '-1', ' -1', 0], mode_value)
                df_total_final['nbv'] = df_total_final['nbv'].replace(['#VALEURMULTI', '-1', '0', '#ERREUR', '11', '12'], mode_value)
                df_total_final['nbv'] = df_total_final['nbv'].replace({'2': 2, '8': 8, '6': 6, '4': 4, '5': 5, '7': 7, '3':3, '1':1, '10':10, '9':9})

                fig, ax = plt.subplots(figsize=(4, 3))  # Taille de la figure

                # Utiliser sns.countplot pour afficher les données sur ax
                sns.countplot(data=df_total_final, x='nbv', ax=ax)

                # Ajouter un titre au graphique
                ax.set_title("Distribution des Nombre de Voies")

                # Afficher la figure sur Streamlit
                st.pyplot(fig)  # Afficher la figure
            
            if 'vma' in choix_lieux:
                st.write("##### Vitesse maximale autorisée")
                valeurs_uniques = df_lieux['vma'].unique()
                st.write ("Les valeurs uniques de la variable 'vma' sont")
                st.write(*valeurs_uniques)
                st.write("Nous constatons qu'il y a énormémement de valeurs aberrantes, que nous décidons de remplacer par le mode de la variable vma")
                st.write('Le mode de la variable vma est')
                mode_vma = df_lieux['vma'].mode()[0]
                st.write(mode_vma)
                df_lieux['vma']=df_lieux['vma'].replace([-1,1,2,31,45,5,15,25,10,40,500,6,35,3,300,900,4,65,700,75,7,12,55,8,0,180,140,770,502,501,9,901,520,600,42,800,560,120,23], df_lieux['vma'].mode()[0])
                st.write('Après transformation, les nouvelles valeurs uniques sont')
                new_mode_vma = df_lieux['vma'].unique()
                st.write(*new_mode_vma)

            if 'catr' in choix_lieux:
                st.write("##### Catégorie de route")
                valeurs_uniques = df_lieux['catr'].unique()
                st.write ("Les valeurs uniques de la variable 'catr' sont")
                st.write(*valeurs_uniques)
                st.write ("  1 – Autoroute","  2 – Route nationale", "  3 – Route Départementale", "  4 – Voie Communales","  5 – Hors réseau public","  6 – Parc de stationnement ouvert à la circulation publique","  7 – Routes de métropole urbaine","  9 – autre")
                st.write ('Nous les regroupons en trois catégories: autoroute, nationale_départementale_communale et autre')
                
            if 'circ' in choix_lieux:
                st.write("##### Sens de circulation")
                valeurs_uniques = df_lieux['circ'].unique()
                st.write ("Les valeurs uniques de la variable 'circ' sont")
                st.write( "-1 – Non renseigné", "1 – A sens unique","2 – Bidirectionnelle","3 – A chaussées séparées", "4 – Avec voies d’affectation variable")
                st.write(*valeurs_uniques)
                st.write ("Nous les regroupons en deux catégories: ' Sens unique', et 'bidirectionnel'")

            if 'surf' in choix_lieux:
                st.write("##### Type de surface")
                data = data = df_lieux["surf"].value_counts().reset_index()
                st.write ("La distribution de la variable 'surf' est")
                st.dataframe(data)
                st.write ("-1 – Non renseigné", '1 – Normale', "2 – Mouillée","3 – Flaques", "4 – Inondée","5 – Enneigée", "6 – Boue"," 7 – Verglacée","8 – Corps gras-huile", "9 – Autre")
                st.write ('Nous les regroupons en trois catégories: #1 normale, #2 mouillée/enneigee , #3 autre')
                
            st.write ("On obtient le dataframe lieux final suivant")
            st.dataframe(df_lieux_final)
            if st.checkbox("Afficher infos et valeurs manquantes df_lieux_final)"):
                afficher_infos_dataframe(df_lieux_final)
        
        if "Rubrique CARACTÉRISTIQUES" in rubriques_selectionnees:
            st.write("### Caractéristiques")
            st.write("Nous travaillons enfin sur la table caractéristiques qui contient une ligne par accident, soit 273226 lignes")
            st.dataframe(df_caracteristiques)
            if st.checkbox("Afficher infos et valeurs manquantes (CARACTÉRISTIQUES)"):
                    afficher_infos_dataframe(df_caracteristiques)

            st.write (" ##### Variables transformées et nettoyées dans ce dataset")
            options = [''] + ['atmosphère']
            choix = st.selectbox ('sélectionner une variable', options)
        
            if 'atmosphère' in choix:
                st.write("##### Atmosphère")
                atm_counts = df_total_final.atm.value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))

                # Tracer un graphique en barres
                ax.bar(atm_counts.index, atm_counts.values)

                # Ajouter un titre et des labels
                ax.set_title("Distribution des valeurs de la variable atmosphère")
                ax.set_xlabel("Atmosphère")
                ax.set_ylabel("Nombre")

                # Afficher la figure dans Streamlit
                st.pyplot(fig)

                def regrouper(val):
                    if val in [1]:
                        return 'temps_normal'
                    elif val in [2, 3]:
                        return 'Temps_pluvieux'
                    elif val in [8]:
                        return 'Temps_couvert'
                    else:
                        return 'Autre'

                df_total_final['atm'] = df_total_final['atm'].apply(regrouper)
                valeurs_uniques = df_total_final.atm.unique()
                st.write('Après regroupement, les valeurs uniques de la variable atmosphère sont les suivantes' )
                st.write(*valeurs_uniques)
            
            st.write("Nous mergeons ensuite les dataframe lieux et caractéristiques, et obtenons le dataframe df_lieux_carac suivant")
            st.dataframe(df_carac_lieux)
            if st.checkbox("Afficher infos et valeurs manquantes df_carac_lieux)"):
                afficher_infos_dataframe(df_carac_lieux)
        
        
# Elaboration dataset      

    if elaboration_dataset_checkbox:
        st.header("Elaboration du dataset final")
        st.write("Nous pouvons enfin merger les dataframes df_usag_veh et df_carac _lieux pour obtenir le dataframe final")
        # Afficher le dataframe des 20 premières lignes
        st.write("### Aperçu du Dataset Final")
        st.dataframe(df_total_final.head(20))
        st.write("Beaucoup de variables ont été encodées lors de la fusion des 4 dataframes, nous pourrons donc supprimer les colonnes 'variable_non_déterminé' pour les exclure sans perdre les autres inforamtions des accidents, usagers, véhicules")


        

        

######################################### Page Datavisualisation et ses sous rubriques ################################
elif st.session_state.page == "Data Visualisation":
    st.header('Data Visualisation')
    st.markdown("""
<div style="background-color: #e8f5e9; border: 2px solid #81c784; padding: 20px; border-radius: 10px;">


L’objectif de cette phase est de <strong>mieux comprendre les données</strong> à notre disposition et d’évaluer l’influence de chaque variable sur la <strong>gravité des accidents</strong>. Il s’agit d’identifier les <strong>facteurs les plus déterminants</strong> afin d’orienter au mieux la construction de notre modèle prédictif.

</div>
""", unsafe_allow_html=True)


# Analyse de la variable gravité
    if distribution_gravité_checkbox:
        st.subheader('Gravité des accidents')
        st.title("Analyse de la gravité des accidents de circulation (2019-2023)")

        # Calcul des fréquences
        frequences = df['grav_max'].value_counts(normalize=True) * 100
        frequences_df = frequences.reset_index()
        frequences_df.columns = ['Gravité', 'Pourcentage']

        # Création du graphique en barres
        fig = px.bar(
            frequences_df,
            x='Gravité',
            y='Pourcentage',
            text='Pourcentage',
            color='Gravité',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#d62728']
        )

        # Mise à jour des annotations
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside',
            width=0.5
        )

        # Mise à jour de la mise en page
        fig.update_layout(
            title="Répartition de la gravité des accidents de circulation corporels en France (2019-2023)",
            xaxis_title="Gravité",
            yaxis_title="Pourcentage (%)",
            yaxis_tickformat=".2f",
            font=dict(
                size=14,
                family="Arial"
            ),
            plot_bgcolor='rgb(245,245,245)',
            xaxis=dict(
                showgrid=False,
                title_font=dict(size=14),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgrey',
                title_font=dict(size=14)
            ),
            title_font=dict(size=16),
            showlegend=False,
            width=1000,
            height=600
        )

        # Affichage du graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.write(" Nous observons un déséquilibre de la répartition des valeurs de la variable cible (gravité de l'accident), qu'il faudra garder en tête pour la phase de modélisation")
        st.write("Nous constatons également qu’il n’y a aucun accident présentant la gravité 'indemne', ce qui signifie que pour chaque accident répertorié, il y a au moins un blessé.")


# Conditions accident
    if conditions_accident_checkbox:
        st.header('Contexte temporel')
        variables_conditions_accident = st.radio(
        "Sélectionnez une variable :",
        ["Tendances annuelles", "Variations Saisonnières Mensuelles", "Rythme Hebdomadaire", "Fluctuations horaires par jour de la semaine"])
       
        if "Tendances annuelles" in variables_conditions_accident:
            st.subheader("Tendances annuelles")

            # Préparation des données
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            data = df.groupby(df['date'].dt.year).agg(Nombre=('Num_Acc', 'nunique')).reset_index().rename(columns={'date': 'annee'})

            # Calcul de la variation en % du Nombre
            variations = [None] + [
                f"\u2191 {delta:+.1f}%" if delta > 0 else f"\u2193 {delta:+.1f}%" if delta < 0 else "0.0%"
                for delta in [
                    ((data['Nombre'][i] - data['Nombre'][i-1]) / data['Nombre'][i-1] * 100)
                    if data['Nombre'][i-1] != 0 else 0
                    for i in range(1, len(data))
                ]
            ]
            data['Variation'] = variations

            # Création du graphique
            fig = px.bar(
                data,
                x='annee',
                y='Nombre',
                text='Nombre',
                color_discrete_sequence=['#003DA5']
            )

            # Personnalisation des étiquettes
            fig.update_traces(
                texttemplate=[
                    f"%{{y:,.0f}} <br><span style='color:red'>{variation}</span>" if '↑' in (variation or '') else
                    f"%{{y:,.0f}} <br><span style='color:green'>{variation}</span>" if '↓' in (variation or '') else
                    f"%{{y:,.0f}} "
                    for variation in data['Variation']
                ],
                textposition='outside',
                textfont=dict(color="black"),
                width=0.6
            )

            # Mise à jour du layout
            fig.update_layout(
                title="Évolution du Nombre d'accidents de circulation corporels en France (2019-2023)",
                title_font=dict(size=16),
                xaxis_title="Année",
                yaxis_title="Nombre d'accidents",
                xaxis=dict(title_font=dict(size=14)),
                yaxis=dict(title_font=dict(size=14)),
                yaxis_tickformat=",.0f",
                showlegend=False,
                width=1000,
                height=550
            )

            # Affichage dans Streamlit
            st.plotly_chart(fig, use_container_width=True)

       

            # 1. S'assurer que la date est bien au format datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['Annee'] = df['date'].dt.year

            # 2. Définir l'ordre des gravités
            modalites_grav = ['Blessé léger', 'Blessé hospitalisé', 'Décès']
            df['grav_max'] = pd.Categorical(df['grav_max'], categories=modalites_grav, ordered=True)

            # 3. Groupement par année et gravité
            data_an = df.groupby(['Annee', 'grav_max'], observed=False).agg(Nombre=('Num_Acc', 'nunique')).reset_index()

            # 4. Renommer pour clarté
            data_an.rename(columns={'grav_max': 'Gravité'}, inplace=True)

            # 5. Calcul des totaux par année
            totaux_par_annee = data_an.groupby('Annee')['Nombre'].transform('sum')

            # 6. Fréquences relatives
            data_an['Fréquence'] = 0.0
            mask_total_non_zero = totaux_par_annee > 0
            data_an.loc[mask_total_non_zero, 'Fréquence'] = \
                (data_an.loc[mask_total_non_zero, 'Nombre'] / totaux_par_annee[mask_total_non_zero] * 100)

            # 7. Création du graphique
            fig = px.bar(
                data_an,
                x='Annee',
                y='Nombre',
                color='Gravité',
                text='Fréquence',
                color_discrete_map={
                    'Blessé léger': '#1f77b4',
                    'Blessé hospitalisé': '#ff7f0e',
                    'Décès': '#d62728'
                },
                category_orders={"Gravité": modalites_grav},
                title="Répartition Annuelle des Accidents (Distincts) par Gravité",
                labels={
                    'Nombre': "Nombre d'accidents distincts",
                    'Fréquence': 'Fréquence relative (%)',
                    'Annee': 'Année'
                }
            )

            # 8. Personnalisation des étiquettes
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='inside',
                insidetextanchor='middle',
                textfont_color='white'
            )

            # 9. Mise en page
            fig.update_layout(
                barmode='stack',
                xaxis_title="Année",
                yaxis_title="Nombre d'accidents distincts",
                xaxis=dict(
                    title_font=dict(size=14),
                    tickmode="linear",
                    dtick=1
                ),
                yaxis=dict(title_font=dict(size=14)),
                yaxis_tickformat=",.0f",
                height=600,
                legend_title_text='Gravité',
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )

            # 10. Affichage dans Streamlit
            st.plotly_chart(fig, use_container_width=True)

        if "Variations Saisonnières Mensuelles" in variables_conditions_accident:
            st.subheader("Variations Saisonnières Mensuelles")

            # --- Préparation des données ---
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['annee'] = df['date'].dt.year
            df['mois_num'] = df['date'].dt.month.astype(int)
            mois_mapping = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            df['mois'] = df['mois_num'].map(mois_mapping)

            modalites_grav = ['Blessé léger', 'Blessé hospitalisé', 'Décès']
            couleurs = {
                'Blessé léger': '#1f77b4',
                'Blessé hospitalisé': '#ff7f0e',
                'Décès': '#d62728'
            }

            # Vue globale
            total_mensuel = df.groupby(['mois_num', 'mois'])['Num_Acc'].nunique().reset_index()
            total_mensuel.columns = ['Mois_Num', 'Mois', 'Total_Accidents_Global']

            global_data = df.groupby(['mois_num', 'mois', 'grav_max'])['Num_Acc'].nunique().reset_index()
            global_data.columns = ['Mois_Num', 'Mois', 'Gravite', 'Nombre']
            global_data = global_data.merge(total_mensuel, on=['Mois_Num', 'Mois'])
            global_data['Pourcentage'] = global_data.apply(lambda row: (row['Nombre'] / row['Total_Accidents_Global']) * 100 if row['Total_Accidents_Global'] > 0 else 0, axis=1)

            annees = sorted(df['annee'].dropna().unique())

            # --- Construction de la figure ---
            fig = go.Figure()

            # Traces globales
            for grav in modalites_grav:
                subset_global = global_data[global_data['Gravite'] == grav]
                if not subset_global.empty:
                    fig.add_trace(go.Bar(
                        x=subset_global['Mois'],
                        y=subset_global['Nombre'],
                        name=f"{grav} - Global",
                        customdata=subset_global['Pourcentage'],
                        text=subset_global['Pourcentage'],
                        texttemplate='%{text:.2f}%',
                        textposition='inside',
                        marker_color=couleurs.get(grav),
                        hovertemplate=(
                            "<b>Mois :</b> %{x}<br>" +
                            "<b>Gravité :</b> " + grav + "<br>" +
                            "<b>Nombre :</b> %{y}<br>" +
                            "<b>Pourcentage :</b> %{customdata:.2f}%<extra></extra>"
                        ),
                        visible=True
                    ))

            # Traces par année
            for annee in annees:
                df_annee = df[df['annee'] == annee]
                total_annuel = df_annee.groupby(['mois_num', 'mois'])['Num_Acc'].nunique().reset_index()
                total_annuel.columns = ['Mois_Num', 'Mois', 'Total_Accidents_Annee']

                for grav in modalites_grav:
                    data_grav = df_annee[df_annee['grav_max'] == grav]
                    acc_count = data_grav.groupby(['mois_num', 'mois'])['Num_Acc'].nunique().reset_index()
                    acc_count.columns = ['Mois_Num', 'Mois', 'Nombre']
                    acc_count = acc_count.merge(total_annuel, on=['Mois_Num', 'Mois'], how='left')
                    acc_count['Pourcentage'] = acc_count.apply(lambda row: (row['Nombre'] / row['Total_Accidents_Annee']) * 100 if row['Total_Accidents_Annee'] > 0 else 0, axis=1)

                    if not acc_count.empty:
                        fig.add_trace(go.Bar(
                            x=acc_count['Mois'],
                            y=acc_count['Nombre'],
                            name=f"{grav} - {annee}",
                            customdata=acc_count['Pourcentage'],
                            text=acc_count['Pourcentage'],
                            texttemplate='%{text:.2f}%',
                            textposition='inside',
                            marker_color=couleurs.get(grav),
                            hovertemplate=(
                                "<b>Mois :</b> %{x}<br>" +
                                f"<b>Année :</b> {annee}<br>" +
                                "<b>Gravité :</b> " + grav + "<br>" +
                                "<b>Nombre :</b> %{y}<br>" +
                                "<b>Pourcentage :</b> %{customdata:.2f}%<extra></extra>"
                            ),
                            visible=False
                        ))

            # --- Slider ---
            steps = []
            nb_traces_par_vue = len(modalites_grav)
            for i, annee in enumerate(['Global'] + list(annees)):
                visible = [False] * len(fig.data)
                for j in range(nb_traces_par_vue):
                    index = i * nb_traces_par_vue + j
                    if index < len(fig.data):
                        visible[index] = True
                steps.append(dict(
                    method="update",
                    args=[{"visible": visible},
                        {"title": f"Évolution mensuelle des accidents corporels - {annee}"}],
                    label=str(annee)
                ))

            fig.update_layout(
                sliders=[dict(
                    active=0,
                    currentvalue={"prefix": "Vue : "},
                    pad={"t": 50},
                    steps=steps
                )],
                barmode='stack',
                title= dict(text="Évolution mensuelle des accidents corporels",font=dict(size=16)),
                xaxis=dict(
                    title="Mois",
                    categoryorder='array',
                    categoryarray=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                ),
                yaxis=dict(
                    title="Nombre d'accidents",
                    tickformat=",.0f"
                ),
                height=650,
                legend_title_text="Gravité - Vue",
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )

            # --- Affichage dans Streamlit ---
            st.plotly_chart(fig, use_container_width=True)

           

            df['date'] = pd.to_datetime(df['date'])
            df['annee'] = df['date'].dt.year
            df['mois_num'] = df['date'].dt.month.astype(int)

            mois_mapping = {
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }
            df['mois'] = df['mois_num'].map(mois_mapping)

            # --- Préparation des données ---
            total_mensuel = df.groupby(['mois_num', 'mois'])['Num_Acc'].nunique().reset_index()
            total_mensuel.columns = ['Mois_Num', 'Mois', 'Total_Accidents_Global']

            global_data = df.groupby(['mois_num', 'mois', 'grav_max'])['Num_Acc'].nunique().reset_index()
            global_data.columns = ['Mois_Num', 'Mois', 'Gravite', 'Nombre']
            global_data = global_data.merge(total_mensuel, on=['Mois_Num', 'Mois'])
            global_data['Pourcentage'] = (global_data['Nombre'] / global_data['Total_Accidents_Global']) * 100

            annees = sorted(df['annee'].unique())
            modalites_grav = ['Blessé léger', 'Blessé hospitalisé', 'Décès']
            couleurs = {
                'Blessé léger': '#1f77b4',
                'Blessé hospitalisé': '#ff7f0e',
                'Décès': '#d62728'
            }

            # --- Création de la figure ---
            fig = go.Figure()

            # Traces globales visibles par défaut
            for grav in modalites_grav:
                subset_global = global_data[global_data['Gravite'] == grav]
                fig.add_trace(go.Scatter(
                    x=subset_global['Mois'],
                    y=subset_global['Nombre'],
                    mode='lines+markers',
                    name=f"{grav} - Global",
                    hovertemplate=(
                        "<b>Mois :</b> %{x}<br>" +
                        "<b>Gravité :</b> " + grav + "<br>" +
                        "<b>Nombre :</b> %{y}<br>" +
                        "<b>Pourcentage :</b> %{customdata:.2f}%<extra></extra>"
                    ),
                    customdata=subset_global['Pourcentage'],
                    line=dict(color=couleurs[grav]),
                    visible=True
                ))

            # Traces par année masquées par défaut
            for annee in annees:
                subset_annee = df[df['annee'] == annee]
                for grav in modalites_grav:
                    data_grav = subset_annee[subset_annee['grav_max'] == grav]
                    accident_count = data_grav.groupby(['mois_num', 'mois'])['Num_Acc'].nunique().reset_index()
                    accident_count.columns = ['Mois_Num', 'Mois', 'Nombre']
                    accident_count['Pourcentage'] = (accident_count['Nombre'] / accident_count['Nombre'].sum()) * 100

                    fig.add_trace(go.Scatter(
                        x=accident_count['Mois'],
                        y=accident_count['Nombre'],
                        mode='lines+markers',
                        name=f"{grav} - {annee}",
                        hovertemplate=(
                            "<b>Mois :</b> %{x}<br>" +
                            "<b>Gravité :</b> " + grav + "<br>" +
                            "<b>Nombre :</b> %{y}<br>" +
                            "<b>Pourcentage :</b> %{customdata:.2f}%<extra></extra>"
                        ),
                        customdata=accident_count['Pourcentage'],
                        line=dict(color=couleurs[grav]),
                        visible=False
                    ))

            # --- Slider pour changer de vue ---
            steps = []

            # Étape globale
            steps.append(dict(
                method="update",
                args=[{
                    "visible": [True] * len(modalites_grav) + [False] * (len(annees) * len(modalites_grav))
                }, {
                    "title": "Évolution mensuelle des accidents corporels - Vue globale (2019-2023)"
                }],
                label="Global"
            ))

            # Étapes par année
            for i, annee in enumerate(annees):
                visibilities = [False] * len(fig.data)
                start_index = len(modalites_grav) + i * len(modalites_grav)
                for j in range(len(modalites_grav)):
                    visibilities[start_index + j] = True
                steps.append(dict(
                    method="update",
                    args=[{"visible": visibilities},
                        {"title": f"Évolution mensuelle des accidents corporels - Année {annee}"}],
                    label=str(annee)
                ))

            # Configuration du slider
            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Vue : "},
                pad={"t": 50},
                steps=steps
            )]

            # Layout final
            fig.update_layout(
                sliders=sliders,
                title=dict(
                    text="Évolution des accidents corporels en France par gravité avec segmentation annuelle",
                    font=dict(size=16)
                ),
                xaxis=dict(
                    title="Mois",
                    categoryorder='array',
                    categoryarray=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ),
                yaxis=dict(
                    title="Nombre d'accidents",
                    tickformat=",.0f"
                ),
                legend=dict(title="Gravité"),
                height=600
            )

            # --- Affichage dans Streamlit ---
            st.plotly_chart(fig, use_container_width=True)

        if "Rythme hebdomadaire" in variables_conditions_accident:
            st.subheader("Rythme hebdomadaire")

            # Extraire le jour de la semaine
            jours_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            df['jour'] = df['date'].dt.day_name().replace({
                'Monday': 'Lundi', 'Tuesday': 'Mardi', 'Wednesday': 'Mercredi',
                'Thursday': 'Jeudi', 'Friday': 'Vendredi', 'Saturday': 'Samedi',
                'Sunday': 'Dimanche'})

            df['jour'] = pd.Categorical(df['jour'], categories=jours_order, ordered=True)

            # Groupement et agrégation
            data_sem = df.groupby(['jour', 'grav_max']).agg(Nombre=('Num_Acc', 'nunique')).reset_index()
            data_sem.rename(columns={'grav_max': 'Gravité'}, inplace=True)

            # Calculs pour les fréquences
            totaux_par_jour = data_sem.groupby('jour')['Nombre'].transform('sum')
            data_sem['Fréquence'] = (data_sem['Nombre'] / totaux_par_jour * 100).round(2)
            data_sem = data_sem.sort_values(by=['jour', 'Nombre'], ascending=[True, False])

            # Affichage du graphique
            fig = px.bar(
                data_sem,
                x='jour',
                y='Nombre',
                color='Gravité',
                text='Fréquence',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#d62728'],
                title="Répartition Hebdomadaire des Accidents de Circulation par Gravité",
                labels={'Nombre': "Nombre d'accidents", 'Fréquence': 'Fréquence relative (%)'}
            )

            fig.update_traces(texttemplate='%{text}%', textposition='outside')

            fig.update_layout(
                xaxis_title="Jour de la Semaine",
                yaxis_title="Nombre d'accidents",
                xaxis=dict(title_font=dict(size=14)),
                yaxis=dict(title_font=dict(size=14), tickformat=",.0f"),
                height=620,
                showlegend=True
            )

            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig, use_container_width=True)

        if "Fluctuations horaires par jour de la semaine" in variables_conditions_accident:
            st.subheader('Fluctuations horaires par jour de la semaine')

                        # --- Préparation des données ---
            data = df[['Num_Acc', 'grav_max', 'date']].copy()
            data['hrmn'] = df['date'].dt.hour

            # Extraire le jour de la semaine (Lundi = 0, Dimanche = 6)
            data['jour_semaine'] = data['date'].dt.dayofweek

            # Supprimer les lignes avec heures nulles
            data = data.dropna(subset=['date'])

            # Extraire le jour de la semaine
            data['jour_semaine'] = data['date'].dt.dayofweek

            jours_de_la_semaine = {
                0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi',
                4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'
            }
            data['jour_semaine'] = data['jour_semaine'].map(jours_de_la_semaine)

            # Global : toutes journées confondues
            global_data = data.groupby(['hrmn', 'grav_max']).agg(Nombre=('Num_Acc', 'nunique')).reset_index()
            global_data['jour_semaine'] = 'Global'

            # Par jour
            accidents_count = data.groupby(['hrmn', 'grav_max', 'jour_semaine']).agg(Nombre=('Num_Acc', 'nunique')).reset_index()

            # Fusion avec la vue globale
            accidents_count = pd.concat([global_data, accidents_count], ignore_index=True)

            # Calcul des pourcentages
            accidents_count['Pourcentage'] = accidents_count.groupby(['jour_semaine', 'hrmn'])['Nombre'].transform(
                lambda x: x / x.sum() * 100)

            # Ordre d'affichage des jours
            categories_order = ['Global', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            accidents_count['jour_semaine'] = pd.Categorical(accidents_count['jour_semaine'], categories=categories_order, ordered=True)

            # --- Graphique ---
            fig = px.line(
                accidents_count,
                x='hrmn',
                y='Nombre',
                color='grav_max',
                markers=True,
                labels={'hrmn': 'Heure de la journée', 'Nombre': 'Nombre d\'accidents'},
                title="Évolution du nombre d'accidents par heure de la journée",
                color_discrete_map={
                    'Blessé léger': '#1f77b4',
                    'Blessé hospitalisé': '#ff7f0e',
                    'Décès': '#d62728'
                },
                animation_frame='jour_semaine',
                category_orders={'jour_semaine': categories_order}
            )

            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[0, 23], title_font=dict(size=14)),
                yaxis=dict(showgrid=False, title="Nombre d'accidents", title_font=dict(size=14)),
                xaxis_title="Heure de la journée",
                yaxis_tickformat=",.0f",
                title_font=dict(size=16),
                font=dict(size=14, family="Arial"),
                plot_bgcolor='rgb(245,245,245)',
                legend=dict(title_font=dict(size=14)),
                legend_title="Gravité",
                width=1000,
                height=600
            )

            # Affichage dans Streamlit
            st.plotly_chart(fig, use_container_width=True)

          
    # Lieux
    if localisation_checkbox:
        st.subheader('Facteurs géographiques et infrastructurels')
        variables_lieux = st.radio("Sélectionnez une variable :",["Vitesse maximale autorisée", "Nombre de voies", "Catégorie de la route", "Localisation de l'accident"])

        if "Vitesse maximale autorisée" in variables_lieux:
            st.write("Vitesse maximale autorisée")

                # --- Étape 1 : Préparation des données ---
            data_vma = df[['Num_Acc','grav_max','vma']].copy()

            # Étape 2 : Filtrage des VMA d'intérêt
            vma_to_include = [30, 50, 70, 80, 90]
            df_filtered = data_vma[data_vma['vma'].isin(vma_to_include)]

            # Étape 3 : Agrégation
            agg_data = df_filtered.groupby(['vma', 'grav_max'])['Num_Acc'].nunique().reset_index()
            agg_data.rename(columns={'Num_Acc': 'Nombre_Accidents_Uniques'}, inplace=True)

            # Étape 4 : Totaux par VMA
            total_par_vma = df_filtered.groupby('vma')['Num_Acc'].nunique().reset_index()
            total_par_vma.rename(columns={'Num_Acc': 'Total_Accidents_VMA'}, inplace=True)

            # Étape 5 : Pourcentages
            agg_data = pd.merge(agg_data, total_par_vma, on='vma')
            agg_data['Pourcentage'] = agg_data.apply(
                    lambda row: (row['Nombre_Accidents_Uniques'] / row['Total_Accidents_VMA']) * 100 if row['Total_Accidents_VMA'] > 0 else 0,
                    axis=1
                )

            # Étape 6 : Heatmap DataFrames
            heatmap_percentages = agg_data.pivot_table(
                index='grav_max',
                    columns='vma',
                    values='Pourcentage',
                    fill_value=0
                )

            heatmap_counts = agg_data.pivot_table(
                    index='grav_max',
                    columns='vma',
                    values='Nombre_Accidents_Uniques',
                    fill_value=0
                )

                # Étape 7 : Ordres
            vma_order_xaxis = total_par_vma.sort_values(by='Total_Accidents_VMA', ascending=False)['vma'].tolist()
            grav_order_yaxis = ['Blessé léger', 'Blessé hospitalisé', 'Décès']
            grav_order_yaxis = [grav for grav in grav_order_yaxis if grav in heatmap_percentages.index]

            heatmap_percentages = heatmap_percentages.reindex(index=grav_order_yaxis, columns=vma_order_xaxis, fill_value=0)
            heatmap_counts = heatmap_counts.reindex(index=grav_order_yaxis, columns=vma_order_xaxis, fill_value=0)

                # --- Étape 8 : Création du graphique ---
            fig = px.imshow(
                    heatmap_percentages,
                    labels=dict(x="Vitesse (km/h)", y="Gravité", color="Pourcentage"),
                    title="Pourcentage d'accidents par gravité au sein de chaque VMA",
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='YlGnBu'
                )

                # Ajout des données custom & formatage du hover
            fig.update_traces(
                    customdata=heatmap_counts.values,
                    texttemplate='%{z:.2f}%',
                    hovertemplate=(
                        "<b>Vitesse:</b> %{x} km/h<br>" +
                        "<b>Gravité:</b> %{y}<br>" +
                        "<b>Pourcentage:</b> %{z:.2f}%<br>" +
                        "<b>Nombre d'accidents:</b> %{customdata:,.0f}<extra></extra>"
                    )
                )

                # Layout final
            fig.update_layout(
                    xaxis=dict(title='Vitesse Maximale Autorisée (km/h)', type='category'),
                    yaxis=dict(title='Gravité'),
                    title_font_size=18,
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    coloraxis_colorbar=dict(title="Pourcentage"),
                    height=600
                )

                # Affichage
            st.plotly_chart(fig, use_container_width=True)
            
        if "Nombre de voies" in variables_lieux:
            st.write("Nombre de voies")
            # --- 1. Chargement et Préparation Initiale ---
            data_nbv = df[['Num_Acc', 'grav_max', 'nbv']].copy()

                # --- 2. Filtrage et Agrégation ---
            nbv_to_include = [2, 4, 1, 3, 6, 0, 5]
            df_filtered = data_nbv[data_nbv['nbv'].isin(nbv_to_include)]
            agg_data = df_filtered.groupby(['nbv', 'grav_max'])['Num_Acc'].nunique().reset_index()
            agg_data.rename(columns={'Num_Acc': 'Nombre_Accidents_Uniques'}, inplace=True)

                # --- 3. Calcul du Total par NBV ---
            total_par_nbv = df_filtered.groupby('nbv')['Num_Acc'].nunique().reset_index()
            total_par_nbv.rename(columns={'Num_Acc': 'Total_Accidents_NBV'}, inplace=True)

                # --- 4. Calcul des Pourcentages ---
            agg_data = pd.merge(agg_data, total_par_nbv, on='nbv')
            agg_data['Pourcentage'] = agg_data.apply(
                    lambda row: (row['Nombre_Accidents_Uniques'] / row['Total_Accidents_NBV']) * 100 if row['Total_Accidents_NBV'] > 0 else 0,
                    axis=1
                )

                # --- 5. Préparation des Données pour le Heatmap ---
            heatmap_percentages = agg_data.pivot_table(
                    index='grav_max', columns='nbv', values='Pourcentage', fill_value=0
                )
            heatmap_counts = agg_data.pivot_table(
                    index='grav_max', columns='nbv', values='Nombre_Accidents_Uniques', fill_value=0
                )

                # --- 6. Définition de l'Ordre des Axes et des Libellés ---
            total_par_nbv_sorted = total_par_nbv.sort_values(by='Total_Accidents_NBV', ascending=False)
            nbv_order_xaxis = total_par_nbv_sorted['nbv'].tolist()
            grav_order_yaxis = ['Blessé léger', 'Blessé hospitalisé', 'Décès']
            grav_order_yaxis = [grav for grav in grav_order_yaxis if grav in heatmap_percentages.index]

            heatmap_percentages = heatmap_percentages.reindex(index=grav_order_yaxis, columns=nbv_order_xaxis, fill_value=0)
            heatmap_counts = heatmap_counts.reindex(index=grav_order_yaxis, columns=nbv_order_xaxis, fill_value=0)

            nbv_label_mapping = {
                    2: '2 Voies',
                    4: '4 Voies',
                    1: '1 Voie',
                    3: '3 Voies',
                    6: '6 Voies',
                    0: '0 Voie',
                    5: '5 Voies'
                }
            ticktext_labels = [nbv_label_mapping.get(nbv, str(nbv)) for nbv in nbv_order_xaxis]

                # --- 7. Création du Heatmap ---
            fig = px.imshow(
                    heatmap_percentages,
                    labels=dict(x="Nombre de Voies", y="Gravité", color="Pourcentage"),
                    title="Pourcentage d'Accidents par Gravité selon le Nombre de Voies",
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='YlGnBu'
                )

                # --- 8. Mise en Forme Finale du Heatmap ---
            fig.update_traces(
                    customdata=heatmap_counts.values,
                    texttemplate='%{z:.2f}%',
                    hovertemplate=(
                        "<b>Nombre de Voies:</b> %{x}<br>" +
                        "<b>Gravité:</b> %{y}<br>" +
                        "<b>Pourcentage:</b> %{z:.2f}%<br>" +
                        "<b>Nombre Accidents:</b> %{customdata:,.0f}<extra></extra>"
                    )
                )

            fig.update_layout(
                    xaxis=dict(
                        title='Nombre de Voies',
                        type='category',
                        tickvals=nbv_order_xaxis,
                        ticktext=ticktext_labels
                    ),
                    yaxis=dict(title='Gravité'),
                    title_font_size=18,
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    coloraxis_colorbar=dict(title="Pourcentage")
                )
            fig.update_xaxes(side="bottom")

                # --- Affichage dans Streamlit ---
            st.plotly_chart(fig, use_container_width=True)

            st.write("Le constat le plus alarmant de l'analyse concerne les routes à 2 voies. Celles-ci présentent, de manière très marquée, la proportion la plus élevée d'accidents aux conséquences graves ou mortelles. Le taux de décès y atteint un record de 7.02 %, ce qui est 1.5 à près de 3 fois supérieur aux autres types de routes. De plus, la proportion de blessés nécessitant une hospitalisation y est écrasante, s'élevant à 36.49 %, soit presque le double de la plupart des autres configurations. En combinant ces deux issues, plus de 43 % des accidents survenant sur les routes à 2 voies entraînent une hospitalisation ou un décès, constituant un signal d'alarme majeur quant à la dangerosité spécifique de ces infrastructures.")

        if "Catégorie de la route " in variables_lieux:
                
            st.write("##### Analyse de la Gravité des Accidents par Catégorie de Route")

                # --- 0. Chargement des données ---
                # À personnaliser selon ta source
            df = pd.read_csv("accidents.csv")  # Remplace par ton fichier réel

                # --- 1. Préparation Initiale ---
            data_catr = df[['Num_Acc', 'grav_max', 'catr']].copy()

                # Mapping des libellés
            catr_label_mapping = {
                    4: 'Communale', 3: 'Departementale', 1: 'Autoroute',
                    2: 'Nationale', 7: 'Route Urbaine', 6: 'Parc Stationnement',
                    5: 'Hors réseau public', 9: 'Autres_catr'
                }

                # Conversion et mapping
            data_catr['catr'] = pd.to_numeric(data_catr['catr'], errors='coerce')
            data_catr['catr'] = data_catr['catr'].map(catr_label_mapping)
            data_catr.dropna(subset=['catr'], inplace=True)

            if not data_catr.empty:

                    # --- 2. Agrégation ---
                agg_data = data_catr.groupby(['catr', 'grav_max'])['Num_Acc'].nunique().reset_index()
                agg_data.rename(columns={'Num_Acc': 'Nombre_Accidents_Uniques'}, inplace=True)

                    # --- 3. Totaux par catégorie ---
                total_par_catr = data_catr.groupby('catr')['Num_Acc'].nunique().reset_index()
                total_par_catr.rename(columns={'Num_Acc': 'Total_Accidents_CATR'}, inplace=True)

                    # --- 4. Pourcentages ---
                agg_data = pd.merge(agg_data, total_par_catr, on='catr', how='left')
                agg_data['Pourcentage'] = agg_data.apply(
                        lambda row: (row['Nombre_Accidents_Uniques'] / row['Total_Accidents_CATR']) * 100
                        if row['Total_Accidents_CATR'] > 0 else 0, axis=1
                    )
                agg_data['Pourcentage'].fillna(0, inplace=True)

                    # --- 5. Matrices pour le heatmap ---
                heatmap_percentages = agg_data.pivot_table(
                        index='grav_max', columns='catr', values='Pourcentage', fill_value=0
                    )
                heatmap_counts = agg_data.pivot_table(
                        index='grav_max', columns='catr', values='Nombre_Accidents_Uniques', fill_value=0
                    )

                    # --- 6. Ordres (Top 5) ---
                total_par_catr_sorted = total_par_catr.sort_values(by='Total_Accidents_CATR', ascending=False)
                top_5_catr_labels = total_par_catr_sorted['catr'].head(5).tolist()

                grav_order_yaxis = ['Blessé léger', 'Blessé hospitalisé', 'Décès']
                grav_order_yaxis = [g for g in grav_order_yaxis if g in heatmap_percentages.index]

                heatmap_percentages = heatmap_percentages.reindex(index=grav_order_yaxis, columns=top_5_catr_labels, fill_value=0)
                heatmap_counts = heatmap_counts.reindex(index=grav_order_yaxis, columns=top_5_catr_labels, fill_value=0)

                    # --- 7. Customdata pour le hover ---
                total_counts_ordered = total_par_catr_sorted.set_index('catr').loc[top_5_catr_labels]['Total_Accidents_CATR'].values
                num_gravity_levels = len(grav_order_yaxis)
                totals_array = np.tile(total_counts_ordered, (num_gravity_levels, 1))
                specific_counts_array = heatmap_counts.values
                customdata_combined = np.stack([specific_counts_array, totals_array], axis=-1)

                    # --- 8. Création du graphique ---
                fig = px.imshow(
                        heatmap_percentages,
                        labels=dict(x="Catégorie de Route", y="Gravité", color="Pourcentage (%)"),
                        title="Répartition (%) des Accidents par Gravité et Catégorie de Route (Top 5)",
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='YlGnBu'
                    )

                fig.update_traces(
                        customdata=customdata_combined,
                        texttemplate='%{z:.1f}%',
                        hovertemplate=(
                            "<b>Catégorie:</b> %{x}<br>"
                            "<b>Gravité:</b> %{y}<br>"
                            "<b>Pourcentage:</b> %{z:.2f}%<br>"
                            "<b>Nb Accidents (Gravité):</b> %{customdata[0]:,.0f}<br>"
                            "<b>Total Accidents (Catégorie):</b> %{customdata[1]:,.0f}"
                            "<extra></extra>"
                        )
                    )

                fig.update_layout(
                        xaxis=dict(title='Catégorie de Route', type='category'),
                        yaxis=dict(title='Gravité'),
                        title_font_size=18,
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        coloraxis_colorbar=dict(title="Pourcentage (%)")
                    )
                fig.update_xaxes(side="bottom")

                    # --- 9. Affichage Streamlit ---
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Le jeu de données est vide ou aucun 'catr' valide trouvé.")

            st.write (" Les routes départementales et nationales émergent comme les réseaux les plus dangereux, concentrant les taux les plus élevés d'hospitalisation (42.5 % sur Départementales) et de décès (8.9 % sur Nationales), indiquant des conséquences particulièrement sévères lors d'accidents. À l'opposé, les Autoroutes et les Routes Communales présentent une gravité proportionnellement moindre, avec les taux de décès et d'hospitalisation les plus bas, malgré les vitesses élevées possibles sur autoroute")
       
        if "Localisation de l'accident" in variables_lieux:
            @st.cache_data
            def process_accident_data(df):
                # Transformation des coordonnées
                df['lat'] = df['lat'].astype('str').apply(lambda x: x.replace('\t', '.').replace(',', '.'))
                df['lat'] = df['lat'].astype('float').round(5)
                df['long'] = df['long'].astype('str').apply(lambda x: x.replace('\t', '.').replace(',', '.'))
                df['long'] = df['long'].astype('float').round(5)

                # Sélection des lignes avec des départements numériques
                df_dep_num = df.loc[df['dep'].astype('str').apply(lambda x: x.isnumeric())].astype('str')

                # Sélection des lignes avec des départements non numériques (Corse)
                df_dep_corse = df.loc[df['dep'].astype('str').apply(lambda x: x.isnumeric() == False)].astype('str')

                # Sélectionner les accidents en métropole
                df_m = df[(df['dep'].isin(df_dep_num.dep.unique())) | (df['dep'].isin(df_dep_corse.dep.unique()))]

                # Supprimer les accidents avec des coordonnées nulles ou manquantes
                df_m = df_m[(df_m['lat'] != 0.0) & (df_m['long'] != 0.0)].dropna(subset=['lat', 'long'], axis=0)

                # Créer une géométrie Point pour chaque paire de coordonnées (longitude, latitude)
                geometry = [Point(xy) for xy in zip(df_m['long'], df_m['lat'])]

                # Créer un GeoDataFrame à partir des données et de la géométrie
                geo_df = gpd.GeoDataFrame(df_m, geometry=geometry)

                return geo_df

            # Fonction pour afficher la carte avec cache
            @st.cache_data
            def plot_accident_map(_geo_df):
                # Créer une figure et des axes pour la carte
                figure, ax = plt.subplots(figsize=(13, 12))

                # Masquer les axes pour une apparence plus propre
                plt.axis('off')

                # Définir les limites des axes pour inclure toute la France
                ax.set_xlim([-5.5, 9.5])  # Longitude (Ouest-Est)
                ax.set_ylim([41.0, 51.5])  # Latitude (Sud-Nord)

                # Tracer les accidents sur la carte en utilisant différentes couleurs pour chaque gravité
                geo_df[geo_df['gravité_accident'] == 2].plot(ax=ax, markersize=5, color='green', label='Blessés légers')
                geo_df[geo_df['gravité_accident'] == 3].plot(ax=ax, markersize=5, color='orange', label='Blessés graves')
                geo_df[geo_df['gravité_accident'] == 4].plot(ax=ax, markersize=5, color='red', label='Tués')

                # Titre et légende
                plt.title('Représentation de la gravité des accidents en France entre 2019 et 2023')
                plt.legend()

                # Retourner la figure pour affichage
                return figure

            # Traitement des données avec cache
            geo_df = process_accident_data(df_total_final)

            # Affichage dans Streamlit
            st.write("### Carte de la répartition des accidents en France par gravité")
            fig = plot_accident_map(geo_df)

            # Affichage de la carte avec Streamlit
            st.pyplot(fig)

            st.write("On observe une forte concentration d’accidents dans les zones densément peuplées, notamment en Île-de-France, autour de Lyon, Marseille, Lille ou encore Toulouse. Cette densité est cohérente avec l’intensité du trafic routier dans les agglomérations.")
          
            
            def plot_accidents_by_agg_and_gravite(df):
                # 1. S'assurer que grav_max est bien en chaîne
                df['grav_max'] = df['grav_max'].astype(str)

                # 2. Nettoyage et mapping de la variable 'agg'
                agg_map = {2: 'En agglomération', 1: 'Hors agglomération'}
                df['agg_label'] = df['agg'].map(agg_map)

                # 3. Supprimer les lignes avec valeurs manquantes
                df.dropna(subset=['grav_max', 'agg_label'], inplace=True)

                # 4. Agrégation par gravité et localisation
                agg_data = df.groupby(['grav_max', 'agg_label'])['Num_Acc'].nunique().reset_index()
                agg_data.rename(columns={'Num_Acc': 'Nombre_Accidents'}, inplace=True)

                # 5. Calcul des proportions
                total_per_gravite = agg_data.groupby('grav_max')['Nombre_Accidents'].sum().reset_index()
                total_per_gravite.rename(columns={'Nombre_Accidents': 'Total_Gravite'}, inplace=True)

                agg_data_bar = pd.merge(agg_data.copy(), total_per_gravite, on='grav_max', how='left')
                agg_data_bar['Proportion'] = agg_data_bar.apply(
                    lambda row: (row['Nombre_Accidents'] / row['Total_Gravite']) * 100 if row['Total_Gravite'] > 0 else 0,
                    axis=1
                )

                # 6. Ordre des gravités
                gravite_order = ['Blessé léger', 'Blessé hospitalisé', 'Décès']
                agg_data_bar['grav_max'] = pd.Categorical(agg_data_bar['grav_max'], categories=gravite_order, ordered=True)

                # 7. Création du graphique
                fig_bar_agg = px.bar(
                    agg_data_bar,
                    x='grav_max',
                    y='Nombre_Accidents',
                    color='agg_label',
                    barmode='group',
                    text='Proportion',
                    title='Répartition des Accidents par Gravité et Localisation',
                    labels={
                        'Nombre_Accidents': "Nombre d'accidents",
                        'grav_max': 'Gravité',
                        'agg_label': 'Localisation'
                    },
                    category_orders={'grav_max': gravite_order},
                    custom_data=['Proportion'],
                    color_discrete_map={'Hors agglomération':  '#d62728', 'En agglomération':'#1f77b4'}
                )

                # 8. Personnalisation
                fig_bar_agg.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    hovertemplate="<b>%{x}</b><br>" +
                                "Localisation: %{fullData.name}<br>" +
                                "Nombre d'accidents: %{y:,}<br>" +
                                "Proportion: %{customdata[0]:.2f}%" +
                                "<extra></extra>"
                )

                fig_bar_agg.update_layout(
                    height=600,
                    yaxis_title="Nombre d'accidents",
                    xaxis_title="Gravité",
                    legend_title_text="Localisation",
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    yaxis_tickformat=',.0f'
                )

                # 9. Affichage dans Streamlit
                st.plotly_chart(fig_bar_agg, use_container_width=True)

            plot_accidents_by_agg_and_gravite(df)
    
            st.write("Les routes hors agglomération sont le théâtre de deux tiers (66.4%) des décès, contre seulement un tiers (33.6%) en ville. Cette surreprésentation dramatique est directement liée aux vitesses plus élevées pratiquées hors agglomération, qui transforment les accidents en événements aux conséquences souvent fatales. Les zones urbaines concentrent la majorité des accidents, en particulier les blessures légères, en raison d’un trafic dense et de nombreuses interactions routières, tandis que les zones rurales, bien que moins fréquentées, présentent une surreprésentation dramatique des accidents mortels liée aux vitesses élevées et à l’éloignement des secours. Cette répartition souligne la nécessité de stratégies de sécurité routière différenciées : réduire le volume d’accidents en agglomération et limiter leur gravité hors agglomération.")
     

    ## Usagers
    if usagers_checkbox:
        st.subheader("Profils et statuts des usagers")
        variables_usagers = st.radio(
        "Sélectionnez une variable :",
        ["Sexe", "Age", "Système de sécurité"])
       
        if "Sexe" in variables_usagers:
            st.write('Sexe')
            def plot_accidents_by_sexe_and_gravite(df):
            # 1. S'assurer que grav_max est en chaîne
                df['grav_max'] = df['grav_max'].astype(str)

            # 2. Nettoyer 'sexe'
                df.loc[:, 'sexe'] = df['sexe'].replace(-1, 1)

            # 3. Mapper en labels
                sex_map = {1: 'Homme', 2: 'Femme'}
                df['sexe_label'] = df['sexe'].map(sex_map)

            # 4. Supprimer les lignes incomplètes
                df.dropna(subset=['grav_max', 'sexe_label'], inplace=True)

            # 5. Agrégation
                agg_data = df.groupby(['grav_max', 'sexe_label'])['Num_Acc'].nunique().reset_index()
                agg_data.rename(columns={'Num_Acc': 'Nombre_Accidents'}, inplace=True)

            # 6. Proportions
                total_per_gravite = agg_data.groupby('grav_max')['Nombre_Accidents'].sum().reset_index()
                total_per_gravite.rename(columns={'Nombre_Accidents': 'Total_Gravite'}, inplace=True)

                agg_data_bar = pd.merge(agg_data.copy(), total_per_gravite, on='grav_max', how='left')
                agg_data_bar['Proportion'] = agg_data_bar.apply(
                lambda row: (row['Nombre_Accidents'] / row['Total_Gravite']) * 100 if row['Total_Gravite'] > 0 else 0,
                axis=1
            )

            # Ordre des gravités
                gravite_order = ['Blessé léger', 'Blessé hospitalisé', 'Décès']
                agg_data_bar['grav_max'] = pd.Categorical(agg_data_bar['grav_max'], categories=gravite_order, ordered=True)

            # 7. Création du graphique
                fig_bar_sexe = px.bar(
                    agg_data_bar,
                    x='grav_max',
                y='Nombre_Accidents',
                color='sexe_label',
                barmode='group',
                text='Proportion',
                title='Répartition des Accidents par Gravité et Sexe',
                labels={
                    'Nombre_Accidents': "Nombre d'accidents",
                    'grav_max': 'Gravité',
                    'sexe_label': 'Sexe'
                },
                category_orders={'grav_max': gravite_order},
                custom_data=['Proportion'],
                color_discrete_map={'Homme': '#d62728', 'Femme': '#1f77b4'}
            )

                fig_bar_sexe.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>" +
                            "Sexe: %{fullData.name}<br>" +
                            "Nombre d'accidents: %{y:,}<br>" +
                            "Proportion: %{customdata[0]:.2f}%" +
                            "<extra></extra>"
            )

                fig_bar_sexe.update_layout(
                height=600,
                yaxis_title="Nombre d'accidents",
                xaxis_title="Gravité",
                legend_title_text="Sexe",
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                yaxis_tickformat=',.0f'
            )

            # 8. Affichage Streamlit
                st.plotly_chart(fig_bar_sexe, use_container_width=True)
            
            plot_accidents_by_sexe_and_gravite(df)


            # 1. S'assurer que grav_max est en chaîne
            df['grav_max'] = df['grav_max'].astype(str)

            # 2. Nettoyer 'sexe'
            df.loc[:, 'sexe'] = df['sexe'].replace(-1, 1)

            # 3. Mapper en labels
            sex_map = {1: 'Homme', 2: 'Femme'}
            df['sexe_label'] = df['sexe'].map(sex_map)

            # 4. Supprimer les lignes incomplètes
            df.dropna(subset=['grav_max', 'sexe_label'], inplace=True)

            # 5. Agrégation
            agg_data = df.groupby(['grav_max', 'sexe_label'])['Num_Acc'].nunique().reset_index()
            agg_data.rename(columns={'Num_Acc': 'Nombre_Accidents'}, inplace=True)

            # 6. Proportions
            total_per_gravite = agg_data.groupby('grav_max')['Nombre_Accidents'].sum().reset_index()
            total_per_gravite.rename(columns={'Nombre_Accidents': 'Total_Gravite'}, inplace=True)

            agg_data_bar = pd.merge(agg_data.copy(), total_per_gravite, on='grav_max', how='left')
            agg_data_bar['Proportion'] = agg_data_bar.apply(
                lambda row: (row['Nombre_Accidents'] / row['Total_Gravite']) * 100 if row['Total_Gravite'] > 0 else 0,
                axis=1
            )

                    # 2. Filtrer les données agrégées par sexe
            data_hommes = agg_data[agg_data['sexe_label'] == 'Homme'].copy()  
            data_femmes = agg_data[agg_data['sexe_label'] == 'Femme'].copy()  

            # 3. Définir le mappage des couleurs pour la GRAVITÉ (pour les parts des camemberts)  
            color_map_gravite = {  
                'Blessé léger': '#1f77b4',  
                'Blessé hospitalisé': '#ff7f0e',  
                'Décès': '#d62728'  
            }  

            # 4. Créer la figure avec des sous-graphiques (1 ligne, 2 colonnes)  
            fig_pie_sexe = make_subplots(
                rows=1, cols=2, 
                specs=[[{'type':'domain'}, {'type':'domain'}]],  
                subplot_titles=('Hommes', 'Femmes')  # Titres des subplots
            )  

            # 5. Ajouter le camembert pour les Hommes (colonne 1)  
            if not data_hommes.empty:  
                fig_pie_sexe.add_trace(go.Pie(  
                    labels=data_hommes['grav_max'],  
                    values=data_hommes['Nombre_Accidents'],  
                    name='Hommes',  
                    marker=dict(colors=[color_map_gravite.get(grav, '#cccccc') for grav in data_hommes['grav_max']]),  
                    pull=[0.02 if grav == 'Décès' else 0 for grav in data_hommes['grav_max']],  
                    hole=.3,  
                    textinfo='percent+label',  
                    insidetextorientation='radial',  
                    sort=False  # Important pour potentiellement garder l'ordre des couleurs  
                ), 1, 1)  
            else:  
                st.write("Pas de données pour les Hommes.")  

            # 6. Ajouter le camembert pour les Femmes (colonne 2)  
            if not data_femmes.empty:  
                fig_pie_sexe.add_trace(go.Pie(  
                    labels=data_femmes['grav_max'],  
                    values=data_femmes['Nombre_Accidents'],  
                    name='Femmes',  
                    marker=dict(colors=[color_map_gravite.get(grav, '#cccccc') for grav in data_femmes['grav_max']]),  
                    pull=[0.02 if grav == 'Décès' else 0 for grav in data_femmes['grav_max']],  
                    hole=.3,  
                    textinfo='percent+label',  
                    insidetextorientation='radial',  
                    sort=False  
                ), 1, 2)  
            else:  
                st.write("Pas de données pour les Femmes.")  

            # 7. Mettre à jour la mise en page générale des camemberts  
            fig_pie_sexe.update_layout(  
                title_text='Répartition Interne de la Gravité par Sexe',  
                height=500,  
                showlegend=False  # Les infos sont sur les parts  
            )  

            # 8. Afficher la figure dans Streamlit
            st.plotly_chart(fig_pie_sexe, use_container_width=True)
            

            # Conclusion
            st.write("L’analyse met en évidence une surreprésentation marquée des hommes dans les accidents de la route, tous niveaux de gravité confondus. Ce déséquilibre s’accentue avec la sévérité : les hommes représentent 70,4 % des décès. Au-delà de leur implication plus fréquente, ils présentent aussi un risque plus élevé que les femmes d’être hospitalisés (30,4 % vs 28,0 %) ou tués (5,88 % vs 4,36 %) après un accident.")

          

        if "Age" in variables_usagers:
            st.write ("Age")

            agg_data_age = df.groupby(['grav_max', 'Classe_Age'])['Num_Acc'].nunique().reset_index()
            agg_data_age.rename(columns={'Num_Acc': 'Nombre_Accidents'}, inplace=True)

            # Totaux pour les proportions
            total_per_gravite_age = agg_data_age.groupby('grav_max')['Nombre_Accidents'].sum().reset_index()
            total_per_gravite_age.rename(columns={'Nombre_Accidents': 'Total_Gravite'}, inplace=True)

            # Fusion + calcul des proportions
            agg_data_age = pd.merge(agg_data_age, total_per_gravite_age, on='grav_max', how='left')
            agg_data_age['Proportion'] = agg_data_age.apply(
                lambda row: (row['Nombre_Accidents'] / row['Total_Gravite']) * 100 if row['Total_Gravite'] > 0 else 0,
                axis=1
            )

            # --- Paramètres du graphe ---
            gravite_order = ['Blessé léger', 'Blessé hospitalisé', 'Décès']
            age_order = ['<18ans', '18-60ans', '>60ans']

            fig_bar_age = px.bar(
                agg_data_age,
                x='grav_max',
                y='Nombre_Accidents',
                color='Classe_Age',
                barmode='group',
                text='Proportion',
                title='Répartition des Accidents par Gravité et Classe d\'Âge',
                labels={
                    'Nombre_Accidents': 'Nombre d\'Accidents',
                    'grav_max': 'Gravité Maximale Constatée',
                    'Classe_Age': 'Classe d\'Âge'
                },
                category_orders={
                    'grav_max': gravite_order,
                    'Classe_Age': age_order
                },
                custom_data=['Proportion']
            )

            fig_bar_age.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>" +
                            "Classe d'Âge: %{fullData.name}<br>" +
                            "Nombre d'Accidents: %{y:,}<br>" +
                            "Proportion: %{customdata[0]:.2f}%" +
                            "<extra></extra>"
            )

            fig_bar_age.update_layout(
                width=1000,
                height=600,
                yaxis_title='Nombre d\'Accidents',
                xaxis_title='Gravité Maximale Constatée',
                legend_title_text='Classe d\'Âge',
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                yaxis_tickformat=',.0f'
            )

            # --- Affichage dans Streamlit ---
            st.plotly_chart(fig_bar_age, use_container_width=True)

            data_moins_18 = agg_data_age[agg_data_age['Classe_Age'] == '<18ans'].copy()
            data_18_60 = agg_data_age[agg_data_age['Classe_Age'] == '18-60ans'].copy()
            data_plus_60 = agg_data_age[agg_data_age['Classe_Age'] == '>60ans'].copy()

            # --- Mappage des couleurs pour la gravité ---
            color_map_gravite = {
                'Blessé léger': '#1f77b4',
                'Blessé hospitalisé': '#ff7f0e',
                'Décès': '#d62728'
            }

            # --- Création des sous-graphes (3 camemberts côte à côte) ---
            fig_pie_age = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
                subplot_titles=('<18 ans', '18-60 ans', '>60 ans')
            )

            # --- Ajout des données pour chaque classe d'âge ---
            if not data_moins_18.empty:
                fig_pie_age.add_trace(go.Pie(
                    labels=data_moins_18['grav_max'],
                    values=data_moins_18['Nombre_Accidents'],
                    name='<18ans',
                    marker=dict(colors=[color_map_gravite.get(g, '#cccccc') for g in data_moins_18['grav_max']]),
                    pull=[0.02 if g == 'Décès' else 0 for g in data_moins_18['grav_max']],
                    hole=0.3,
                    textinfo='percent+label',
                    insidetextorientation='radial',
                    sort=False
                ), 1, 1)

            if not data_18_60.empty:
                fig_pie_age.add_trace(go.Pie(
                    labels=data_18_60['grav_max'],
                    values=data_18_60['Nombre_Accidents'],
                    name='18-60ans',
                    marker=dict(colors=[color_map_gravite.get(g, '#cccccc') for g in data_18_60['grav_max']]),
                    pull=[0.02 if g == 'Décès' else 0 for g in data_18_60['grav_max']],
                    hole=0.3,
                    textinfo='percent+label',
                    insidetextorientation='radial',
                    sort=False
                ), 1, 2)

            if not data_plus_60.empty:
                fig_pie_age.add_trace(go.Pie(
                    labels=data_plus_60['grav_max'],
                    values=data_plus_60['Nombre_Accidents'],
                    name='>60ans',
                    marker=dict(colors=[color_map_gravite.get(g, '#cccccc') for g in data_plus_60['grav_max']]),
                    pull=[0.02 if g == 'Décès' else 0 for g in data_plus_60['grav_max']],
                    hole=0.3,
                    textinfo='percent+label',
                    insidetextorientation='radial',
                    sort=False
                ), 1, 3)

            # --- Mise en page ---
            fig_pie_age.update_layout(
                title_text="Répartition Interne de la Gravité par Classe d'Âge",
                height=500,
                showlegend=False
            )

            # --- Affichage dans Streamlit ---
            st.plotly_chart(fig_pie_age, use_container_width=True)

            st.write("L'analyse révèle une double réalité claire du risque routier lié à l'âge : d'une part, les 18-60 ans dominent massivement le volume des accidents (représentant 60-70% des victimes). D'autre part, les seniors (>60 ans) affichent une vulnérabilité critique : ils sont surreprésentés parmi les victimes les plus graves (constituant 27.9% des décès) et, surtout, leur risque intrinsèque de décéder en cas d'accident est nettement plus élevé (8.33%).")
        
        if "Système de sécurité" in variables_usagers:
            
            # Fonction pour agréger les données avec cache
            @st.cache_data
            def aggregate_gravity_data(df):
                return df.groupby('gravité_accident').agg({
                    'total_sans_secu': 'sum', 
                    'total_ceinture': 'sum', 
                    'total_casque': 'sum', 
                    'total_secu_enfant': 'sum', 
                    'total_gilet': 'sum', 
                    'total_airbag': 'sum', 
                    'total_gants': 'sum', 
                    'total_gants_airbag': 'sum', 
                    'total_autre': 'sum'
                }).reset_index()

            # Fonction pour créer le graphique avec cache
            @st.cache_data
            def create_bar_chart(grav):
                fig = px.bar(
                    grav, 
                    x='gravité_accident', 
                    y=['total_sans_secu', 'total_ceinture', 'total_casque', 'total_secu_enfant', 'total_gilet', 'total_airbag', 'total_gants', 'total_gants_airbag', 'total_autre'], 
                    title="Répartition du nombre d'usagers par catégorie de système de sécurité selon la gravité", 
                    barmode='group'
                )
                fig.update_xaxes(tickmode='array', tickvals=[2, 3, 4], ticktext=['blessé_léger', 'blessé_grave', 'tué'])
                return fig

            # Agrégation des données avec cache
            grav = aggregate_gravity_data(df_total_final)

            # Affichage du titre
            st.write("### Système de sécurité")

            # Création du graphique avec cache
            fig_bar = create_bar_chart(grav)

            # Affichage du graphique
            st.plotly_chart(fig_bar)

            st.write("L’absence d’équipement de sécurité multiplie par deux à deux fois et demie le risque de décès ou d’hospitalisation, confirmant l’efficacité protectrice majeure du port de la ceinture et du casque et soulignant l’urgence de renforcer contrôle, sensibilisation et prévention ciblée.")


########## Véhicules
    if vehicules_checkbox:
        st.subheader("Véhicules impliqués")
        variables_vehicule = st.radio(
        "Sélectionnez une variable :",
        ["Type de véhicule", "Point de choc"])
       
        if "Type de véhicule" in variables_vehicule:
            st.write("Type de véhicule")

            # Fonction de mise en cache pour les données filtrées et les calculs de proportion
            @st.cache_data
            def get_accidents_with_vehicle(df, column_name):
                # Filtrer les accidents où au moins 1 véhicule de type spécifié est impliqué
                accidents = df.loc[df[column_name] >= 1]
                # Calculer la proportion des gravités d'accidents
                accident_by_severity = accidents.groupby("gravité_accident").size()
                return accident_by_severity.div(accident_by_severity.sum())

            # Affichage des résultats et graphiques
            def plot_accident_distribution(acc_avec_pl, acc_avec_2_3roues, acc_avec_bus_car, 
                                            acc_av_vl_vu, acc_av_velo_trott_edp, acc_av_pietons):
                labels = ['Blessés légers', 'Blessés hospitalisés', 'Tués']
                
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 lignes et 3 colonnes

                # Graphiques pour la première ligne
                axs[0, 0].pie(acc_avec_pl, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[0, 0].set_title("Avec au moins PL")
                axs[0, 1].pie(acc_avec_2_3roues, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[0, 1].set_title("Avec au moins un 2 ou 3 roues ou quad")
                axs[0, 2].pie(acc_avec_bus_car, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[0, 2].set_title("Avec au moins bus/car")

                # Graphiques pour la deuxième ligne
                axs[1, 0].pie(acc_av_vl_vu, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[1, 0].set_title("Avec au moins un VL VU")
                axs[1, 1].pie(acc_av_velo_trott_edp, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[1, 1].set_title("Avec au moins un vélo, trott ou edp")
                axs[1, 2].pie(acc_av_pietons, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[1, 2].set_title("Avec au moins un piéton")

                fig.suptitle("Répartition des accidents par gravité pour différents types de véhicules")
                fig.legend(labels, loc='center')
                
                # Affichage dans Streamlit
                st.pyplot(fig)

                st.write("L’analyse révèle que la gravité des accidents varie fortement selon le type de véhicule : les deux-roues motorisés présentent un risque élevé d’hospitalisation, tandis que les engins agricoles et spécialisés enregistrent les taux de mortalité les plus élevés.")

            # Calculer les proportions pour différents types de véhicules
            acc_avec_pl = get_accidents_with_vehicle(df_total_final, "PL")
            acc_avec_2_3roues = get_accidents_with_vehicle(df_total_final, "2roues_3roues_quad")
            acc_avec_bus_car = get_accidents_with_vehicle(df_total_final, "bus_car")
            acc_av_vl_vu = get_accidents_with_vehicle(df_total_final, "VL_VU")
            acc_av_velo_trott_edp = get_accidents_with_vehicle(df_total_final, "velo_trott_edp")
            acc_av_pietons = get_accidents_with_vehicle(df_total_final, "place_pieton")


            # Afficher les graphiques
            plot_accident_distribution(acc_avec_pl, acc_avec_2_3roues, acc_avec_bus_car, 
                                        acc_av_vl_vu, acc_av_velo_trott_edp, acc_av_pietons)

            st.write("Les accidents les plus graves sont ceux impliquant des tonneaux et des 2/3 roues. Les accidents les plus mortels sont ceux impliquant au moins un poids lourd.")

        if "Point de choc" in variables_vehicule:
            st.write("Point de choc")

            # Cache des données traitées
            @st.cache_data
            def get_gravité_accidents(df):
                grav = df.groupby('gravité_accident').agg({
                    'choc_AV':'sum', 
                    'choc_AR': 'sum', 
                    'choc_cote': 'sum', 
                    'choc_tonneaux':'sum', 
                    'aucun_choc':'sum'
                }).reset_index()
                return grav

            @st.cache_data
            def get_accidents_with_choc_type(df, choc_column, condition):
                accidents = df.loc[condition]
                return accidents.groupby("gravité_accident").size()

            # Calculer les agrégations nécessaires pour les graphiques
            grav = get_gravité_accidents(df_total_final)

            # Affichage du graphique avec Plotly
            fig = px.bar(grav, x='gravité_accident', y=['choc_AV', 'choc_AR', 'choc_cote', 'choc_tonneaux', 'aucun_choc'], 
                        title="Répartition du nombre d'accidents selon la gravité et le point de choc", 
                        barmode='group')
            fig.update_xaxes(tickmode='array', tickvals=[2, 3, 4], ticktext=['blessé_léger', 'blessé_grave', 'tué'])

            # Affichage du graphique dans Streamlit
            st.plotly_chart(fig)

            # Extraire les accidents pour chaque type de choc
            acc_sans_choc = get_accidents_with_choc_type(df_total_final, 'aucun_choc', (df_total_final["aucun_choc"] >= 1) & (df_total_final["choc_AV"] == 0) & (df_total_final["choc_AR"] == 0) & (df_total_final["choc_tonneaux"] == 0) & (df_total_final["choc_cote"] == 0))
            acc_av_choc_av = get_accidents_with_choc_type(df_total_final, 'choc_AV', (df_total_final["aucun_choc"] == 0) & (df_total_final["choc_AV"] >= 1) & (df_total_final["choc_AR"] == 0) & (df_total_final["choc_tonneaux"] == 0) & (df_total_final["choc_cote"] == 0))
            acc_av_tonneaux = get_accidents_with_choc_type(df_total_final, 'choc_tonneaux', (df_total_final["aucun_choc"] == 0) & (df_total_final["choc_AV"] == 0) & (df_total_final["choc_AR"] == 0) & (df_total_final["choc_tonneaux"] >= 1) & (df_total_final["choc_cote"] == 0))
            acc_av_choc_ar = get_accidents_with_choc_type(df_total_final, 'choc_AR', (df_total_final["aucun_choc"] == 0) & (df_total_final["choc_AV"] == 0) & (df_total_final["choc_AR"] >= 1) & (df_total_final["choc_tonneaux"] == 0) & (df_total_final["choc_cote"] == 0))
            acc_av_choc_cote = get_accidents_with_choc_type(df_total_final, 'choc_cote', (df_total_final["aucun_choc"] == 0) & (df_total_final["choc_AV"] == 0) & (df_total_final["choc_AR"] == 0) & (df_total_final["choc_tonneaux"] == 0) & (df_total_final["choc_cote"] >= 1))

            # Extraire les accidents avec plusieurs chocs
            acc_chocs_multiples = df_total_final.loc[
                ((df_total_final["choc_AV"] >= 1).astype(int) + (df_total_final["choc_AR"] >= 1).astype(int) + (df_total_final["choc_tonneaux"] >= 1).astype(int) + (df_total_final["choc_cote"] >= 1).astype(int)) >= 2
            ]
            acc_chocs_multiples = acc_chocs_multiples.groupby("gravité_accident").size()

            # Création des graphiques en secteurs
            labels = ['Blessés légers', 'Blessés hospitalisés', 'Tués']
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            # Graphiques en secteurs pour chaque type de choc
            axs[0, 0].pie(acc_sans_choc, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[0, 0].set_title("Sans choc")
            axs[0, 1].pie(acc_av_choc_av, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[0, 1].set_title("Avec choc AV")
            axs[0, 2].pie(acc_av_choc_ar, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[0, 2].set_title("Avec choc AR")
            axs[1, 0].pie(acc_av_tonneaux, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[1, 0].set_title("Avec tonneaux")
            axs[1, 1].pie(acc_av_choc_cote, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[1, 1].set_title("Avec choc coté")
            axs[1, 2].pie(acc_chocs_multiples, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[1, 2].set_title("Avec plusieurs chocs")

            # Titre et légende des graphiques
            fig.suptitle("Répartition des accidents par gravité pour différents types de chocs")
            fig.legend(labels, loc='center')

            # Affichage du graphique dans Streamlit
            st.pyplot(fig)

            st.write("L’analyse montre que la gravité des accidents varie fortement selon le type de choc : les chocs tonneaux se révèlent les plus dangereux, avec les taux les plus élevés d’hospitalisations (40,6 %) et de décès (10,0 %), tandis que les chocs arrière et latéraux entraînent majoritairement des blessures légères. Par ailleurs, les situations sans choc apparent affichent une gravité surprenante, suggérant des mécanismes spécifiques comme les sorties de route, qui nécessitent des mesures de prévention adaptées.")


# Matrices de corrélation
    if matrices_corrélation_checkbox:
        st.subheader("Matrices de corrélation")
        variables_matrices_corrélation = st.radio(
        "Sélectionnez une variable :",
        ["Corrélation selon les conditions des usagers", "Corrélation selon les conditions de la route","Corrélation selon les obstacles et types de chocs" ])
       
        if "Corrélation selon les conditions des usagers" in variables_matrices_corrélation:
            @st.cache_data
            def get_filtered_data(df):
                to_drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                df_filtré = df.drop(df.columns[to_drop], axis=1)
                return df_filtré

            @st.cache_data
            def get_correlation_matrix(df_filtré, selected_columns1):
                corr_matrix_subset1 = df_filtré[selected_columns1].corr()
                return corr_matrix_subset1

            # Sélectionner les colonnes pour la matrice de corrélation
            selected_columns1 = [
                'usager_count', '0-17', '18-60', '61-95', 'gravité_accident', 'homme', 'femme', 
                'place_conducteur', 'pax_AV', 'pax_AR', 'pax_Milieu', 'place_pieton', 
                'blessé_léger', 'blessé_hospitalisé', 'tué'
            ]

            # Appliquer le cache sur les données filtrées et la matrice de corrélation
            df_filtré = get_filtered_data(df_total_final)
            corr_matrix_subset1 = get_correlation_matrix(df_filtré, selected_columns1)

            # Affichage de la matrice de corrélation avec Seaborn
            st.subheader("Matrice de corrélation selon les conditions des usagers")
            # Créer un objet figure explicitement
            fig, ax = plt.subplots(figsize=(12, 8))

            # Tracer la carte de chaleur sur cet axe (ax)
            sns.heatmap(corr_matrix_subset1, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)

            # Passer l'objet figure à st.pyplot()
            st.pyplot(fig)

            st.write("L’analyse des corrélations montre que la gravité d’un accident prédit fidèlement le risque de décès ou d’hospitalisation, et que les accidents les plus graves impliquent souvent plusieurs usagers adultes, majoritairement des hommes en places avantt.")
            
          

        if "Corrélation selon les conditions de la route" in variables_matrices_corrélation:
            @st.cache_data
            def get_filtered_data(df):
                to_drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                df_filtré = df.drop(df.columns[to_drop], axis=1)
                df_filtré['nbv'] = df['nbv'].replace(to_replace=['#ERREUR', '#VALEURMULTI'], value=np.nan)
                return df_filtré

            @st.cache_data
            def compute_correlation(df, selected_columns):
                # Calculer la matrice de corrélation pour les colonnes sélectionnées
                return df[selected_columns].corr()

            # Appliquer le pré-traitement
            df = get_filtered_data(df_total_final)

            # Colonnes sélectionnées pour la corrélation
            selected_columns3 = ['gravité_accident', 'nationale_departementale_communale', 'autoroute', 'autre_route',
                                'sens_unique', 'vma', 'bidirectionnel', 'route_seche', 'route_mouillee_enneigee',
                                'etat_route_autre', 'nbv', 'blessé_léger', 'blessé_hospitalisé', 'tué']

            
            # Calcul de la matrice de corrélation
            corr_matrix_subset3 = compute_correlation(df, selected_columns3)

            # Affichage de la matrice de corrélation avec Streamlit
            st.subheader("Matrice de corrélation selon les conditions de la route")

            # Créer un objet figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Tracer la carte de chaleur sur l'axe
            sns.heatmap(corr_matrix_subset3, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)

            # Afficher la figure dans Streamlit
            st.pyplot(fig)

            st.write("Cette matrice révèle que la gravité des accidents est moins liée aux conditions de route qu’à la vitesse autorisée, au type de réseau et au caractère bidirectionnel, soulignant l’importance de la conception routière et du comportement des conducteurs dans la sévérité des issues.")
            
        if "Corrélation selon les obstacles et types de chocs" in variables_matrices_corrélation:
            def get_filtered_data(df):
                to_drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                df_filtré = df.drop(df.columns[to_drop], axis=1)
                return df_filtré
            
            @st.cache_data
            def compute_correlation(df, selected_columns):
                # Calculer la matrice de corrélation pour les colonnes sélectionnées
                return df[selected_columns].corr()
            
            selected_columns2 = ['gravité_accident', 'obstacle_fixe', 'obstacle_mobile', 'aucun_choc', 'choc_AV', 'choc_AR', 
                     'choc_cote', 'choc_tonneaux', 'blessé_léger', 'blessé_hospitalisé', 'tué']

            df_filtré = get_filtered_data(df_total_final)
            # Calcul de la matrice de corrélation
            corr_matrix_subset2 = compute_correlation(df_filtré, selected_columns2)

            # Affichage de la matrice de corrélation avec Streamlit
            st.subheader("Matrice de corrélation selon les obstacles et types de chocs")

            # Créer un objet figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Tracer la carte de chaleur sur l'axe
            sns.heatmap(corr_matrix_subset2, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)

            # Afficher la figure dans Streamlit
            st.pyplot(fig)

            st.write("Cette matrice met en évidence que la gravité des accidents est davantage influencée par la nature de l’obstacle (fixe > mobile) et l’énergie de l’impact que par le type de choc, soulignant l’importance de sécuriser les abords routiers et de mieux contrôler la vitesse pour réduire les conséquences les plus graves.")
            




elif st.session_state.page == "Modélisation":
    st.header("Modélisation")
    #"Evaluation modèles 3 classes"
    if Méthodologie_et_résultats_checkbox:
        tab1, tab2, tab3= st.tabs(["Méthodologie", " Synthèse des résultats", "Conclusions"])

# Contenu du premier onglet
        with tab1:
          

            st.subheader("Méthodologie")
        
            st.markdown("""
<div style="background-color: #e8f5e9; border: 2px solid #81c784; padding: 20px; border-radius: 10px;">



L'objectif de cette étude est de tester différents <strong>modèles de classification</strong> et plusieurs <strong>configurations d'entraînement</strong>, dans le but d'obtenir les <strong>meilleures prédictions possibles</strong> quant à la gravité des accidents.

<hr style="border:1px solid #81c784;">

<h4>Contexte</h4>

La variable cible, <strong>gravité</strong>, est catégorisée en trois classes :
<ul>
<li>Blessé léger</li>
<li>Blessé hospitalisé</li>
<li>Tué</li>
</ul>

<hr style="border:1px solid #81c784;">

<h4>Quelle classe souhaite-t-on prédire en priorité ?</h4>

Nous cherchons à prédire la gravité d’un accident afin de <strong>mobiliser les secours de manière appropriée</strong>.  
Notre priorité est donc de <strong>bien identifier les blessés hospitalisés</strong> :

<ul>
<li>Pour les blessés légers : la mobilisation urgente des secours n’est pas nécessaire.</li>
<li>Pour les tués : l’intervention rapide ne permet malheureusement plus d’agir.</li>
<li>Les blessés hospitalisés représentent la <strong>zone critique</strong>, où une intervention rapide peut sauver des vies.</li>
</ul>

<hr style="border:1px solid #81c784;">

<h4>Critères d’évaluation</h4>

Nous cherchons à :
<ul>
<li><strong>Maximiser le rappel (Recall)</strong> : bien identifier les vrais blessés hospitalisés.</li>
<li><strong>Optimiser le F1-Score</strong> : trouver un équilibre entre rappel et précision, en limitant les fausses alertes (faux positifs).</li>
</ul>

<hr style="border:1px solid #81c784;">

<h4>Approche méthodologique</h4>

Il s'agit d’un <strong>problème de classification multi-classes déséquilibrées</strong>.  
Pour gérer ce déséquilibre, nous testons plusieurs stratégies :
<ul>
<li><strong>Oversampling</strong></li>
<li><strong>Undersampling</strong></li>
<li><strong>Personnalisation des métriques (poids de classe)</strong></li>
</ul>

Nous avons évalué <strong>quatre modèles de classification</strong> :
<ul>
<li>Random Forest</li>
<li>Logistic Regression</li>
<li>XGBoost</li>
<li>K-Nearest Neighbors (KNN)</li>
</ul>

Pour chacun des modèles, deux configurations sont testées :
<ul>
<li>Classification en <strong>3 classes</strong> : Blessé léger, hospitalisé, tué</li>
<li>Classification en <strong>2 classes</strong> : Urgent vs. Non urgent</li>
</ul>

Nous combinons également ces stratégies avec du <strong>Grid Search</strong> (quand applicable) afin d’optimiser les hyperparamètres et les performances.

<hr style="border:1px solid #81c784;">

<h4>Jeu de données</h4>

Les données sont divisées en deux ensembles :
<ul>
<li><strong>X_train</strong> : 80 % des données pour l’entraînement</li>
<li><strong>X_test</strong> : 20 % pour l’évaluation</li>
</ul>

</div>
""", unsafe_allow_html=True)
        
        with tab2:
            st.header("Synthèse des résultats")
            st.write('')

            st.subheader("Modèles 3 classes")
            modeles_selectionnes_3 = st.selectbox(
            "Sélectionnez un modèle :",
            ["Random Forest_3_classes", "Logistic Regression_3_classes", "XGBoost_3_classes","KNN_3_classes"])
        
            if modeles_selectionnes_3 == "Random Forest_3_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\Random_Forest.jpg"
                st.image(image_path, use_column_width=True)

                st.write("Les différentes approches de classification basées sur le modèle de Random Forest montrent une précision globale variant entre 0.58 et 0.72. Le score de la métrique « Recall » pour la classe ‘Blessé hospitalisé’ atteint 0.62 pour les approches pondérées . De plus, la valeur du F1-Score atteint jusqu’à 0.57 grâce aux pondérations ciblées, montrant ainsi un excellent équilibre entre les métriques Précision et Rappel.Le modèle Random Forest présente une satisfaisante détestation des cas « Blessé hospitalisé » grâce aux approches pondérées. Le niveau de F1-Score élevé prouve un compromis idéal entre capacité de détection (Recall) et précision.")

            elif modeles_selectionnes_3 == "Logistic Regression_3_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\Logistic_Reg.jpg"
                st.image(image_path, use_column_width=True)

                st.write("Les différentes approches testées montrent un taux de prédictions correctes parmi l’ensemble des prédictions effectuées variant entre 0.59 et 0.70 mettant en évidence une robustesse globale. Le score maximum de détection des vrais « Blessés hospitalisés » s’élève à 0.44 avec validation par Grid Search mais demeurent inférieur au modèle Random Forest. Quant au F1-Score, il demeure stagnant à 0.48 dans les différentes approches limitant son équilibre entre Précision et Rappel.En somme,bien que  simple et rapide à l'entraînement, ce modèle affiche une sous performance marquée pour la détection des « Blessé hospitalisé » comparativement au modèle Random Forest. Le score moins performant du F1-Score démontre un déséquilibre entre détection et erreurs de classification.")


            elif modeles_selectionnes_3 == "XGBoost_3_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\XGBOOST.jpg"
                st.image(image_path, use_column_width=True)
                st.write("La proportion de prédictions correctes parmi l’ensemble des prédictions effectuées (variant entre 0.63 et 0.72) est comparable au modèle Random Forest. Les différentes approches testées donnent un score maximum de détection correcte des vrais « Blessés hospitalises » à 0.49, inférieur au modèle Random Forest.. Le score maximum d’équilibre en entre les détections correctes (Recall) et Fausses alertes (Précision) est de 0.53, légèrement meilleur que dans le modèle Logistic Regression mais inférieur à celui du modèle Random Forest (0.57).En somme, le modèle Xtreme Gradient Boosting (XGBoost) présente une efficacité comparable à Random Forest dans les approches basiques et une bonne robustesse globale sur l’Accuracy et la Précision. En revanche, ce modèle montre un score de Rappel sur les cas « Hospitalisé » inférieur à Random Forest.")
            
            elif modeles_selectionnes_3 == "KNN_3_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\KNN.jpg"
                st.image(image_path, use_column_width=True)
                st.write("Dans ce dernier modèle de classification, le score global de prédictions correctes sur l’ensemble de prédictions réalisées varie entre 0.68 et 0.69, inférieur aux modèles de Random Forest et XGBoost. Le taux maximum de détection des vrais cas de « Blessé hospitalisé » est de 0.44, bien inférieur au modèle Random Forest. Le niveau maximal d’équilibre entre Rappel et Précision est de 0.48. Cette performance est comparable à celle du modèle Logistic Regression mais inférieure à Random Forest et XGBoost.En somme, les performances du modèle KNN pour notre classe prioritaire (Blessé hospitalisé) laissent à désirer. Les métriques Rappel et F1-Score demeurent médiocres en comparaison avec Random Forest et XGBoost.")

        
            st.subheader("Modèles 2 classes")
            modeles_selectionnes_2 = st.selectbox(
                "Sélectionnez un modèle :",["Random Forest_2_classes", "Logistic Regression_2_classes", "KNN_2_classes", "XGBoost_2_classes"])
        
            if modeles_selectionnes_2 == "Random Forest_2_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\2_classes_Random_Forest.jpg"
                st.image(image_path, use_column_width=True)
                st.write("Le modèle Random Forest s’impose comme une solution équilibrée et robuste, offrant un excellent compromis entre précision (jusqu’à 0.54), rappel (jusqu’à 0.67) et F1-Score (0.57) sur la classe Urgent, ce qui en fait un choix stratégique pertinent pour répondre aux priorités métiers.")
            elif modeles_selectionnes_2 == "Logistic Regression_2_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\2_classes_Logistic_Reg.jpg"
                st.image(image_path, use_column_width=True)
                st.write("La régression logistique offre une solution simple, lisible et globalement fiable, avec une accuracy stable (0.68–0.73) et un Recall élevé (jusqu’à 0.68) sur la classe Urgent, mais montre moins de flexibilité et un F1-Score (0.56) légèrement inférieur à Random Forest, limitant son efficacité pour notre problématique.")
            elif modeles_selectionnes_2 == "KNN_2_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\2_classes_KNN.jpg"
                st.image(image_path, use_column_width=True)
                st.write("KNN s’avère inadapté dans ce contexte critique, avec une faible précision (0.49), un Recall limité (0.42) et un F1-Score de 0.45, soulignant son incapacité à détecter efficacement les cas urgents.")
                
            elif modeles_selectionnes_2 == "XGBoost_2_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\2_classes_XGBOOST.jpg"
                st.image(image_path, use_column_width=True)
                st.write("Le modèle XGBoost se distingue par une excellente capacité de détection des urgences, avec un Recall atteignant 0.70 et un F1-Score de 0.59, surpassant légèrement Random Forest, bien qu’il n’égale pas toujours sa robustesse globale selon les scénarios.")
        
        with tab3:
          

            st.subheader("Conclusions")

            st.write('##### Conclusion modèles de classification 3 classes')
            st.write("Parmi les différents modèles testés, Random Forest s’impose comme le plus adapté aux priorités métier, notamment pour la détection des blessés hospitalisés, avec un Recall allant jusqu’à 0.62 et un F1-Score maximal de 0.57 grâce aux pondérations ciblées. Les approches pondérées (axées soit sur le Recall, soit sur le F1-score) permettent un bon compromis entre détection et précision pour cette classe prioritaire. En revanche, les modèles simplifiés ou non équilibrés montrent des biais importants en faveur de la classe dominante (blessé léger), rendant leur utilisation limitée en contexte opérationnel. Globalement, les approches pondérées sont à privilégier, pour renforcer la détection des hospitalisations.")
        # Ligne horizontale
            st.markdown("---")  # Cela ajoute une ligne horizontale
            st.write('##### Conclusion modèles de classification binaire')
            st.write("Parmi les trois approches testées pour XGBoost (simple, oversampling, grid search personnalisé), l’approche simple se distingue comme la plus efficace pour la détection des cas urgents, avec un Recall de 0.70 et un F1-Score de 0.59. Les autres approches, bien qu’ayant une meilleure accuracy globale, montrent une dégradation du rappel et de la précision sur les urgences, les rendant moins adaptées aux besoins métiers. Enfin, la simplification en modèle binaire n’a pas apporté de gains significatifs, confirmant que la complexité du problème ne réside pas uniquement dans le nombre de classes.")


# Contenu du deuxième onglet
        
    if Prédictions_checkbox:
        st.subheader('Prédictions')
        tab1, tab2= st.tabs(["Méthodologie", "Simulation"])

        with tab1:
            
            paragraphe = """
<div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; color: #155724;">
    <p>
        Nous envisageons que les véhicules de demain puissent être équipés d’un boîtier, similaire à la boîte noire d’un avion, capable d'enregistrer un ensemble précis de paramètres (via la connexion GPS et les interactions avec le conducteur). En cas d'accident, ce boîtier pourrait alors déclencher une alerte aux secours.
    </p>
    <p>
        Les boîtiers des véhicules accidentés pourraient fusionner leurs données, appliquer un modèle de machine learning, et transmettre au centre de secours une estimation du nombre de blessés hospitalisés.
    </p>
    <p>
        Il semble réaliste que ce boîtier puisse collecter les informations suivantes :
        <br>
        <strong>'heure', 'agg', 'nbv', 'vma', 'homme', 'femme', '0-17', '18-60', '61-95', 'PL', 'bus_car', 'VL_VU', '2roues_3roues_quad'</strong>.
    </p>
    <p>
        En nous basant sur les meilleurs modèles obtenus — <strong>Random Forest 3 classes</strong> et <strong>XGBoost 2 classes</strong> — nous testons ces configurations avec différentes variables et paramètres, dans le but de rapprocher au maximum les résultats de ceux obtenus précédemment et présentés dans la section précédente.
    </p>
    </p>
    "Pour le modèle Random Forest 3 classes, nous retenons ces variables: 'heure', 'agg', 'nbv', 'vma','homme', 'femme', '0-17','18-60', '61-95' et obtenons les scores suivants:"
    </p>
</div>
"""

            st.markdown(paragraphe, unsafe_allow_html=True)
            image_path= r"C:\Users\macha\Desktop\Test_py\Boitier_3_classes_Random_Forest.png"
            st.image(image_path, use_column_width=True)
            texte = """
<div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border: 1px solid #c3e6cb; color: #155724;">
    Pour le modèle <strong>XGBoost 2 classes</strong>, nous avons besoin de plus de variables pour un résultat pertinent :
    <br>
    <strong>'heure', 'agg', 'nbv', 'vma', 'homme', 'femme', '0-17', '18-60', '61-95', 'PL', 'bus_car', 'VL_VU', '2roues_3roues_quad'</strong>
    <br><br>
    et obtenons les scores suivants :
</div>
"""

            st.markdown(texte, unsafe_allow_html=True)
            image_path= r"C:\Users\macha\Desktop\Test_py\Boitier_2_classes_XGBoost.png"
            st.image(image_path, use_column_width=True)
            texte = """
<div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border: 1px solid #c3e6cb; color: #155724;">
    Les modèles sont tout de même moins performants avec cette sélection de variables, mais il paraît pertinent, pour la prédiction, de s’adapter à une situation concrète où les variables pourraient être renseignées par un boîtier intégré à la voiture.
</div>
"""

            st.markdown(texte, unsafe_allow_html=True)

        with tab2:
            model_choice = st.selectbox("Choisissez un modèle :", [" ", "Random Forest 3 classes", "XGBoost 2 classes"])

            # Charger le bon modèle en fonction du choix
            if model_choice == "Random Forest 3 classes":
                    
                model = joblib.load("model_boitier_random_forest_3_classes.joblib")  # Remplace par ton chemin réel
                
                preprocessor_random = joblib.load("preprocessor_random.joblib")


                st.write("")
                st.write('Conditions')
                st.write("")
                # Widgets pour choisir les valeurs des features
                col1, col2, col3 = st.columns(3)

            
                with col1:
                    agg = st.selectbox("Localisation", ["Hors_agglomération", "En agglomération"])
                with col2:
                    atm = st.selectbox("Atmosphère", ['temps_normal', 'Temps_pluvieux', 'Temps_couvert', 'Autre'])
                with col3:
                    heure = st.time_input("Heure", value=time(12, 0))
                
                st.write("")
                st.write('Route')
                st.write("")

                col4,col5 = st.columns(2)
                
                with col4:
                    nbv= st.slider("Nombre de voies", min_value=0, max_value=10, value=0)
                with col5:
                    vma = st.selectbox("Vitesse maximale autorisée", [20, 30, 50, 60, 70, 80, 90,100, 110, 130])

                st.write("")
                st.write('Sexe')
                st.write("")

                col6,col7 = st.columns(2)
                
                with col6:
                    homme = st.slider("Nombre d'hommes", min_value=0, max_value=10, value=0)
                with col7:
                    femme = st.slider("Nombre de femmes", min_value=0, max_value=10, value=0)
                
                st.write("")
                st.write('Age')
                st.write("")
                
                col8, col9, col10 = st.columns(3)
                
                with col8:
                    inf_17 = st.slider("Nombre de personnes ayant 17 ans ou moins", min_value=0, max_value=10, value=0)
                with col9:
                    entre_18_60 = st.slider("Nombre de personnes entre 18 et 60 ans", min_value=0, max_value=10, value=0)
                with col10:
                    entre_61_95= st.slider("Nombre de personnes ayant plus de 61 ans", min_value=0, max_value=10, value=0)

            
                input_data_random = pd.DataFrame([{
                'agg': agg,
                'atm': atm,
                'heure': heure.hour + heure.minute / 60,  # Convertir l'heure en une valeur numérique
                'nbv': nbv,
                'vma': vma,
                'homme': homme,
                'femme': femme,
                '0-17': inf_17,
                '18-60': entre_18_60,
                '61-95': entre_61_95
            }])

                # Bouton pour la prédiction
                if st.button("⚡ Prédire"):
                    # Appliquer le ColumnTransformer
                    new_data_transformed_random = preprocessor_random.transform(input_data_random)

                    # Faire la prédiction avec le modèle
                    prediction_random = model.predict(new_data_transformed_random)

                    # Dictionnaire pour renommer les prédictions
                    st.write('Le blessé le plus grave lors de cet accident est:')
                    labels_random = {0: "un blessé_léger", 1: "Un blessé grave", 2: "Un tué"}
                    # La prédiction est un tableau, extraire la valeur
                    if isinstance(prediction_random, (list, np.ndarray)):  
                        prediction_random = prediction_random[0]

                    # Afficher le résultat
                    st.success(f"📌 La prédiction est : **{labels_random.get(prediction_random, 'Inconnu')}**")




            else:
                    
                model = joblib.load("model_boitier_XGBoost_2_classes.joblib")  # Remplace par ton chemin réel
                preprocessor = joblib.load("preprocessor.joblib")
                # Définition de la classe pour la transformation de l'heure
            
                

                st.write("")
                st.write('Conditions')
                st.write("")
                # Widgets pour choisir les valeurs des features
                col1, col2, col3 = st.columns(3)
                with col1:
                    agg = st.selectbox("Localisation", ["Hors_agglomération", "En agglomération"], key="agg_box")
                with col2:
                    atm = st.selectbox("Atmosphère", ['temps_normal', 'Temps_pluvieux', 'Temps_couvert', 'Autre'], key="atm_slider")
                with col3:
                    heure = st.time_input("Sélectionnez une heure", value=time(12, 0), key="heure_timer")

                st.write("")
                st.write('Route')
                st.write("")
                
                col4,col5 = st.columns(2)
                with col4:
                    nbv= st.slider("Nombre de voies", min_value=0, max_value=10, value=0, key="nbv_slider")
                with col5:
                    vma = st.selectbox("Vitesse maximale autorisée", [20, 30, 50, 60, 70, 80, 90,100, 110, 130],key="vma_box")

                st.write("")
                st.write('Sexe')
                st.write("")

                col6,col7 = st.columns(2)
                with col6:
                    homme = st.slider("Nombre d'hommes", min_value=0, max_value=10, value=0, key="homme_slider")
                with col7:
                    femme = st.slider("Nombre de femmes", min_value=0, max_value=10, value=0,key="femme_slider")
                
                st.write("")
                st.write('Type de véhicules')
                st.write("")
                
                col8,col9 = st.columns(2)
                with col8:
                    PL = st.slider("Nombre de poids lourds", min_value=0, max_value=10, value=0, key = 'PL_slider')
                with col9:
                    bus_car= st.slider("Nombre de bus/cars", min_value=0, max_value=10, value=0, key = 'bus_car_slider')
                
                col10,col11 = st.columns(2)
                with col10:
                    VL_VU= st.slider("Nombre de véhicules légers/ tilitaires", min_value=0, max_value=10, value=0, key ='VL_VU_slider')
                with col11:
                    deux_roues= st.slider("Nombre de deux roues", min_value=0, max_value=10, value=0, key = 'deux roues_slider')

                st.write("")
                st.write('Age')
                st.write("")

                col12, col13, col14 = st.columns(3)
                with col12:
                    inf_17 = st.slider("Nombre de personnes ayant 17 ans ou moins", min_value=0, max_value=10, value=0, key="inf_17_slider")
                with col13:
                    entre_18_60 = st.slider("Nombre de personnes entre 18 et 60 ans", min_value=0, max_value=10, value=0,key="entre18_60_slider")
                with col14:
                    entre_61_95= st.slider("Nombre de personnes ayant plus de 61 ans", min_value=0, max_value=10, value=0,key="entre_61_95_slider")

            
                input_data = pd.DataFrame([{
                'agg': agg,
                'atm': atm,
                'heure': heure.hour + heure.minute / 60,  # Convertir l'heure en une valeur numérique
                'nbv': nbv,
                'vma': vma,
                'homme': homme,
                'femme': femme,
                'PL': PL,
                'bus_car': bus_car,
                'VL_VU': VL_VU,
                '2roues_3roues_quad': deux_roues,
                '0-17': inf_17,
                '18-60': entre_18_60,
                '61-95': entre_61_95
            }])

                # Bouton pour la prédiction
                if st.button("⚡ Prédire", key= 'xgboost'):
                    new_data_transformed = preprocessor.transform(input_data) # Appliquer le ColumnTransformer
                
                    # Faire la prédiction avec le modèle
                    prediction = model.predict(new_data_transformed)
                    st.write('Le blessé le plus grave lors de cet accident est:')
                    labels = {0: "Non urgent (blessé ou tué)", 1: "Urgent (blessé grave)", 2: "Un tué"}

                # La prédiction est un tableau, extraire la valeur
                    if isinstance(prediction, (list, np.ndarray)):  
                        prediction = prediction[0]

                    # Afficher le résultat avec le label correspondant
                    st.success(f"📌 La prédiction est : **{labels.get(prediction, 'Inconnu')}**")

elif st.session_state.page == "Conclusion":
    st.header('Conclusion')
    

    texte = """
<div style="background-color: #e8f5e9; border: 2px solid #81c784; padding: 20px; border-radius: 10px;">

#### Analyse des résultats et perspectives

Au cours de ce projet, nous avons testé plusieurs modèles de machine learning et constaté que les **résultats pouvaient varier de manière significative**, non seulement selon le type de modèle utilisé, mais également en fonction des **paramètres choisis** et des **variables sélectionnées**.

---

#### Une application concrète à fort potentiel

Dans une mise en œuvre réelle, un tel modèle pourrait représenter un **véritable atout pour les services de secours**, notamment via une **intégration dans un boîtier embarqué**. Ce dispositif permettrait d’évaluer instantanément la gravité d’un accident et de déclencher **le bon niveau d’intervention**, optimisant ainsi la réactivité des secours.

Cependant, pour qu’un tel outil soit déployé efficacement, **une amélioration des performances actuelles serait indispensable**. Plusieurs axes d'amélioration sont envisageables :

- **Retravailler les données**  
  Un modèle performant repose avant tout sur des données de qualité. Il serait pertinent d’explorer certaines **variables mises de côté** jusqu’ici, ou encore de **créer de nouvelles variables dérivées** afin d’enrichir le jeu de données et d’apporter davantage de valeur prédictive.

- **Élargir la période d’analyse**  
  Si nous avons volontairement restreint notre étude aux années **2019 à 2023**, il pourrait être intéressant d’**inclure des années supplémentaires** afin d’observer l’impact sur les performances des modèles.

- **Optimiser l’utilisation en conditions réelles**  
  Une orientation plus ciblée sur les **variables pouvant être collectées par un boîtier** permettrait de tester des modèles et configurations spécifiques à ce cas d’usage. L’objectif serait d’obtenir **des scores de Recall et de F1-score significativement plus élevés**, en garantissant une réelle utilité opérationnelle.

- **Aller plus loin dans la prédiction**  
  Aujourd’hui, nos modèles se concentrent sur la **gravité maximale d’un accident**. Une évolution possible consisterait à **prédire le nombre de blessés par niveau de gravité**, apportant ainsi une information plus fine pour la prise de décision des secours.

---

#### Une première mise en pratique enrichissante

Ce projet a été pour nous l’occasion de **mettre en œuvre l’ensemble des compétences acquises au cours de notre formation** :  
- Analyse, compréhension et préparation d’un jeu de données  
- Mise en place et entraînement de modèles de machine learning  
- Interprétation critique des résultats  

C’est une expérience riche qui nous a permis de lier théorie et application concrète.

</div>
"""

    st.markdown(texte, unsafe_allow_html=True)
