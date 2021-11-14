from math import nan
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import statistics
import folium
import s3fs
import random
import os
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from streamlit_echarts import st_echarts
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts
st.set_page_config(page_title="Immo", page_icon="https://www.camping-croisee-chemins.fr/wp-content/uploads/2021/02/Recyclage.png",layout="wide")
st.title("Immo Datasets dashboards | PONNOU Wilfried")
#################################################################### DATA ###############################################################################""
datasetchoice=st.sidebar.selectbox("Choisissez l'année (NE CHANGEZ PAS L'ANNEE SI VOUS RUNNEZ SUR ST SHARE, CELA VA CRASHER L'APP)", ["2020","2019","2018","2017","2016"] )


# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)

# Retrieve file contents.
# Uses st.cache to only rerun when the query changes or after 10 min.
#@st.cache(ttl=600)
#def read_file(filename):
#    with fs.open(filename) as f:
#        return f.read().decode("utf-8")
dataset2020='https://jtellier.fr/DataViz/full_2020.csv'
dataset2019='https://jtellier.fr/DataViz/full_2019.csv'
dataset2018='https://jtellier.fr/DataViz/full_2018.csv'
dataset2017='https://jtellier.fr/DataViz/full_2017.csv'
dataset2016='https://jtellier.fr/DataViz/full_2016.csv'


DEPARTEMENTS = {
    '1': 'Ain', 
    '2': 'Aisne', 
    '3': 'Allier', 
    '4': 'Alpes-de-Haute-Provence', 
    '5': 'Hautes-Alpes',
    '6': 'Alpes-Maritimes', 
    '7': 'Ardèche', 
    '8': 'Ardennes', 
    '9': 'Ariège', 
    '10': 'Aube', 
    '11': 'Aude',
    '12': 'Aveyron', 
    '13': 'Bouches-du-Rhône', 
    '14': 'Calvados', 
    '15': 'Cantal', 
    '16': 'Charente',
    '17': 'Charente-Maritime', 
    '18': 'Cher', 
    '19': 'Corrèze', 
    '2A': 'Corse-du-Sud', 
    '2B': 'Haute-Corse',
    '21': 'Côte-d\'Or', 
    '22': 'Côtes-d\'Armor', 
    '23': 'Creuse', 
    '24': 'Dordogne', 
    '25': 'Doubs', 
    '26': 'Drôme',
    '27': 'Eure', 
    '28': 'Eure-et-Loir', 
    '29': 'Finistère', 
    '30': 'Gard', 
    '31': 'Haute-Garonne', 
    '32': 'Gers',
    '33': 'Gironde', 
    '34': 'Hérault', 
    '35': 'Ille-et-Vilaine', 
    '36': 'Indre', 
    '37': 'Indre-et-Loire',
    '38': 'Isère', 
    '39': 'Jura', 
    '40': 'Landes', 
    '41': 'Loir-et-Cher', 
    '42': 'Loire', 
    '43': 'Haute-Loire',
    '44': 'Loire-Atlantique', 
    '45': 'Loiret', 
    '46': 'Lot', 
    '47': 'Lot-et-Garonne', 
    '48': 'Lozère',
    '49': 'Maine-et-Loire', 
    '50': 'Manche', 
    '51': 'Marne', 
    '52': 'Haute-Marne', 
    '53': 'Mayenne',
    '54': 'Meurthe-et-Moselle', 
    '55': 'Meuse', 
    '56': 'Morbihan', 
    '57': 'Moselle', 
    '58': 'Nièvre', 
    '59': 'Nord',
    '60': 'Oise', 
    '61': 'Orne', 
    '62': 'Pas-de-Calais', 
    '63': 'Puy-de-Dôme', 
    '64': 'Pyrénées-Atlantiques',
    '65': 'Hautes-Pyrénées', 
    '66': 'Pyrénées-Orientales', 
    '67': 'Bas-Rhin', 
    '68': 'Haut-Rhin', 
    '69': 'Rhône',
    '70': 'Haute-Saône', 
    '71': 'Saône-et-Loire', 
    '72': 'Sarthe', 
    '73': 'Savoie', 
    '74': 'Haute-Savoie',
    '75': 'Paris', 
    '76': 'Seine-Maritime', 
    '77': 'Seine-et-Marne', 
    '78': 'Yvelines', 
    '79': 'Deux-Sèvres',
    '80': 'Somme', 
    '81': 'Tarn', 
    '82': 'Tarn-et-Garonne', 
    '83': 'Var', 
    '84': 'Vaucluse', 
    '85': 'Vendée',
    '86': 'Vienne', 
    '87': 'Haute-Vienne', 
    '88': 'Vosges', 
    '89': 'Yonne', 
    '90': 'Territoire de Belfort',
    '91': 'Essonne', 
    '92': 'Hauts-de-Seine', 
    '93': 'Seine-Saint-Denis', 
    '94': 'Val-de-Marne', 
    '95': 'Val-d\'Oise',
    '971': 'Guadeloupe', 
    '972': 'Martinique', 
    '973': 'Guyane', 
    '974': 'La Réunion', 
    '976': 'Mayotte',
}

######################################################################## FONCTIONS ################################################################################""



def timer(func):
    def wrapper(*args,**kwargs):
        
        with open("logs.txt","a") as f:
            before=time.time()
            timestamp=datetime.datetime.now()
            val=func(*args,**kwargs)
            f.write("Function "+ func.__name__ +" started at "+ str(timestamp)+" and took: "+str(time.time()-before)+" seconds to execute\n")
        return val
    return wrapper

#@timer
@st.cache
def loader_preprocesser(dataset):
    # IF YOU ONLY WANT A SAMPLE!#####################################################################################################################
    num_lines = 1500000
    size = int(num_lines / 10)
    skip_idx = random.sample(range(1, num_lines), num_lines - size)
    dataframe = pd.read_csv(dataset, skiprows=skip_idx)
    #IF YOU WANT ALL THE DATA COMMENT ALL THE ABOVE PART OF THIS FUNCTION!(and uncomment the line 171)##################################################
    #dataframe=pd.read_csv(dataset)
    df_notpreprocessed=dataframe
    dataframe['date_mutation']=pd.to_datetime(dataframe['date_mutation'])
    dataframe=dataframe.drop(['adresse_suffixe','ancien_code_commune','ancien_nom_commune','ancien_id_parcelle','numero_volume','lot1_numero','lot1_surface_carrez'
    ,'lot2_numero','lot2_surface_carrez','lot3_numero','lot3_surface_carrez','lot4_numero','lot4_surface_carrez','lot5_numero','lot5_surface_carrez',
    'code_type_local','code_nature_culture','code_nature_culture_speciale'],axis=1)
    dataframe['adresse_code_voie']=str(dataframe['adresse_code_voie'])
    dataframe['code_commune']=dataframe['code_commune'].astype(str)
    dataframe['nom_commune']=dataframe['nom_commune'].astype(str)
    dataframe['code_departement']=dataframe['code_departement'].astype(str)
    dataframe['id_parcelle']=dataframe['id_parcelle'].astype(str)
    dataframe['nature_culture']=dataframe['nature_culture'].astype(str)
    dataframe['code_postal']=dataframe['code_postal'].fillna(0).astype(int)
    dataframe['adresse_numero']=dataframe['adresse_numero'].fillna(0).astype(int)
    dataframe['nombre_pieces_principales']=dataframe['nombre_pieces_principales'].fillna(0).astype(int)
    dataframe['surface_terrain']=dataframe['surface_terrain'].fillna(0).astype(int)
    dataframe['valeur_fonciere']=dataframe['valeur_fonciere'].fillna(0).astype(int)
    dataframe['nature_culture_speciale']=dataframe['nature_culture_speciale'].astype(str)
    dataframe['surface_totale']=dataframe['surface_reelle_bati']+dataframe['surface_terrain']
    deps=[]

    for l in dataframe['code_departement']:
        deps.append(DEPARTEMENTS[l])
    dataframe['nom_departement']=deps
    
    dataframe=dataframe.dropna(subset=["latitude","longitude","surface_terrain"])


    return dataframe,df_notpreprocessed

def mapparam(sliderparam,dflon,dflat,coordparam,start,stop,begin):
    param_to_filter = st.slider(sliderparam, start, stop, begin)
    d={'lon':dflon[coordparam.dt.month == param_to_filter],'lat':dflat[coordparam.dt.month == param_to_filter]}
    filtered_data = pd.DataFrame(data=d)
    st.subheader(f'Map of all orders at {param_to_filter}')
    st.map(filtered_data)

def histplotter(column,title,xlabel,ylabel):
    fig,ax= plt.subplots()
    ax.hist(column)
    ax.set_xticklabels(df['nature_mutation'].unique(),rotation=45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)
def list_dict_category_counter(dfcolumn):
    values=(df[dfcolumn].value_counts()).tolist()
    names=df[dfcolumn].value_counts().index.tolist()
    l=[]
    for value,name in zip(values,names):
        l.append({'value':value,'name':name})
    return l
def mean_calculator(selected_city,selected_type,selected_rooms,selected_subtype):
    if selected_type in ('Maison','Appartement'):
        valeurs_foncieres_par_ville_et_type=(df.loc[(df['nom_commune']==selected_city) & (df['type_local']==selected_type) & (df['surface_totale']!=0) & (df['nombre_pieces_principales']>=selected_rooms),['valeur_fonciere']]).stack()
        surface_bati_par_ville_et_type=(df.loc[(df['nom_commune']==selected_city) & (df['type_local']==selected_type) & (df['surface_reelle_bati']!=0) & (df['nombre_pieces_principales']>=selected_rooms),['surface_reelle_bati']]).stack()
        return meanreturner(valeurs_foncieres_par_ville_et_type,surface_bati_par_ville_et_type,selected_type,selected_subtype,selected_city,selected_rooms)
    elif selected_type in ('Dépendance','Local industriel. commercial ou assimilé'):
        valeurs_foncieres_par_ville_et_type=(df.loc[(df['nom_commune']==selected_city) & (df['type_local']==selected_type) & (df['surface_totale']!=0), ['valeur_fonciere']]).stack()
        surface_totale_par_ville_et_type=(df.loc[(df['nom_commune']==selected_city) & (df['type_local']==selected_type) & (df['surface_totale']!=0), ['surface_terrain','surface_reelle_bati']]).stack()
        return meanreturner(valeurs_foncieres_par_ville_et_type,surface_totale_par_ville_et_type,selected_type,selected_subtype,selected_city,selected_rooms)
    else:
        valeurs_foncieres_par_ville_et_type=(df.loc[(df['nom_commune']==selected_city) & (df['nature_culture']==selected_subtype) , ['valeur_fonciere']]).stack()
        surface_totale_par_ville_et_type=(df.loc[(df['nom_commune']==selected_city) & (df['nature_culture']==selected_subtype) , ['surface_terrain','surface_reelle_bati']]).stack()
        print(len(valeurs_foncieres_par_ville_et_type))
        return meanreturner(valeurs_foncieres_par_ville_et_type,surface_totale_par_ville_et_type,selected_type,selected_subtype,selected_city,selected_rooms)

def meanreturner(valeurs_foncieres_par_ville_et_type,surface,selected_type,selected_subtype,selected_city,selected_rooms):
    sub_liste=[]
    if len(valeurs_foncieres_par_ville_et_type)!=0:
        v_surface_manquantes=0
        for v,s in zip(valeurs_foncieres_par_ville_et_type,surface):
            if s!=0:
                sub_liste.append(v/s)
            else:
                v_surface_manquantes+=1
        if v_surface_manquantes==len(valeurs_foncieres_par_ville_et_type):
            returntext=("Malheureusement nous n'avons pas de données sur les surfaces de ces biens pour calculer le prix au m², nous avons seulement la carte ci-dessous !")
        else:
            if selected_type in ('Dépendance','Local industriel. commercial ou assimilé'):
                returntext=("Le prix moyen au m² à "+str(selected_city)+" pour des biens de type "+str(selected_type)+" est de "+str(round(statistics.mean(sub_liste)))+" € en "+str(datasetchoice))
            elif selected_type in ('Maison','Appartement'):
              returntext=("Le prix moyen au m² à "+str(selected_city)+" pour des biens de type "+str(selected_type)+" avec "+str(selected_rooms)+"(ou plus) pièce(s) principale(s) est de "+str(round(statistics.mean(sub_liste)))+" € en "+str(datasetchoice))
            else:
                returntext=("Le prix moyen au m² à "+str(selected_city)+" pour des biens de type "+str(selected_type)+" / "+str(selected_subtype)+" est de "+str(round(statistics.mean(sub_liste)))+" € en "+str(datasetchoice))
    else:
        returntext=("Malheureusement nous n'avons pas de données pour ce genre de bien sur cette ville pour ce type de bien !")  
    return returntext 


def piechartplotter(d,title,subtitle):
    options = {
        "title": {"text": title, "subtext": subtitle, "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"orient": "vertical", "left": "left",},
        "series": [
            {
                "name": "Nombre de :",
                "type": "pie",
                "radius": "50%",
                "data": d,
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                        }
                    },
                }
            ],
        }
    return st_echarts(options=options, height="600px",)

def barchartplotter(xaxis,maintitle,subtitle,dfcolumn):
    b = (
        Bar()
        .add_xaxis(xaxis)
        .add_yaxis(
            subtitle, dfcolumn.groupby(df.date_mutation.dt.month).count().tolist()
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=maintitle, subtitle=str(datasetchoice)
            ),
            toolbox_opts=opts.ToolboxOpts(),
        )
    )
    st_pyecharts(b)    

def adaptativebarchartplotter(dfcolumn,title):
    option1 = {
            "title": {
                "left": 'center',
                "text": title + str(datasetchoice),
            },
            "tooltip": {"trigger": "item"},
            "dataZoom": [
                    {
                        "type": 'slider',
                        "start": 0,
                        "end": 2
                    }
                ],
            "xAxis": {
                "type": "category",
                "data": dfcolumn.value_counts().index.tolist()
            },
            "yAxis": {"type": "value"},
            "series": [{"data": dfcolumn.value_counts().tolist(), "type": "bar"}],
        }
    return st_echarts(options=option1)
#def squarredprice()
def foliumapplotter(map_data,selected_type):
    if selected_type in('Maison','Appartement','Dépendance','Local industriel. commercial ou assimilé'):
        map_data=map_data.fillna(0)
        m = folium.Map(location=[map_data['latitude'].mean(),map_data['longitude'].mean()], zoom_start=13)
        marker_cluster = MarkerCluster().add_to(m)
        for index,row in map_data.iterrows():
            folium.Marker(
                location=[row['latitude'],row['longitude']], #position, latitude/longitude
                popup=str(int(row['valeur_fonciere']))+" € pour "+str(int(row['surface_reelle_bati']))+" m², "+str(int(row['surface_terrain']))+" m² de terrain et "+str(int(row['nombre_pieces_principales']))+" pièce(s) principale(s), type= "+str(row['type_local']),           #afficher la classe du déchet si on clique dessus

                ).add_to(marker_cluster)
        folium_static(m)
    else:
        map_data=map_data.fillna(0)
        m = folium.Map(location=[map_data['latitude'].mean(),map_data['longitude'].mean()], zoom_start=13)
        marker_cluster = MarkerCluster().add_to(m)
        for index,row in map_data.iterrows():
            folium.Marker(
                location=[row['latitude'],row['longitude']], #position, latitude/longitude
                popup=str(int(row['valeur_fonciere']))+" € pour "+str(int(row['surface_reelle_bati']))+" m², "+str(int(row['surface_terrain']))+" m² de terrain et de type= "+str(row['nature_culture']),           #afficher la classe du déchet si on clique dessus

                ).add_to(marker_cluster)
        folium_static(m)

def prixmoyenaumcarre(types,subtypes):
    with st.form('Form1'):
        selected_city=st.selectbox('Choisissez',df['nom_commune'].unique(), key=1)
        selected_type=st.selectbox("Choisissez le type de bien recherché",types,key=2)
        if selected_type in ('Maison','Appartement'):
            selected_subtype=None
            selected_rooms=st.slider(label='Nombre minimal de pièces principales (Choisissez un nombre petit pour commencer)', min_value=0, max_value=15,value=2, key=3)
            if st.checkbox("Peu importe le nombre de pieces"):
                selected_rooms=0
        elif selected_type in ('Dépendance','Local industriel. commercial ou assimilé'):
            selected_subtype=None
            selected_rooms=None
            st.write("Cliquez sur le bouton ci-dessous !")
        else:
            selected_rooms=None
            selected_subtype=st.selectbox("Choisissez le type de culture recherchée",subtypes,key=3)
            st.write("Cliquez sur le bouton ci-dessous !")
        submitted1 = st.form_submit_button('Rechercher')
    if selected_type in ('Maison','Appartement'):
        st.write(mean_calculator(selected_city,selected_type,selected_rooms,selected_subtype))
        st.write("Cliquez sur les icones pour voir les transactions effectuées !")
        map_data = (df.loc[(df['nom_commune']==selected_city) & (df['type_local']==selected_type) & (df['nombre_pieces_principales']>=selected_rooms),['longitude','latitude','valeur_fonciere','surface_reelle_bati','surface_terrain','nombre_pieces_principales','type_local']])
        
        if not map_data.empty:
            foliumapplotter(map_data,selected_type)
        else:
            st.write("Nous n'avons pas de données donc pas de cartes 😞")
    elif selected_type in ('Dépendance','Local industriel. commercial ou assimilé'):       
        st.write(mean_calculator(selected_city,selected_type,selected_rooms,selected_subtype))
        st.write("Cliquez sur les icones pour voir les transactions effectuées !")
        map_data = (df.loc[(df['nom_commune']==selected_city) & (df['type_local']==selected_type),['longitude','latitude','valeur_fonciere','surface_reelle_bati','surface_terrain','nombre_pieces_principales','type_local']])

        if not map_data.empty:
            foliumapplotter(map_data,selected_type)
        else:
            st.write("Nous n'avons pas de données donc pas de cartes 😞")
    else:
        st.write(mean_calculator(selected_city,selected_type,selected_rooms,selected_subtype))
        st.write("Cliquez sur les icones pour voir les transactions effectuées !")
        map_data = (df.loc[(df['nom_commune']==selected_city) & (df['nature_culture']==selected_subtype),['longitude','latitude','valeur_fonciere','surface_reelle_bati','surface_terrain','nature_culture','type_local']])

        if not map_data.empty:
            foliumapplotter(map_data,selected_type)
        else:
            st.write("Nous n'avons pas de données donc pas de cartes 😞")

########################################################################### STREAMLIT APP#####################################################################

df,df_notpreprocessed=loader_preprocesser(locals()['dataset'+str(datasetchoice)])
    
if st.checkbox("Voir le dataset preprocessé"): 
    st.write(df.head(10))
    
if st.checkbox("Voir le dataset non preprocessé"):
    st.write(df_notpreprocessed.head(10))
    
if st.checkbox("Voir la carte des ventes"):
    mapparam('Month',df['longitude'],df['latitude'],df['date_mutation'],1,12,1)
    
plotchoice=st.selectbox("Choisissez la représentation",["Evolution des ventes durant l'année","Prix moyen par m²","Histogramme des natures de mutations","Camembert des types de biens vendus","Histogramme des natures de culture des biens vendus","Départements ayant le plus de transactions immobilières en "+ str(datasetchoice)])
    
if plotchoice=="Histogramme des natures de mutations":
    st.header('Histogramme des natures de mutation en '+str(datasetchoice))
    st.bar_chart(df['nature_mutation'].value_counts())
        

elif plotchoice=="Camembert des types de biens vendus":
    d=list_dict_category_counter('type_local')
    piechartplotter(d,"Camembert des types de biens vendus en "+str(datasetchoice),"Répartition des types de biens vendus")
    
elif plotchoice=="Histogramme des natures de culture des biens vendus":
    df_sorted=(df['nature_culture'].value_counts(normalize=True).mul(100).round(1))
    fig,ax= plt.subplots()
    ax=df_sorted.plot(kind='bar')
    ax.set_ylabel('Percentage')
    st.pyplot(fig)
    
elif plotchoice=="Evolution des ventes durant l'année":
    x_axis=("Janvier",'Février','Mars','Avril','Mai','Juin','Juillet','Août','Septembre','Octobre','Novembre','Décembre')
    barchartplotter(x_axis,"Nombre de transactions immobilières \n durant l'année "+str(datasetchoice),"Nombre de transactions",df['date_mutation'])

elif plotchoice=="Départements ayant le plus de transactions immobilières en "+ str(datasetchoice):
    adaptativebarchartplotter(df['nom_departement'],"Départements ayant le plus de transactions immobilières en ")
    st.write('utilisez le slider pour voir les chiffres des autres départements !')

elif plotchoice=="Prix moyen par m²": 
    st.header("Vous découvrirez ici le prix moyen par m² selon les données du dataset dans votre ville et selon le type de bien recherché, en "+str(datasetchoice))
    types_locals=(df['type_local'].dropna().unique().tolist())
    types_locals.append('cultures')
    subtypes=(df['nature_culture'].dropna().unique().tolist())
    prixmoyenaumcarre(types_locals,subtypes)

