Structure des dossiers
-------------------------


```
Racine/
├── app/
│   └── streamlit_app.py          
├── src/
│   ├── __init__.py
│   ├── data_prep.py              
│   ├── features.py               
│   └── models.py                 
├── processed/
│   └── dbz.csv                   
├── notebooks/
│   ├── 00_download_and_save_data.ipynb
│   ├── 01_enedis_preparation.ipynb
|   ├── 012_ban.ipynb
│   ├── 02_ademe_preprocessing.ipynb
│   ├── 03_intersection_enedis_dpe.ipynb
│   ├── 04_enedis_ban_link.ipynb
│   ├── 05_comparison_enedis_dpe.ipynb
├── requirements.txt
└── README.txt 
```

---


Lancer l'application Streamlit (en local)
--------------------------------------------

Prérequis :
- Python installé (3.9+ recommandé)
- Le fichier processed/dbz.csv présent dans le dossier `processed/`.

Étapes :

1) Ouvrir un terminal dans le dossier racine du projet.

2) Installer les dépendances :

   pip install -r requirements.txt

3) Lancer l’application Streamlit :

   streamlit run app/streamlit_app.py

4) Un navigateur s’ouvre automatiquement (ou va à l’adresse indiquée dans le terminal)

Les notebooks
--------------------------------------------

Les différents notebooks présent `notebooks/`, contiennent les scripts qui permettent : 

**00** - Télécharger les BDD a partir d'appel sur les API

**01** & **012**- Préparation sur les données de Edenis (pour la partie BAN prendre la base et l'API en local via ce [git](https://github.com/BaseAdresseNationale/addok-docker)). 


**02** - Préparation sur les données de Ademe

**03** - Fais l'intersection entre les adresses BAN présentes de Enedis et Ademe

**04** - Permet de récupérer uniquement les informations liées aux adresses Enedis qui nous intéresse.

**05** - Fais le matching entre nos 2 bases sur les adresses et les années de dernière modification du DPE.

---