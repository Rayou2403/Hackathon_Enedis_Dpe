# app/streamlit_app.py
import streamlit as st
import pandas as pd

import src.data_prep as data_prep
import src.features as features
import src.models as models

st.set_page_config(
    page_title="Hackathon DPE x Enedis",
    page_icon="💡",
    layout="wide",
)


# ----------------------------- DATA & MODEL -------------------------
@st.cache_data
def get_base_df():
    # On ne filtre plus sur l'écart d'années ici : matching déjà fait dans dbz.csv
    return data_prep.base_clean_df()


@st.cache_data
def get_feat_df():
    df = get_base_df()
    df = features.add_conso_features(df)
    return df


@st.cache_resource
def get_model():
    return models.load_model()



# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================


# ----------------------------- PAGES --------------------------------
def page_intro():
    st.title("Analyse DPE & Consommations Réelles — Enedis")

    df_base = get_base_df()
    df = get_feat_df()
    nb_logements_base = len(df_base)
    nb_logements = len(df)
    nb_adresses = (
        df_base["address_ban"].nunique()
        if "address_ban" in df_base.columns
        else None
    )

    # ---- KPI ----
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Logements appariés DPE × Enedis (brut)",
        f"{nb_logements_base:,}".replace(",", " "),
    )
    col2.metric(
        "Logements exploitables (après nettoyage)",
        f"{nb_logements:,}".replace(",", " "),
    )
    if nb_adresses is not None:
        col3.metric("Adresses uniques (BAN)", f"{nb_adresses:,}".replace(",", " "))

    # ---- CONTEXTE ----
    st.markdown(
        """
        Cette application a été développée dans le cadre du **Hackathon Enedis — DPE & énergie**,
        afin de mieux comprendre les écarts entre les diagnostics théoriques et les usages
        observés sur le réseau.

        Elle s’appuie sur deux jeux de données :
        - les **consommations réelles** issues des données Enedis,
        - les informations du **Diagnostic de Performance Énergétique (DPE)**,
        appariées via la Base Adresse Nationale (BAN).

        ---
        ### Objectifs de l’outil
        L’interface permet d'explorer trois axes :

        1. **Comparer la consommation réelle** à la consommation estimée par le **DPE**.  
           Cela met en évidence les écarts structurels entre modèle conventionnel et usage réel.

        2. **Analyser l’impact d’un changement de classe énergétique**  
           (ex : passer de **G → F**, ou **E → C**),  
           en estimant :
           - la baisse attendue de consommation réelle,
           - l’économie financière correspondante.

        3. **Prédire la consommation réelle d’un logement**,  
           grâce à un modèle de Machine Learning entraîné sur des centaines de milliers de logements.

        ---
        ### Utilisation
        Les filtres situés dans la colonne de gauche permettent de se restreindre à :
        - une **région**,  
        - un **type de bâtiment** (maison / appartement),  
        - une **période de construction**,  
        - une **classe DPE**.

        Toutes les statistiques et visualisations s’adaptent en temps réel au périmètre sélectionné.
        """
    )


# ===========================================================================================
# ===========================================================================================
# ============================== DPE VS Consommation Réel ===================================
# ===========================================================================================
# ===========================================================================================

def page_dpe_vs_reel():
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px

    st.header("DPE (ADEME) vs Consommation réelle (ENEDIS)")

    # --- LOAD ---
    df_all = get_feat_df()
    df = df_all.copy()

    if "annee_dpe_matched" not in df.columns:
        st.error("Colonne 'annee_dpe_matched' manquante.")
        return

    # ---------------------------------------------------
    # MODE SELECTION
    # ---------------------------------------------------
    mode = st.radio(
        "Mode d’analyse",
        [
            "Adresse unique",
            "Comparaison multi-adresses",
            "Analyse régionale",
            "Graphe des écarts ADEME vs ENEDIS",
        ]
    )

    # ===================================================
    # 1) ADRESSE UNIQUE
    # ===================================================
    if mode == "Adresse unique":

        addresses = sorted(df["address_ban"].dropna().unique())
        adresse_choice = st.selectbox("Adresse :", ["Sélectionnez une adresse"] + addresses)

        if adresse_choice == "Sélectionnez une adresse":
            st.info("Sélectionnez une adresse.")
            return

        df_addr = df[df["address_ban"] == adresse_choice]

        years = sorted(df_addr["annee_dpe_matched"].dropna().unique())
        selected_years = st.multiselect("Années :", years, default=years)

        if not selected_years:
            st.info("Sélectionnez une année.")
            return

        df_year = df_addr[df_addr["annee_dpe_matched"].isin(selected_years)]

        # --- group ---
        plot_df = (
            df_year.groupby("annee_dpe_matched")[["conso_dpe_kwh", "conso_reelle_kwh"]]
            .mean()
            .reset_index()
            .rename(columns={
                "annee_dpe_matched": "Année",
                "conso_dpe_kwh": "ADEME (kWh/an)",
                "conso_reelle_kwh": "ENEDIS (kWh/an)"
            })
        )

        # --- graph ---
        fig = go.Figure()

        fig.add_bar(
            x=plot_df["Année"],
            y=plot_df["ADEME (kWh/an)"],
            name="ADEME (DPE)", marker_color="#4C78A8",
            width=0.28, marker_line_width=1.2
        )
        fig.add_bar(
            x=plot_df["Année"],
            y=plot_df["ENEDIS (kWh/an)"],
            name="ENEDIS (réel)", marker_color="#F58518",
            width=0.28, marker_line_width=1.2
        )

        fig.update_layout(
            barmode="group",
            title=f"Consommations ADEME vs ENEDIS — {adresse_choice}",
            xaxis_title="Année",
            yaxis_title="kWh/an",
            template="plotly_white",
            bargap=0.35,
            bargroupgap=0.20,
        )
        fig.update_xaxes(tickmode="linear", dtick=1)

        st.plotly_chart(fig, use_container_width=True)

        # --- table ---
        recap = df_year[[
            "annee_dpe_matched", "etiquette_dpe", "conso_dpe_kwh", "conso_reelle_kwh"
        ]].copy()

        recap.rename(columns={
            "annee_dpe_matched": "Année",
            "etiquette_dpe": "Classe DPE",
            "conso_dpe_kwh": "ADEME (kWh/an)",
            "conso_reelle_kwh": "ENEDIS (kWh/an)"
        }, inplace=True)

        recap = recap.sort_values("Année").astype(str)

        st.subheader("Détails des valeurs")
        st.dataframe(recap, hide_index=True, use_container_width=True)

    # ===================================================
    # 2) COMPARAISON MULTI-ADRESSES (HISTOGRAMME)
    # ===================================================
    elif mode == "Comparaison multi-adresses":

        addresses = sorted(df["address_ban"].dropna().unique())
        choix = st.multiselect("Adresses (max 5) :", addresses, max_selections=5)

        if not choix:
            st.info("Sélectionnez une adresse.")
            return

        df_multi = df[df["address_ban"].isin(choix)]

        summary = (
            df_multi.groupby("address_ban")[["conso_dpe_kwh", "conso_reelle_kwh"]]
            .mean()
            .reset_index()
            .rename(columns={
                "conso_dpe_kwh": "ADEME (kWh/an)",
                "conso_reelle_kwh": "ENEDIS (kWh/an)"
            })
        )

        # --- graph amélioré ---
        fig = go.Figure()

        fig.add_bar(
            x=summary["address_ban"],
            y=summary["ADEME (kWh/an)"],
            name="ADEME (DPE)",
            marker_color="#4C78A8",
            width=0.35,
            marker_line_width=1.1
        )
        fig.add_bar(
            x=summary["address_ban"],
            y=summary["ENEDIS (kWh/an)"],
            name="ENEDIS (réel)",
            marker_color="#F58518",
            width=0.35,
            marker_line_width=1.1
        )

        fig.update_layout(
            barmode="group",
            title="Comparaison multi-adresses — ADEME vs ENEDIS",
            xaxis_title="Adresse",
            yaxis_title="kWh/an",
            template="plotly_white",
            bargap=0.28,
            bargroupgap=0.18,
        )

        fig.update_xaxes(tickangle=30)

        st.plotly_chart(fig, use_container_width=True)

        # --- tableau multi-adresses ---
        recap_multi = df_multi[[
            "address_ban", "annee_dpe_matched", "etiquette_dpe",
            "conso_dpe_kwh", "conso_reelle_kwh"
        ]].copy()

        recap_multi.rename(columns={
            "address_ban": "Adresse",
            "annee_dpe_matched": "Année",
            "etiquette_dpe": "Classe DPE",
            "conso_dpe_kwh": "ADEME (kWh/an)",
            "conso_reelle_kwh": "ENEDIS (kWh/an)"
        }, inplace=True)

        recap_multi = recap_multi.sort_values(["Adresse", "Année"]).astype(str)

        st.subheader("Détails multi-adresses")
        st.dataframe(recap_multi, hide_index=True, use_container_width=True)

    # ===================================================
    # 3) ANALYSE RÉGIONALE
    # ===================================================
    elif mode == "Analyse régionale":
        
        if "code_region" not in df.columns:
            st.error("code_region manquant.")
            return

        df_reg = (
            df.groupby("code_region")[["conso_dpe_kwh", "conso_reelle_kwh"]]
            .mean()
            .reset_index()
            .rename(columns={
                "conso_dpe_kwh": "ADEME (kWh/an)",
                "conso_reelle_kwh": "ENEDIS (kWh/an)"
            })
        )

        fig = go.Figure()
        fig.add_bar(
            x=df_reg["code_region"].astype(str),
            y=df_reg["ADEME (kWh/an)"],
            name="ADEME (DPE)", marker_color="#4C78A8"
        )
        fig.add_bar(
            x=df_reg["code_region"].astype(str),
            y=df_reg["ENEDIS (kWh/an)"],
            name="ENEDIS (réel)", marker_color="#F58518"
        )

        fig.update_layout(
            barmode="group",
            title="Consommations ADEME vs ENEDIS par région",
            xaxis_title="Code région INSEE",
            yaxis_title="kWh/an",
            template="plotly_white",
            bargap=0.25,
            bargroupgap=0.20
        )

        st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # 4) GRAPHE DES ÉCARTS
    # ===================================================
    elif mode == "Graphe des écarts ADEME vs ENEDIS":

        df_gap = df.copy()
        df_gap["Écart (kWh/an)"] = df_gap["conso_reelle_kwh"] - df_gap["conso_dpe_kwh"]

        fig = px.histogram(
            df_gap,
            x="Écart (kWh/an)",
            nbins=45,
            title="Distribution des écarts {ENEDIS (réel) − ADEME (prédit)}",
            color_discrete_sequence=["#6A040F"],
            template="plotly_white"
        )

        fig.update_layout(bargap=0.25)

        st.plotly_chart(fig, use_container_width=True)


# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================

def page_impact_dpe():
    import pandas as pd
    import plotly.graph_objects as go

    # ---------- CSS MULTISELECT CLEAN ----------
    custom_css = """
    <style>

        /* Multiselect selected tags */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #2b2b2b !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 4px 10px !important;
            margin: 2px !important;
            font-size: 14px;
        }

        /* Icon inside tags */
        .stMultiSelect [data-baseweb="tag"] svg {
            fill: white !important;
        }

        /* Input area */
        .stMultiSelect > div {
            background-color: #1e1e1e !important;
            border-radius: 10px !important;
            padding: 6px !important;
        }

        /* Dropdown list */
        .stMultiSelect div[role="listbox"] {
            background-color: #1e1e1e !important;
            color: white !important;
        }

        /* Hover option */
        .stMultiSelect div[role="option"]:hover {
            background-color: #333 !important;
            color: white !important;
        }

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # ---------- Style KPI ----------
    kpi_style = """
        <style>
        .kpi-box {
            background-color: #1e1e1e;
            padding: 14px 20px;
            border-radius: 12px;
            text-align: left;
            margin-bottom: 10px;
        }
        .kpi-label {
            color: #bbbbbb;
            font-size: 15px;
        }
        .kpi-value {
            font-size: 30px;
            font-weight: 600;
            color: white;
        }
        .kpi-delta-pos {
            color: #1db954;
            font-weight: 600;
            font-size: 17px;
        }
        .kpi-delta-neg {
            color: #ff4c4c;
            font-weight: 600;
            font-size: 17px;
        }
        </style>
    """
    st.markdown(kpi_style, unsafe_allow_html=True)

    def kpi(label, value, delta=None):
        """KPI custom HTML"""
        if delta is None:
            delta_text = ""
        else:
            arrow = "↑" if delta > 0 else "↓"
            cls = "kpi-delta-pos" if delta > 0 else "kpi-delta-neg"
            delta_text = f"<div class='{cls}'>{arrow} {abs(delta):,.0f} kWh/an</div>"

        html = f"""
        <div class='kpi-box'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            {delta_text}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    # ---------- TITRE ----------
    st.header("Impact d’un changement de classe DPE — Évaluation du gain énergétique")

    df_all = get_feat_df()
    df = df_all.copy()

    # ---------- FILTRES PROS & PROPRES ----------
    st.subheader("Filtres")

    # REGION
    regions = (
        df["code_region"]
        .fillna(-1).astype(int).astype(str)
        .replace("-1", pd.NA)
        .dropna()
        .unique()
    )
    regions = sorted(regions)

    region_choice = st.multiselect(
        "Région (code INSEE)",
        regions,
        default=regions
    )

    df = df[df["code_region"].fillna(-1).astype(int).astype(str).isin(region_choice)]

    # TYPE DE BATIMENT
    types = sorted(df["type_batiment"].dropna().unique())
    type_choice = st.multiselect(
        "Type de bâtiment",
        types,
        default=types
    )
    df = df[df["type_batiment"].isin(type_choice)]

    # PERIODE
    periodes = sorted(df["periode_construction"].dropna().unique())
    periode_choice = st.multiselect(
        "Période de construction",
        periodes,
        default=periodes
    )
    df = df[df["periode_construction"].isin(periode_choice)]

    # CLASSE DPE
    classes_all = list("ABCDEFG")
    classes_pres = sorted(df["etiquette_dpe"].dropna().unique())

    classe_choice = st.multiselect(
        "Classe DPE",
        [c for c in classes_all if c in classes_pres],
        default=[c for c in classes_all if c in classes_pres]
    )
    df = df[df["etiquette_dpe"].isin(classe_choice)]

    if df.empty:
        st.warning("Aucune donnée après filtres.")
        return

    st.caption(f"Nombre de logements retenus : **{len(df):,}**".replace(",", " "))

    # ---------- TABLEAU ----------
    st.subheader("Consommation réelle moyenne par classe DPE")

    tab = (
        df.groupby("etiquette_dpe")["conso_reelle_kwh"]
        .agg(nombre_de_logement="count", conso_moy_kwh="mean", std="std")
        .reset_index()
        .sort_values("etiquette_dpe")
    )
    tab["conso_moy_kwh"] = tab["conso_moy_kwh"].round(0)
    tab["std"] = tab["std"].round(0)

    st.dataframe(tab, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ---------- SIMULATION ----------
    st.subheader("Simulation d’un scénario de rénovation")

    classes_valides = [c for c in list("ABCDEFG") if c in classes_pres]

    if len(classes_valides) < 2:
        st.warning("Pas assez de classes disponibles pour simuler un changement.")
        return

    col1, col2, col3 = st.columns(3)

    classe_depart = col1.selectbox("Classe actuelle :", classes_valides)

    idx = classes_valides.index(classe_depart)
    classe_arrivee = col2.selectbox(
        "Classe visée :", classes_valides, index=max(0, idx - 1)
    )

    prix_kwh = col3.number_input(
        "Prix électricité (€/kWh)",
        min_value=0.05,
        max_value=1.0,
        value=0.20,
        step=0.01,
    )

    # Vérification logique (A est meilleure que B, etc.)
    ordre_dpe = list("ABCDEFG")
    if ordre_dpe.index(classe_arrivee) >= ordre_dpe.index(classe_depart):
        st.warning("La classe visée doit être meilleure (plus proche de A).")
        return

    # ---------- CALCUL ----------
    res = features.gain_entre_classes(df, classe_depart, classe_arrivee)

    # Sécurité : vérifier que res est valide
    if (
        not isinstance(res, dict)
        or "conso_depart" not in res
        or "conso_arrivee" not in res
        or "gain_kwh" not in res
    ):
        st.error(
            "Impossible de calculer le gain énergétique : "
            "pas assez de données pour cette combinaison de classes."
        )
        return

    conso_avant = res["conso_depart"]
    conso_apres = res["conso_arrivee"]
    gain_kwh = res["gain_kwh"]          # kWh/an économisés (moyenne)
    gain_euros = gain_kwh * prix_kwh    # €/an économisés

    # ---------- KPI ----------
    st.subheader("Résultats de la simulation")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi(f"Conso {classe_depart}", f"{conso_avant:,.0f} kWh/an")
    with c2:
        kpi(f"Conso {classe_arrivee}", f"{conso_apres:,.0f} kWh/an")
    with c3:
        kpi("Gain énergétique", f"{gain_kwh:,.0f} kWh/an", delta=gain_kwh)

    # ---------- Mini-Graph ----------
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[conso_avant],
        y=["Avant rénovation"],
        orientation="h",
        marker_color="#4C78A8",
        name="Avant rénovation"
    ))

    fig.add_trace(go.Bar(
        x=[conso_apres],
        y=["Après rénovation"],
        orientation="h",
        marker_color="#F58518",
        name="Après rénovation"
    ))

    fig.update_layout(
        title="Comparaison des consommations (kWh/an)",
        barmode="group",
        height=250,
        template="plotly_white",
        xaxis_title="kWh/an",
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- ANALYSE ÉCONOMIQUE / RENTABILITÉ ----------
    st.markdown("---")
    st.subheader("Analyse économique")

    # Hypothèse : coût de rénovation énergétique par m²
    # On demande la surface pour adapter l'ordre de grandeur
    col_surf, col_info = st.columns([1, 1.5])

    with col_surf:
        surface = st.number_input(
            "Surface du logement (m²)",
            min_value=10.0,
            max_value=500.0,
            value=80.0,
            step=5.0,
        )

    # Nombre de classes gagnées (ex : G -> E = 1, G -> C = 3, etc.)
    nb_classes_gagnees = (
        ordre_dpe.index(classe_depart) - ordre_dpe.index(classe_arrivee)
    )

    # Coût au m² : ordre de grandeur selon l'ampleur de la rénovation
    # 1 classe : ~300 €/m² ; 2 classes : ~350 €/m² ; 3+ classes : ~400 €/m²
    if nb_classes_gagnees <= 1:
        cout_m2 = 300.0
    elif nb_classes_gagnees == 2:
        cout_m2 = 350.0
    else:
        cout_m2 = 400.0

    cout_travaux = surface * cout_m2  # en €

    with col_info:
        st.caption(
            f"Ordre de grandeur utilisé : **~{cout_m2:.0f} € / m²** "
            f"pour un gain de **{nb_classes_gagnees} classe(s)**."
        )

    # Temps de retour sur investissement
    if gain_euros > 0:
        roi_annees = cout_travaux / gain_euros
    else:
        roi_annees = None

    c4, c5, c6 = st.columns(3)
    with c4:
        kpi("Coût estimé des travaux", f"{cout_travaux:,.0f} €")
    with c5:
        kpi("Économie annuelle estimée", f"{gain_euros:,.0f} €/an")
    with c6:
        if (roi_annees is not None) and (roi_annees < 1000):
            kpi("Temps de retour estimé", f"{roi_annees:,.1f} ans")
        else:
            kpi("Temps de retour estimé", "Non pertinent")


# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================

def page_prediction_ml():
    st.header("Prédiction de la consommation réelle")

    df = get_feat_df()
    model_obj = get_model()

    st.markdown(
        """
        Cette interface permet d’estimer la **consommation réelle d’un logement**
        à partir des caractéristiques déclarées et des données observées dans le réseau
        Enedis.

        Le modèle repose sur plusieurs millions de relevés de consommation
        et sur les informations issues des Diagnostics de Performance Énergétique (DPE).
        Il fournit une estimation **plus proche des usages réels** que la valeur
        théorique du DPE, qui est calculée de manière conventionnelle.

        Renseignez les informations du logement ci-dessous afin d’obtenir :

        • une estimation de la consommation annuelle réelle  
        • l’écart par rapport à la valeur indiquée sur le DPE  
        • une estimation indicative de la facture énergétique annuelle

        """
    )


    # Listes des options
    classes = (
        sorted(df["etiquette_dpe"].dropna().unique())
        if "etiquette_dpe" in df.columns
        else list("ABCDEFG")
    )
    types = (
        sorted(df["type_batiment"].dropna().unique())
        if "type_batiment" in df.columns
        else ["Maison", "Appartement", "Autre"]
    )
    regions = (
        df["code_region"]
        .dropna()
        .astype("Int64")
        .astype(str)
        .sort_values()
        .unique()
        if "code_region" in df.columns
        else ["11"]
    )

    # Formulaire utilisateur
    with st.form("form_prediction"):
        col1, col2 = st.columns(2)

        with col1:
            conso_dpe_kwh = st.number_input(
                "Consommation estimée par le DPE (kWh/an)",
                min_value=0.0,
                step=100.0,
                format="%.0f",
            )
            surface_habitable = st.number_input(
                "Surface habitable (m²)",
                min_value=5.0,
                step=1.0,
                value=60.0,
                format="%.0f",
            )
            annee_construction = st.number_input(
                "Année de construction",
                min_value=1900,
                max_value=2050,
                value=1975,
                step=1,
            )

        with col2:
            etiquette_dpe = st.selectbox("Classe DPE", options=classes)
            type_batiment = st.selectbox("Type de bâtiment", options=types)
            code_region = st.selectbox("Code région (INSEE)", options=regions)

        submitted = st.form_submit_button("Prédire la consommation")

        if submitted:
            # Toujours STRING (fixe définitivement le bug sklearn)
            code_region_clean = str(code_region)

            user_data = {
                "conso_dpe_kwh": conso_dpe_kwh,
                "surface_habitable": surface_habitable,
                "annee_construction": annee_construction,
                "etiquette_dpe": etiquette_dpe,
                "type_batiment": type_batiment,
                "code_region": code_region_clean,
            }

            # Prédiction robuste
            y_pred = models.predict_conso(model_obj, user_data)

            st.success(
                f"Consommation réelle estimée : **{y_pred:,.0f} kWh/an**".replace(",", " ")
            )

            # Affichage écart vs DPE
            if conso_dpe_kwh > 0:
                diff = y_pred - conso_dpe_kwh
                pct = diff / conso_dpe_kwh * 100

                st.write(
                    f"Écart par rapport à la valeur DPE : "
                    f"**{diff:,.0f} kWh/an** ({pct:+.1f} %).".replace(",", " ")
                )

            # Facture estimée
            prix_kwh = 0.20
            facture_estimee = y_pred * prix_kwh

            st.write(
                f"Facture annuelle estimée (à {prix_kwh:.2f} €/kWh) : "
                f"**{facture_estimee:,.0f} €**".replace(",", " ")
            )

            st.markdown(
                """
                Cette estimation repose sur les consommations réelles observées sur des logements
                similaires (type, région, période de construction, classe DPE, etc.).
                Elle peut donc différer de la valeur indiquée sur ton DPE, qui est calculée
                de manière conventionnelle.
                """
            )



# ----------------------------- NAVIGATION ---------------------------
def main():
    st.sidebar.title("Tableau de bord")
    page = st.sidebar.radio(
        "",
        [
            "Introduction",
            "DPE vs Conso réelle",
            "Changement de classe DPE",
            "Prédiction ML"
        ],
    )

    if page == "Introduction":
        page_intro()
    elif page == "DPE vs Conso réelle":
        page_dpe_vs_reel()
    elif page == "Changement de classe DPE":
        page_impact_dpe()
    elif page == "Prédiction ML":
        page_prediction_ml()

if __name__ == "__main__":
    main()

