# app/streamlit_app.py
import streamlit as st
import pandas as pd

import src.data_prep as data_prep
import src.features as features
import src.models as models



st.set_page_config(
    page_title="Hackathon DPE x Enedis",
    page_icon="📊",
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

# ----------------------------- FILTRES GLOBAUX ----------------------
def filtre_df(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filtres globaux")
    df_filt = df.copy()

    # Région (codes propres : "11", "76", etc.)
    if "code_region" in df.columns:
        regions = (
            df["code_region"].dropna().astype("Int64").astype(str).sort_values().unique()
        )
        region_choice = st.sidebar.multiselect(
            "Région (code INSEE)", options=list(regions), default=list(regions)
        )
        df_filt = df_filt[
            df_filt["code_region"].astype("Int64").astype(str).isin(region_choice)
        ]

    # Type de bâtiment
    if "type_batiment" in df.columns:
        types = sorted(df["type_batiment"].dropna().unique())
        type_choice = st.sidebar.multiselect(
            "Type de bâtiment", options=types, default=types
        )
        df_filt = df_filt[df_filt["type_batiment"].isin(type_choice)]

    # Période de construction
    if "periode_construction" in df.columns:
        periodes = sorted(df["periode_construction"].dropna().unique())
        periode_choice = st.sidebar.multiselect(
            "Période de construction", options=periodes, default=periodes
        )
        df_filt = df_filt[df_filt["periode_construction"].isin(periode_choice)]

    # Classe DPE (A à G)
    if "etiquette_dpe" in df.columns:
        classes_all = list("ABCDEFG")
        classes_presentes = sorted(df["etiquette_dpe"].dropna().unique())
        options = [c for c in classes_all if c in classes_presentes]
        classe_choice = st.sidebar.multiselect(
            "Classe DPE", options=options, default=options
        )
        df_filt = df_filt[df_filt["etiquette_dpe"].isin(classe_choice)]

    return df_filt


# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================


# ----------------------------- PAGES --------------------------------
def page_intro():
    st.title("Hackathon – DPE & consommations électriques")

    df_base = get_base_df()
    df = get_feat_df()
    nb_logements_base = len(df_base)
    nb_logements = len(df)
    nb_adresses = df_base["address_ban"].nunique() if "address_ban" in df_base.columns else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Logements Enedis x DPE (brut)", f"{nb_logements_base:,}".replace(",", " "))
    col2.metric("Logements utilisables (après nettoyage conso)", f"{nb_logements:,}".replace(",", " "))
    if nb_adresses is not None:
        col3.metric("Adresses uniques", f"{nb_adresses:,}".replace(",", " "))

    st.markdown(
        """
        Cette application répond à deux questions principales :

        1. **Montrer l’écart** entre la consommation prédite par le **DPE**
           et la consommation électrique **réelle** mesurée par Enedis.
        2. **Évaluer le gain financier moyen** sur la facture d’électricité
           lorsqu’on passe d’une **classe énergétique DPE** à une autre
           (par exemple de G à F, de E à C, etc.).

        Les filtres à gauche permettent de se restreindre à un type de logements
        (région, type de bâtiment, période, classe DPE). Toutes les statistiques
        sont recalculées sur ce sous-échantillon.
        """
    )

# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================

def page_dpe_vs_reel():
    import plotly.express as px

    st.header("DPE vs consommation réelle")

    # --- Chargement + filtres globaux ---
    df_all = get_feat_df()
    df = filtre_df(df_all)

    if df.empty:
        st.warning("Aucune donnée ne correspond aux filtres choisis.")
        return

    st.caption(f"Sous-échantillon courant : **{len(df):,} logements**".replace(",", " "))

    # -------------------------------
    # 1) MENU DÉROULANT D’ADRESSES
    # -------------------------------
    if "address_ban" in df.columns:
        addresses = ["Toutes les adresses"] + sorted(df["address_ban"].dropna().unique())
        adresse_choice = st.selectbox("Adresse :", options=addresses)
    else:
        adresse_choice = "Toutes les adresses"

    # Filtrage par adresse
    if adresse_choice != "Toutes les adresses":
        df_addr = df[df["address_ban"] == adresse_choice]
    else:
        df_addr = df

    if df_addr.empty:
        st.warning("Aucune donnée pour cette adresse.")
        return

    # -------------------------------
    # 2) STATISTIQUES PRINCIPALES
    # -------------------------------
    stats = features.compute_dpe_vs_real_stats(df_addr)

    col1, col2, col3 = st.columns(3)
    col1.metric("Biais moyen (réel - DPE)",
                f"{stats.biais_moyen_kwh:,.0f} kWh/an/logement".replace(",", " "))
    col2.metric("Écart-type",
                f"{stats.std_biais_kwh:,.0f} kWh/an/logement".replace(",", " "))
    col3.metric("Ratio réel / DPE",
                f"{stats.ratio_moyen:.2f}")

    st.markdown("### Comment interpréter ?")
    st.markdown(features.summarize_subset(df_addr))

    with st.expander("Résumé statistique global"):
        st.dataframe(stats.global_stats)

    # -------------------------------
    # 3) HISTOGRAMME DES ÉCARTS
    # -------------------------------
    st.markdown("### Distribution des écarts (réel - DPE)")
    fig_hist = px.histogram(
        df_addr,
        x="ecart_kwh_logement",
        nbins=40,
        labels={"ecart_kwh_logement": "Écart (kWh/an/logement)"},
        title="Écarts entre consommation réelle et estimation DPE",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------------
    # 4) SCATTER REAL vs DPE
    # -------------------------------
    st.markdown("### Nuage de points : conso réelle vs conso DPE")

    marker_size = 6 if adresse_choice == "Toutes les adresses" else 12

    fig_scatter = px.scatter(
        df_addr,
        x="conso_dpe_kwh",
        y="conso_reelle_kwh",
        color="etiquette_dpe" if "etiquette_dpe" in df_addr.columns else None,
        opacity=0.7,
        size_max=marker_size,
        labels={
            "conso_dpe_kwh": "Conso DPE (kWh/an)",
            "conso_reelle_kwh": "Conso réelle (kWh/an)",
        },
        title=(
            "Conso réelle vs DPE (toutes adresses)"
            if adresse_choice == "Toutes les adresses"
            else f"Conso réelle vs DPE — {adresse_choice}"
        ),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -------------------------------
    # 5) TABLEAUX CATÉGORIELS
    # -------------------------------
    st.subheader("Par classe DPE")
    st.dataframe(stats.by_dpe, use_container_width=True)

    st.subheader("Par type de bâtiment")
    st.dataframe(stats.by_type, use_container_width=True)

    st.subheader("Par période de construction")
    st.dataframe(stats.by_periode, use_container_width=True)




# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================


def page_impact_dpe():
    st.header("Impact d'un changement de classe DPE")

    df_all = get_feat_df()
    df = filtre_df(df_all)

    if df.empty:
        st.warning("Aucune donnée ne correspond aux filtres choisis.")
        return

    st.caption(f"Sous-échantillon courant : **{len(df):,} logements**".replace(",", " "))

    # Tableau de conso moyenne par classe
    st.subheader("Consommation réelle moyenne par classe DPE")
    moy_par_classe = (
        df.groupby("etiquette_dpe")["conso_reelle_kwh"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n", "mean": "conso_moy_kwh"})
        .reset_index()
        .sort_values("etiquette_dpe")
    )
    st.dataframe(moy_par_classe, use_container_width=True)

    st.markdown("---")
    st.subheader("Simuler un scénario de rénovation")

    classes_dispo = sorted(df["etiquette_dpe"].dropna().unique())
    ordre = list("ABCDEFG")
    classes_select = [c for c in ordre if c in classes_dispo]

    col1, col2, col3 = st.columns(3)
    with col1:
        classe_depart = st.selectbox(
            "Classe actuelle du logement", options=classes_select
        )
    with col2:
        # Par défaut, classe juste au-dessus (meilleure)
        idx_dep = classes_select.index(classe_depart)
        idx_arr = max(0, idx_dep - 1)
        classe_arrivee = st.selectbox(
            "Classe visée après travaux", options=classes_select, index=idx_arr
        )
    with col3:
        prix_kwh = st.number_input(
            "Prix de l'électricité (€/kWh)",
            min_value=0.05,
            max_value=1.0,
            value=0.20,
            step=0.01,
        )

    if ordre.index(classe_arrivee) >= ordre.index(classe_depart):
        st.warning(
            "La classe visée doit être **meilleure** (plus proche de A) que la classe actuelle."
        )
        return

    res = features.gain_entre_classes(df, classe_depart, classe_arrivee)
    if not res:
        st.warning(
            "Pas assez de données pour cette combinaison de classes avec les filtres actuels."
        )
        return

    gain_kwh = res["gain_kwh"]
    gain_euros = gain_kwh * prix_kwh
    conso_avant = res["conso_depart"]
    conso_apres = res["conso_arrivee"]

    st.markdown(
        f"""
        En moyenne, sur les logements similaires (selon les filtres) :

        - un logement en classe **{classe_depart}** consomme **{conso_avant:,.0f} kWh/an** ;
        - un logement en classe **{classe_arrivee}** consomme **{conso_apres:,.0f} kWh/an** ;
        - le **gain moyen** en passant de {classe_depart} à {classe_arrivee} est de  
          👉 **{gain_kwh:,.0f} kWh/an/logement**  
          👉 soit **{gain_euros:,.0f} €/an/logement** au prix de **{prix_kwh:.2f} €/kWh**.
        """.replace(
            ",", " "
        )
    )

    conso_perso = st.number_input(
        "Si tu connais ta consommation actuelle (kWh/an), saisis-la :",  # optionnel
        min_value=0.0,
        step=100.0,
        format="%.0f",
    )
    if conso_perso > 0:
        conso_perso_apres = max(conso_perso - gain_kwh, 0)
        st.success(
            f"Pour ton logement : conso après travaux ≈ **{conso_perso_apres:,.0f} kWh/an** "
            f"(économie ≈ {gain_kwh:,.0f} kWh/an).".replace(",", " ")
        )


def page_prediction_ml():
    st.header("Prédiction de la consommation réelle")

    df = get_feat_df()
    model_obj = get_model()

    st.markdown(
        """
        Renseigne les informations de ton logement pour estimer
        sa **consommation réelle** à partir d'un modèle entraîné
        sur les données Enedis x DPE.
        """
    )

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
    periodes = (
        sorted(df["periode_construction"].dropna().unique())
        if "periode_construction" in df.columns
        else ["Avant 1948", "1949-1974", "1975-1989", "1990-2004", "Depuis 2005"]
    )
    regions = (
        df["code_region"].dropna().astype("Int64").astype(str).sort_values().unique()
        if "code_region" in df.columns
        else ["11"]
    )

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
            periode_construction = st.selectbox(
                "Période de construction", options=periodes
            )
            code_region = st.selectbox("Code région (INSEE)", options=regions)

        submitted = st.form_submit_button("Prédire la consommation")

    if submitted:
        user_data = {
            "conso_dpe_kwh": conso_dpe_kwh,
            "surface_habitable": surface_habitable,
            "annee_construction": annee_construction,
            "etiquette_dpe": etiquette_dpe,
            "type_batiment": type_batiment,
            "periode_construction": periode_construction,
            "code_region": code_region,
        }

        y_pred = models.predict_conso(model_obj, user_data)
        st.success(
            f"Consommation réelle estimée : **{y_pred:,.0f} kWh/an**".replace(",", " ")
        )

        if conso_dpe_kwh > 0:
            diff = y_pred - conso_dpe_kwh
            pct = diff / conso_dpe_kwh * 100
            st.write(
                f"Écart par rapport à la valeur DPE : **{diff:,+.0f} kWh/an** "
                f"({pct:+.1f} %).".replace(",", " ")
            )

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
            Elle peut donc différer de la valeur indiquée sur ton DPE, qui est calculée de
            manière conventionnelle.
            """
        )


def page_dataviz():
    st.header("Datavisualisation")

    import plotly.express as px

    df_all = get_feat_df()
    df = filtre_df(df_all)

    if df.empty:
        st.warning("Aucune donnée ne correspond aux filtres choisis.")
        return

    st.subheader("Nuage de points : conso réelle vs conso DPE")
    scatter_df = features.prepare_scatter_sample(df, n_max=5000)
    fig_scatter = px.scatter(
        scatter_df,
        x="conso_dpe_kwh",
        y="conso_reelle_kwh",
        color="etiquette_dpe" if "etiquette_dpe" in scatter_df.columns else None,
        opacity=0.6,
        labels={
            "conso_dpe_kwh": "Conso DPE (kWh/an)",
            "conso_reelle_kwh": "Conso réelle (kWh/an)",
        },
        title="Conso réelle vs conso DPE",
    )
    fig_scatter.update_layout(legend_title="Classe DPE")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Boxplot : ratio réel / DPE par classe")
    box_df = features.prepare_boxplot_ratio(df)
    if box_df.empty:
        st.info("Pas assez de données pour afficher le boxplot.")
    else:
        fig_box = px.box(
            box_df,
            x="etiquette_dpe",
            y="ratio_reel_sur_dpe",
            labels={"etiquette_dpe": "Classe DPE", "ratio_reel_sur_dpe": "Réel / DPE"},
            title="Distribution du ratio conso réelle / conso DPE",
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ----------------------------- NAVIGATION ---------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "",
        [
            "Introduction",
            "DPE vs conso réelle",
            "Impact d'un changement de classe DPE",
            "Prédiction ML",
            "Datavisualisation",
        ],
    )

    if page == "Introduction":
        page_intro()
    elif page == "DPE vs conso réelle":
        page_dpe_vs_reel()
    elif page == "Impact d'un changement de classe DPE":
        page_impact_dpe()
    elif page == "Prédiction ML":
        page_prediction_ml()
    elif page == "Datavisualisation":
        page_dataviz()


if __name__ == "__main__":
    main()
