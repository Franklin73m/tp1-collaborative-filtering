"""
TP1 - Interface Streamlit
Collaborative Filtering Item-Item
Fichier : C:\tp1_recommender\app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from recommender import ItemItemCF


st.set_page_config(
    page_title="TP1 - Système de Recommandation",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = ItemItemCF(top_n=10)
    model.fit(
        r"C:\tp1_recommender\data\ratings.csv",
        r"C:\tp1_recommender\data\movies.csv"
    )
    return model

with st.spinner("Chargement du modèle..."):
    model = load_model()

stats = model.stats()


st.title("Système de Recommandation - Big Data")
st.caption("Collaborative Filtering Item-Item (Top-N) — Dataset MovieLens")


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👤 Utilisateurs", f"{stats['users']:,}")
c2.metric("🎬 Films",        f"{stats['films']:,}")
c3.metric("⭐ Ratings",      f"{stats['ratings']:,}")
c4.metric("🕳️ Sparsité",     stats["sparsite"])
c5.metric("📊 Note moy.",    stats["note_moy"])

st.divider()


tab1, tab2, tab3 = st.tabs([
    "🔍 Films similaires",
    "👤 Recommandations",
    "📊 Exploration"
])


with tab1:
    st.subheader("🔍 Films similaires (item-item)")
    col1, col2 = st.columns([1, 2])

    with col1:
        film_options = model.movies[
            model.movies["movieId"].isin(model.matrix.columns)
        ][["movieId", "title"]].sort_values("title")

        film_dict      = dict(zip(film_options["title"], film_options["movieId"]))
        selected_title = st.selectbox("Choisir un film", list(film_dict.keys()))
        selected_id    = film_dict[selected_title]
        n_sim = st.slider("Nombre de films similaires", 5, 20, 10, key="n_sim")

    similaires = model.similar_items(selected_id, n=n_sim)

    with col2:
        fig = px.bar(
            similaires, x="similarite", y="title", orientation="h",
            title=f"Films similaires à « {selected_title[:40]} »",
            color="similarite", color_continuous_scale="Blues",
            labels={"similarite": "Similarité cosinus", "title": "Film"}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        similaires.assign(similarite=similaires["similarite"].round(4)),
        use_container_width=True, hide_index=True
    )


with tab2:
    st.subheader("👤 Recommandations personnalisées")
    col3, col4 = st.columns([1, 2])

    with col3:
        user_id = st.selectbox(
            "Choisir un utilisateur",
            sorted(model.matrix.index.tolist()),
            key="user_sel"
        )
        n_rec = st.slider("Nombre de recommandations", 5, 20, 10, key="n_rec")

    recs = model.recommend(user_id, n=n_rec)

    with col4:
        fig2 = px.bar(
            recs, x="score", y="title", orientation="h",
            title=f"Top-{n_rec} recommandations pour l'utilisateur {user_id}",
            color="score", color_continuous_scale="Greens",
            labels={"score": "Score prédit", "title": "Film"}
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"),
                           coloraxis_showscale=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        recs.assign(score=recs["score"].round(3)),
        use_container_width=True, hide_index=True
    )

    with st.expander(f"Films déjà notés par l'utilisateur {user_id}"):
        deja_notes = model.matrix.loc[user_id].dropna().reset_index()
        deja_notes.columns = ["movieId", "rating"]
        deja_notes = deja_notes.merge(
            model.movies[["movieId", "title", "genres"]], on="movieId"
        ).sort_values("rating", ascending=False)
        st.dataframe(deja_notes[["title", "genres", "rating"]],
                     use_container_width=True, hide_index=True)


with tab3:
    st.subheader("📊 Exploration des données")
    col5, col6 = st.columns(2)

    with col5:
        all_ratings = model.matrix.stack().reset_index()
        all_ratings.columns = ["userId", "movieId", "rating"]
        fig3 = px.histogram(
            all_ratings, x="rating", nbins=9,
            title="Distribution des notes",
            color_discrete_sequence=["#636EFA"]
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col6:
        top_films = model.matrix.notna().sum().sort_values(ascending=False).head(15)
        top_films_df = top_films.reset_index()
        top_films_df.columns = ["movieId", "nb_ratings"]
        top_films_df = top_films_df.merge(
            model.movies[["movieId", "title"]], on="movieId"
        )
        fig4 = px.bar(
            top_films_df, x="nb_ratings", y="title", orientation="h",
            title="Top 15 films les plus notés",
            color="nb_ratings", color_continuous_scale="Oranges",
            labels={"nb_ratings": "Nb ratings", "title": "Film"}
        )
        fig4.update_layout(yaxis=dict(autorange="reversed"),
                           coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Heatmap de similarité (30 premiers films)**")
    sub  = model.similarity.iloc[:30, :30]
    fig5 = go.Figure(data=go.Heatmap(
        z=sub.values, colorscale="RdBu", zmid=0
    ))
    fig5.update_layout(height=500,
                       title="Matrice de similarité cosinus (item-item)")
    st.plotly_chart(fig5, use_container_width=True)