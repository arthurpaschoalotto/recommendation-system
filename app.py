import streamlit as st
import pandas as pd
import joblib
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ========== Configuração da Página ==========
st.set_page_config(page_title="🎬 Recomendador de Filmes", layout="wide")

# ========== Carregamento ==========
@st.cache_data
def load_data():
    df = pd.read_parquet('dataset.parquet')
    model = joblib.load('model_xgboost.pkl')
    feature_cols = joblib.load('features.pkl')
    return df, model, feature_cols

df, model, feature_cols = load_data()

# ========== Funções auxiliares ==========
def extract_all_genres(genres_str):
    if pd.isna(genres_str):
        return []
    return [g.strip() for g in genres_str.split(',') if g.strip()]

def safe_parse(x):
    try:
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            return ast.literal_eval(x)
    except:
        pass
    return x

def ensure_list(x):
    if pd.isna(x) or x == "":
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return [s.strip() for s in parsed]
        except:
            pass
        return [s.strip() for s in x.split(',') if s.strip()]
    return []

# ========== Função de Recomendação ==========
def recommend_movies(df, liked_titles, watched_titles, model, feature_cols, top_n=10):
    df = df.copy()

    if 'type' in df.columns:
        df = df[df['type'] == 'movie']

    liked_df = df[df['title'].isin(liked_titles)].copy()
    if liked_df.empty:
        return pd.DataFrame(), []

    all_genres = liked_df['genres'].dropna().apply(extract_all_genres).explode()
    top_genres = all_genres.value_counts().head(3).index.tolist()

    if not top_genres:
        return pd.DataFrame(), []

    main_genre = top_genres[0]
    secondary_genre = top_genres[1] if len(top_genres) > 1 else None
    tertiary_genre = top_genres[2] if len(top_genres) > 2 else None

    def valid_genre_combination(genres_str):
        if pd.isna(genres_str):
            return False
        genres = extract_all_genres(genres_str)
        if main_genre not in genres:
            return False
        allowed = {main_genre}
        if secondary_genre:
            allowed.add(secondary_genre)
        if tertiary_genre:
            allowed.add(tertiary_genre)
        return set(genres).issubset(allowed)

    df = df[df['genres'].apply(valid_genre_combination)]

    liked_subgenres = liked_df['genres'].dropna().apply(extract_all_genres).explode()
    subgenres_set = set(liked_subgenres.dropna())

    def compute_subgenre_affinity(genres_str):
        genres = extract_all_genres(genres_str)
        return len(set(genres).intersection(subgenres_set))

    df['subgenre_affinity'] = df['genres'].apply(compute_subgenre_affinity)
    df['subgenre_affinity'] = MinMaxScaler().fit_transform(df[['subgenre_affinity']])

    if 'releaseYear' in df.columns and not liked_df['releaseYear'].isnull().all():
        user_year_pref = liked_df['releaseYear'].mean()
        df['year_diff'] = (df['releaseYear'] - user_year_pref).abs()
        df['year_affinity'] = 1 - MinMaxScaler().fit_transform(df[['year_diff']])
    else:
        df['year_affinity'] = 0.5

    user_vector = liked_df[feature_cols].mean().values.reshape(1, -1)
    df['user_similarity'] = cosine_similarity(df[feature_cols], user_vector).flatten()
    df['predicted_rating'] = model.predict(df[feature_cols])

    if 'popularity' not in df.columns:
        popularity = df.groupby('title').size().reset_index(name='popularity')
        df = df.merge(popularity, on='title', how='left')

    df['popularity'] = df['popularity'].fillna(0)
    df['popularity_penalty'] = MinMaxScaler().fit_transform(df[['popularity']])

    df['final_score'] = (
        0.35 * df['predicted_rating'] +
        0.25 * df['user_similarity'] +
        0.15 * df['subgenre_affinity'] +
        0.15 * df['year_affinity'] -
        1.0 * df['popularity_penalty']
    )

    df = df[
        (~df['title'].isin(liked_titles)) &
        (~df['title'].isin(watched_titles))
    ].copy()

    top_popular = df.sort_values(by='popularity', ascending=False).head(5)
    exploratory = df[df['popularity'] <= df['popularity'].quantile(0.3)]
    top_exploratory = exploratory.sort_values(by='final_score', ascending=False).head(5)

    final_df = pd.concat([top_popular, top_exploratory]).drop_duplicates(subset=['title', 'streamming'])

    final_df['streamming'] = final_df['streamming'].apply(safe_parse)
    final_df['streamming'] = final_df['streamming'].apply(ensure_list)

    final_df = final_df.groupby('title').agg({
        'streamming': lambda lsts: ', '.join(sorted(set([item for sublist in lsts for item in sublist]))),
        'predicted_rating': 'mean',
        'final_score': 'mean',
        'popularity': 'mean'
    }).reset_index()

    final_df = final_df.sort_values(by='final_score', ascending=False)

    if len(final_df) < top_n:
        remaining = df[~df['title'].isin(final_df['title'])]                 .sort_values(by='final_score', ascending=False)                 .head(top_n - len(final_df))
        remaining['streamming'] = remaining['streamming'].apply(safe_parse).apply(ensure_list)
        remaining = remaining.groupby('title').agg({
            'streamming': lambda lsts: ', '.join(sorted(set([item for sublist in lsts for item in sublist]))),
            'predicted_rating': 'mean',
            'final_score': 'mean',
            'popularity': 'mean'
        }).reset_index()
        final_df = pd.concat([final_df, remaining])

    return final_df.reset_index(drop=True), top_genres

# ========== Interface ==========
st.title("🎬 Sistema de Recomendação de Filmes")
st.markdown("Selecione **de 5 a 10 filmes** que você gosta. O sistema irá sugerir títulos com base nos gêneros, similaridade e contexto:")

if 'watched' not in st.session_state:
    st.session_state.watched = []
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()
if 'recommendation_genres' not in st.session_state:
    st.session_state.recommendation_genres = []

liked_titles = st.multiselect(
    "Filmes que você gosta:",
    sorted(df['title'].unique()),
    max_selections=10,
    help="Selecione entre 5 e 10 filmes"
)

if st.button("🔍 Gerar Recomendações", use_container_width=True) and len(liked_titles) >= 5:
    with st.spinner("Calculando..."):
        st.session_state.results, st.session_state.recommendation_genres = recommend_movies(
            df, liked_titles, st.session_state.watched, model, feature_cols
        )

if not st.session_state.results.empty:
    st.success("🎯 Aqui estão suas recomendações personalizadas:")

    if st.session_state.recommendation_genres:
        genero_str = ', '.join(st.session_state.recommendation_genres)
        st.markdown(f"📌 **Para você que gosta de filmes de:** *{genero_str}*")

    updated = False

    cols = st.columns(2)
    for idx, (i, row) in enumerate(st.session_state.results.iterrows()):
        col = cols[idx % 2]
        with col:
            with st.container():
                st.markdown(f"### 🎬 {row['title']}")
                st.markdown(f"**📺 Plataformas:** {row['streamming']}")
                btn_key = f"watched_btn_{i}_{row['title']}"
                if st.button("✅ Já assisti", key=btn_key):
                    if row['title'] not in st.session_state.watched:
                        st.session_state.watched.append(row['title'])
                        updated = True
                        st.rerun()

    if updated:
        st.session_state.results, st.session_state.recommendation_genres = recommend_movies(
            df, liked_titles, st.session_state.watched, model, feature_cols
        )
        st.rerun()

if st.session_state.watched:
    st.subheader("✅ Já assistidos")
    for title in st.session_state.watched:
        st.markdown(f"- {title}")