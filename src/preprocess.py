def load_and_preprocess():
    import pandas as pd
    import numpy as np
    import ast
    from sklearn.preprocessing import MultiLabelBinarizer


    df_credits = pd.read_csv(r'data\tmdb_5000_credits.csv')
    df_movies = pd.read_csv(r'data\tmdb_5000_movies.csv')
    df = df_movies.merge(df_credits,on='title')

    #Removal of unwanted features and cleaning
    df.drop(['title','id','original_language','overview','production_companies','spoken_languages','status','original_title','vote_count','tagline','homepage','keywords','movie_id','crew'],axis=1,inplace=True)
    df.drop(df[df['revenue']==0].index,inplace=True)
    df.reset_index(drop=True,inplace=True)

    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_month'] = df['release_date'].dt.month
    df['release_year'] = df['release_date'].dt.year
    df.drop(['release_date'],axis=1,inplace=True)

    #feature encoding
    df['genres'] = df['genres'].apply(lambda x: [dict['name'] for dict in ast.literal_eval(x)])
    top_genres = pd.Series([genre for sublist in df['genres'] for genre in sublist]).value_counts().head(20).index
    df['genres'] = df['genres'].apply(lambda x:[genre for genre in x if genre in top_genres])
    genre_mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(genre_mlb.fit_transform(df['genres']), columns=genre_mlb.classes_, index=df.index)

    df['production_countries'] = df['production_countries'].apply(lambda x: [dict['name'] for dict in ast.literal_eval(x)])
    top_countries = pd.Series([country for sublist in df['production_countries'] for country in sublist]).value_counts().head(20).index
    df['production_countries'] = df['production_countries'].apply(lambda x:[country for country in x if country in top_countries])
    country_mlb = MultiLabelBinarizer()
    country_encoded = pd.DataFrame(country_mlb.fit_transform(df['production_countries']), columns=country_mlb.classes_, index=df.index)

    df['cast'] = df['cast'].apply(lambda x: [dict['name'] for dict in ast.literal_eval(x)])
    top_cast = pd.Series([actor for sublist in df['cast'] for actor in sublist]).value_counts().head(20).index
    df['cast'] = df['cast'].apply(lambda x:[actor for actor in x if actor in top_cast])
    cast_mlb = MultiLabelBinarizer()
    cast_encoded = pd.DataFrame(cast_mlb.fit_transform(df['cast']), columns=cast_mlb.classes_, index=df.index)

    df.drop(['genres','production_countries','cast'],axis=1,inplace=True)
    df_encoded = pd.concat([df, genre_encoded,country_encoded,cast_encoded], axis=1)
    df_encoded['Success'] = df_encoded.apply(lambda x:'Success' if x['revenue']>1.5*x['budget'] else 'Fail',axis=1)
    
    return df_encoded


