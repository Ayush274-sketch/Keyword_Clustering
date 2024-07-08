import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import chardet
from detect_delimiter import detect
import numpy as np
import plotly.graph_objects as go
import colorsys
from sklearn.cluster import AgglomerativeClustering, KMeans
import torch
from sklearn.decomposition import PCA
from langdetect import detect_langs
from iso639 import languages

def generate_colors(num_colors):
    colors = []
    hue_values = np.linspace(0, 1, num_colors, endpoint=False)
    np.random.shuffle(hue_values)
    
    for hue in hue_values:
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

def calculate_cluster_coherence(embeddings, cluster_labels):
    unique_labels = np.unique(cluster_labels)
    coherences = []
    for label in unique_labels:
        cluster_embeddings = embeddings[cluster_labels == label]
        if len(cluster_embeddings) > 1:
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            coherence = 1 / (1 + np.mean(distances))
            coherences.append(coherence)
    return coherences

def detect_encoding(file):
    raw_data = file.read()
    file.seek(0)  # Reset file pointer
    result = chardet.detect(raw_data)
    return result['encoding']

def detect_language(text):
    try:
        detected_langs = detect_langs(text)
        main_language_code = detected_langs[0].lang
        language_name = languages.get(part1=main_language_code).name
        return language_name
    except:
        return 'Unknown Language'

def preprocess_keywords(df):
    # Custom preprocessing for specific keywords
    location_keywords = ["delhi", "gurgaon", "pune"]
    df['Keyword'] = df['Keyword'].str.lower()  # Convert to lowercase for consistency

    # Ensure the specific keywords are treated distinctly
    df['Keyword'] = df['Keyword'].apply(lambda x: x if x not in location_keywords else f"loc_{x}")
    
    return df

st.title("Semantic Keyword Clustering Tool")
st.write("Upload a CSV or XLSX file containing keywords for clustering.")

clustering_method = st.sidebar.selectbox(
    "Select Clustering Method",
    ["Community Detection", "Agglomerative", "K-means"],
    help="**Community Detection:** Finds natural groups in your data.\n\nWhen to use: If you're unsure about the number of groups you need.\n\n**Agglomerative:** Groups keywords based on their similarity, step by step.\n\nWhen to use: If you want control over the size of the groups by adjusting the threshold.\n\n**K-means:** Creates a fixed number of groups based on keyword similarity.\n\nWhen to use: If you already know how many groups you want."
)

if clustering_method == "Community Detection":
    cluster_accuracy = st.slider("Cluster Accuracy (0-100)", 0, 100, 91) / 100
    min_cluster_size = st.number_input("Minimum Cluster Size", min_value=1, max_value=100, value=3)
elif clustering_method == "Agglomerative":
    distance_threshold = st.sidebar.number_input("Distance Threshold for Agglomerative Clustering", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
elif clustering_method == "K-means":
    n_clusters = st.number_input("Number of Clusters for K-means", min_value=2, max_value=50000, value=50)

transformer = st.selectbox(
    "Select Transformer Model",
    ['distiluse-base-multilingual-cased-v2', 'paraphrase-multilingual-mpnet-base-v2', 'all-MiniLM-L6-v2'],
    help="**distiluse-base-multilingual-cased-v2:** Supports 50+ languages, good for multilingual datasets.\n\n**paraphrase-multilingual-mpnet-base-v2:** Very accurate, supports 100+ languages.\n\n**all-MiniLM-L6-v2:** Fast, but primarily for English."
)

uploaded_file = st.file_uploader("Upload Keyword CSV or XLSX", type=["csv", "xlsx"])

if uploaded_file:
    try:
        encoding = detect_encoding(uploaded_file)
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding=encoding)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.write("File loaded successfully!")
        st.write(f"Detected encoding: '{encoding}'")

        df.rename(columns={"Search term": "Keyword", "keyword": "Keyword", "query": "Keyword", "Top queries": "Keyword", "queries": "Keyword", "Keywords": "Keyword", "keywords": "Keyword", "Search terms report": "Keyword"}, inplace=True)

        if "Keyword" not in df.columns:
            st.error("Error! Please make sure your CSV or XLSX file contains a column named 'Keyword'!")
        else:
            st.write(f"Total rows: {len(df)}")
            st.write(f"Number of unique keywords: {df['Keyword'].nunique()}")
            st.write("Data types in 'Keyword' column:")
            st.write(df['Keyword'].apply(type).value_counts())
            
            df['Keyword'] = df['Keyword'].astype(str)
            
            st.write("Sample of the data (first 5 rows):")
            st.write(df['Keyword'].head())

            # Preprocess keywords to handle specific cases
            df = preprocess_keywords(df)

            sample_keywords = df['Keyword'].sample(min(100, len(df))).tolist()
            detected_languages = [detect_language(keyword) for keyword in sample_keywords]
            main_language = max(set(detected_languages), key=detected_languages.count)

            st.write(f"Detected main language: {main_language}")
            st.write("Other detected languages: " + ', '.join(set(detected_languages) - {main_language, 'unknown'}))

            model = SentenceTransformer(transformer)
            corpus_set = set(df['Keyword'])
            corpus_set_all = corpus_set
            cluster_name_list = []
            corpus_sentences_list = []
            df_all = []
            cluster = True
            iterations = 0

            while cluster:
                corpus_sentences = list(corpus_set)
                check_len = len(corpus_sentences)

                if len(corpus_sentences) == 0:
                    break

                corpus_embeddings = model.encode(corpus_sentences, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
                
                if len(corpus_embeddings.shape) == 1:
                    corpus_embeddings = corpus_embeddings.reshape(1, -1)
                
                if clustering_method == "Community Detection":
                    clusters = util.community_detection(corpus_embeddings, min_community_size=min_cluster_size, threshold=cluster_accuracy)
                    cluster_labels = np.array([i for i, cluster in enumerate(clusters) for _ in cluster])
                    
                    coherences = []
                    for cluster in clusters:
                        if len(cluster) > 1:
                            cluster_embeddings = corpus_embeddings[cluster]
                            centroid = torch.mean(cluster_embeddings, dim=0)
                            distances = torch.norm(cluster_embeddings - centroid, dim=1)
                            coherence = 1 / (1 + torch.mean(distances).item())
                            coherences.append(coherence)
                    
                    if coherences:
                        overall_coherence = np.mean(coherences)
                        st.write(f"Overall Clustering Coherence: {overall_coherence:.4f}")
                        st.write("Cluster Coherences:")
                        for cluster, coherence in zip(clusters, coherences):
                            cluster_keywords = [corpus_sentences[i] for i in cluster]
                            cluster_name = min(cluster_keywords, key=len)
                            st.write(f"{cluster_name}: {coherence:.4f}")
                    
                elif clustering_method == "Agglomerative":
                    max_clusters = len(corpus_sentences) // 4
                    while True:
                        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
                        cluster_labels = clustering_model.fit_predict(corpus_embeddings.cpu().numpy())
                        n_clusters = len(np.unique(cluster_labels))
                        if n_clusters <= max_clusters:
                            break
                        distance_threshold += 0.1
                    st.write(f"Adjusted distance threshold: {distance_threshold:.2f}")
                    st.write(f"Number of clusters formed: {n_clusters}")
                elif clustering_method == "K-means":
                    clustering_model = KMeans(n_clusters=n_clusters)
                    cluster_labels = clustering_model.fit_predict(corpus_embeddings.cpu().numpy())

                if clustering_method == "Community Detection":
                    for keyword, cluster in enumerate(clusters):
                        for sentence_id in cluster:
                            corpus_sentences_list.append(corpus_sentences[sentence_id])
                            cluster_name_list.append(f"Cluster {keyword + 1}, #{len(cluster)} Elements")
                else:
                    for sentence_id, cluster_id in enumerate(cluster_labels):
                        corpus_sentences_list.append(corpus_sentences[sentence_id])
                        cluster_name_list.append(f"Cluster {cluster_id + 1}")

                df_new = pd.DataFrame({'Cluster Name': cluster_name_list, 'Keyword': corpus_sentences_list})

                df_all.append(df_new)
                have = set(df_new["Keyword"])

                corpus_set = corpus_set_all - have
                remaining = len(corpus_set)
                iterations += 1
                if check_len == remaining:
                    break

            if len(df_all) == 0:
                st.error("No clusters were formed. Please check your data or adjust the clustering parameters.")
            else:
                df_new = pd.concat(df_all)
                df = df.merge(df_new.drop_duplicates('Keyword'), how='left', on="Keyword")

                df['Cluster Name'] = df['Cluster Name'].fillna("no_cluster")

                df['Length'] = df['Keyword'].astype(str).map(len)
                df = df.sort_values(by="Length", ascending=True)
                df['Cluster Name'] = df.groupby('Cluster Name')['Keyword'].transform('first')
                df.sort_values(['Cluster Name', "Keyword"], ascending=[True, True], inplace=True)
                df = df.drop('Length', axis=1)

                df = df[['Cluster Name', 'Keyword']]

                df.sort_values(["Cluster Name", "Keyword"], ascending=[True, True], inplace=True)

                uncluster_percent = (remaining / len(df)) * 100
                clustered_percent = 100 - uncluster_percent
                st.write(f"{clustered_percent:.2f}% of rows clustered successfully!")
                st.write(f"Number of iterations: {iterations}")
                st.write(f"Total unclustered keywords: {remaining}")

                st.write(f"Number of clusters: {len(df['Cluster Name'].unique())}")

                if clustering_method != "Community Detection" and len(corpus_embeddings) > 1:
                    cluster_coherences = calculate_cluster_coherence(corpus_embeddings.cpu().numpy(), cluster_labels)
                    overall_coherence = np.mean(cluster_coherences)

                    st.write(f"Overall Clustering Coherence: {overall_coherence:.4f}")
                    st.write("Cluster Coherences:")
                    cluster_names = df.groupby('Cluster Name')['Keyword'].first()
                    for (cluster_id, cluster_name), coherence in zip(cluster_names.items(), cluster_coherences):
                        st.write(f"{cluster_name}: {coherence:.4f}")

                result_df = df.groupby('Cluster Name')['Keyword'].apply(', '.join).reset_index()
                result_df.columns = ['Cluster', 'Keywords']

                st.write(result_df)

                embeddings = model.encode(df['Keyword'].tolist(), batch_size=256, show_progress_bar=True)

                if embeddings.shape[1] > 3:
                    pca = PCA(n_components=3)
                    embeddings_3d = pca.fit_transform(embeddings)
                elif embeddings.shape[1] < 3:
                    st.error("Error: Embeddings have fewer than 3 dimensions. Please choose a different model.")
                    st.stop()
                else:
                    embeddings_3d = embeddings

                embeddings_normalized = (embeddings_3d - embeddings_3d.min(axis=0)) / (embeddings_3d.max(axis=0) - embeddings_3d.min(axis=0))

                colors = ['rgb({},{},{})'.format(
                    int(r*255), 
                    int(g*255), 
                    int(b*255)
                ) for r, g, b in embeddings_normalized]

                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=embeddings_3d[:, 0],
                    y=embeddings_3d[:, 1],
                    z=embeddings_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors,
                        opacity=0.8
                    ),
                    text=df['Keyword'],
                    hoverinfo='text'
                )])

                fig_3d.update_layout(
                    width=1200,
                    height=675,
                    title='Keyword Embeddings in 3D Space',
                    scene=dict(
                        xaxis_title='Dimension 1',
                        yaxis_title='Dimension 2',
                        zaxis_title='Dimension 3'
                    ),
                    margin=dict(l=0, r=0, b=0, t=40)
                )

                st.plotly_chart(fig_3d, use_container_width=True)
                with st.expander("ℹ️ About this visualization"):
                    st.write("The position of each point in 3D space reflects the semantic similarity between keywords. Points that are closer together represent keywords with more similar meanings or contexts. This visualization works for multiple languages.")
                
                csv_data_clustered = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Clustered Keywords",
                    data=csv_data_clustered,
                    file_name="Clustered_Keywords.csv",
                    mime="text/csv"
                )

                if remaining > 0:
                    st.write("Unclustered Keywords:")
                    st.write(list(corpus_set))
                    
                    unclustered_df = pd.DataFrame(list(corpus_set), columns=['Unclustered Keyword'])
                    
                    csv_data_unclustered = unclustered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Unclustered Keywords",
                        data=csv_data_unclustered,
                        file_name="Unclustered_Keywords.csv",
                        mime="text/csv"
                    )

                st.write("Note: This tool supports clustering of keywords in multiple languages. The effectiveness may vary depending on the selected model and the languages present in your data.")

    except pd.errors.EmptyDataError:
        st.error("EmptyDataError: No columns to parse from file. Please upload a valid CSV or XLSX file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Please check your data and try again.")
