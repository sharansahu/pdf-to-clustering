import re
import fitz
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
from umap import UMAP
import plotly.express as px
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model for the spaCy dependency parser\n"
          "(only required the first time this is run)\n")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text(data_path, pattern):
    doc = fitz.open(data_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return re.findall(pattern, text, re.DOTALL)

def clean_sentences(matches, subpatterns=None, break_condition=None):
    requirements = []
    for match in matches:
        sentence = match[1].strip()
        if subpatterns:
            for subpattern in subpatterns:
                sentence = re.sub(subpattern, "", sentence)
        if break_condition and break_condition in sentence:
            break
        requirements.append(sentence)
    return requirements

def cluster_sentences(sentence_embeddings, clustering_method, umap_n_neighbors, umap_n_components, **kwargs):
    if len(sentence_embeddings) == 0:
        raise ValueError("The sentence embeddings are empty. Check the text extraction and cleaning process.")
    
    umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, random_state=42, n_jobs=-1)
    umap_embeddings = umap_model.fit_transform(sentence_embeddings)

    clustering_params = {k: int(v) if isinstance(v, float) else v for k, v in kwargs.items() if k not in ['umap_n_neighbors', 'umap_n_components']}
    
    if clustering_method == 'dbscan':
        cluster = DBSCAN(**clustering_params)
    elif clustering_method == 'hdbscan':
        cluster = HDBSCAN(**clustering_params)
    elif clustering_method == 'kmeans':
        cluster = KMeans(**clustering_params)
    elif clustering_method == 'gaussian_mixture':
        cluster = GaussianMixture(**clustering_params)
    elif clustering_method == 'hierarchical':
        cluster = AgglomerativeClustering(**clustering_params)
    else:
        raise ValueError("Unsupported clustering method")
    
    labels = cluster.fit_predict(umap_embeddings)
    
    return labels, umap_embeddings

def optimize_parameters(sentence_embeddings, clustering_method):
    if len(sentence_embeddings) == 0:
        raise ValueError("The sentence embeddings are empty. Check the text extraction and cleaning process.")
    
    def objective(params):
        umap_model = UMAP(n_neighbors=int(params['n_neighbors']), n_components=int(params['n_components']), random_state=42, n_jobs=-1)
        umap_embeddings = umap_model.fit_transform(sentence_embeddings)
        
        clustering_params = {k: int(v) if isinstance(v, float) else v for k, v in params.items() if k not in ['n_neighbors', 'n_components']}
        
        if clustering_method == 'dbscan':
            cluster = DBSCAN(**clustering_params)
        elif clustering_method == 'hdbscan':
            cluster = HDBSCAN(**clustering_params)
        elif clustering_method == 'kmeans':
            cluster = KMeans(**clustering_params)
        elif clustering_method == 'gaussian_mixture':
            cluster = GaussianMixture(**clustering_params)
        elif clustering_method == 'hierarchical':
            cluster = AgglomerativeClustering(**clustering_params)
        else:
            raise ValueError("Unsupported clustering method")
        
        labels = cluster.fit_predict(umap_embeddings)
        
        if len(set(labels)) <= 1:
            return {'loss': 1, 'status': STATUS_OK}
        
        if clustering_method == 'hdbscan':
            confidence_threshold = 0.05
            cost = np.mean(cluster.probabilities_ < confidence_threshold)
        else:
            cost = -silhouette_score(umap_embeddings, labels)
        
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if num_clusters < 75 or num_clusters > 300:
            cost += 0.25
        
        return {'loss': cost, 'status': STATUS_OK}
    
    search_space = {
        'n_neighbors': hp.quniform('n_neighbors', 2, 50, 1),
        'n_components': hp.quniform('n_components', 2, 50, 1)
    }
    
    if clustering_method == 'dbscan':
        search_space.update({
            'eps': hp.uniform('eps', 0.1, 10),
            'min_samples': hp.quniform('min_samples', 2, 20, 1)
        })
    elif clustering_method == 'hdbscan':
        search_space.update({
            'min_cluster_size': hp.quniform('min_cluster_size', 2, 20, 1),
            'min_samples': hp.quniform('min_samples', 1, 20, 1)
        })
    elif clustering_method == 'kmeans':
        search_space.update({
            'n_clusters': hp.quniform('n_clusters', 2, 100, 1)
        })
    elif clustering_method == 'gaussian_mixture':
        search_space.update({
            'n_components': hp.quniform('n_components_gm', 2, 100, 1)
        })
    elif clustering_method == 'hierarchical':
        search_space.update({
            'n_clusters': hp.quniform('n_clusters', 2, 100, 1)
        })
    
    trials = Trials()
    best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=100, trials=trials, rstate=np.random.default_rng(42))
    
    return {key: int(value) for key, value in best_params.items()}

def print_clusters(sentences, cluster_labels):
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        if label == -1:
            continue
        clusters[label].append(sentences[idx])
    
    sorted_clusters = dict(sorted(clusters.items()))
    for cluster_id, sentences in sorted_clusters.items():
        print(f"Cluster {cluster_id}:")
        for sentence in sentences:
            print(sentence)
        print()

def visualize_clusters(embeddings, labels, sentences, title):
    reducer = UMAP(n_components=2, random_state=42, n_jobs=-1)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels,
        'text': sentences
    })
    
    fig = px.scatter(
        df, x='x', y='y', color='label', hover_data={'text': True},
        labels={'x': 'UMAP Component 1', 'y': 'UMAP Component 2'},
        title=title
    )
    
    fig.show()

def extract_labels(sentences, labels):
    cluster_sentences = defaultdict(list)
    for sentence, label in zip(sentences, labels):
        if label != -1:
            cluster_sentences[label].append(sentence)

    cluster_labels = {}
    for label, sentences in cluster_sentences.items():
        verb_counts = Counter()
        dobj_counts = Counter()
        noun_counts = Counter()

        for sentence in sentences:
            doc = nlp(sentence)
            for token in doc:
                if token.dep_ == 'ROOT':
                    verb_counts[token.lemma_] += 1
                elif token.dep_ == 'dobj':
                    dobj_counts[token.lemma_] += 1
                elif token.pos_ == 'NOUN':
                    noun_counts[token.lemma_] += 1

        top_verb = verb_counts.most_common(1)[0][0] if verb_counts else 'unknown'
        top_dobj = dobj_counts.most_common(1)[0][0] if dobj_counts else 'unknown'
        top_nouns = [noun for noun, count in noun_counts.most_common(2)]
        
        cluster_labels[label] = f"{top_verb}-{top_dobj}-{'-'.join(top_nouns)}"
        if '@' in cluster_labels[label]:
            cluster_labels[label] = cluster_labels[label].replace("@", "")
    
    return cluster_labels

def main(data_path, pattern, model_name, clustering_method, umap_n_neighbors, umap_n_components, optimize=False, subpatterns=None, break_condition=None, **kwargs):
    matches = extract_text(data_path, pattern)
    sentences = clean_sentences(matches, subpatterns, break_condition)

    if not sentences:
        raise ValueError("No sentences extracted. Check the text extraction and cleaning process.")

    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(sentences)
    
    best_params = {}
    title = 'Clusters Visualization'
    
    if optimize:
        best_params = optimize_parameters(sentence_embeddings, clustering_method)
        kwargs.update(best_params)
        umap_n_neighbors = best_params.get('n_neighbors', umap_n_neighbors)
        umap_n_components = best_params.get('n_components', umap_n_components)
        title += f' (Optimized Parameters: {best_params})'
    
    labels, embeddings = cluster_sentences(sentence_embeddings, clustering_method, umap_n_neighbors, umap_n_components, **kwargs)
    cluster_labels = extract_labels(sentences, labels)
    
    df = pd.DataFrame({
        'Sentence': sentences,
        'Cluster': labels,
        'Cluster Label': [cluster_labels[label] if label != -1 else 'Noise' for label in labels]
    })
    df.to_excel("clustered_sentences.xlsx", index=False)
    
    visualize_clusters(embeddings, labels, sentences, title)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Clustering Script")
    parser.add_argument('data_path', type=str, help='Path to the document')
    parser.add_argument('--pattern', type=str, default=r"(?<!UNCLASSIFIED//FOR OFFICIAL USE ONLY\n)(\d+\.\d+\.\d+\.\d+)\s(.*?)(?=\n\d+\.\d+\.\d+\.\d+\s|$)(?!\nUNCLASSIFIED//FOR OFFICIAL USE ONLY)", help='Regex pattern for extracting text')
    parser.add_argument('--subpatterns', nargs='*', default=[r"(UNCLASSIFIED//FOR OFFICIAL USE ONLY\s+\d+\s+)", r"UNCLASSIFIED//FOR OFFICIAL USE ONLY"], help='List of regex subpatterns to remove/clean from extracted text')
    parser.add_argument('--break_condition', type=str, default="Acronyms", help='Condition to break sentence extraction')
    parser.add_argument('--model_name', type=str, default='paraphrase-MiniLM-L6-v2', help='Sentence embedding model name')
    parser.add_argument('--clustering_method', type=str, choices=['dbscan', 'hdbscan', 'kmeans', 'gaussian_mixture', 'hierarchical'], required=True, help='Clustering method')
    parser.add_argument('--optimize', action='store_true', help='Optimize clustering parameters')
    
    # Clustering method specific parameters
    parser.add_argument('--eps', type=float, help='DBSCAN eps parameter')
    parser.add_argument('--min_samples', type=int, help='DBSCAN min_samples parameter')
    parser.add_argument('--min_cluster_size', type=int, help='HDBSCAN min_cluster_size parameter')
    parser.add_argument('--n_clusters', type=int, help='KMeans and Hierarchical n_clusters parameter')
    parser.add_argument('--n_components', type=int, help='Gaussian Mixture n_components parameter')
    parser.add_argument('--umap_n_neighbors', type=int, default=15, help='Number of neighboring points used in local approximations of manifold structure for UMAP.')
    parser.add_argument('--umap_n_components', type=int, default=30, help='Number of components you want to reduce to using UMAP.')
    
    args = parser.parse_args()
    
    clustering_kwargs = {key: value for key, value in vars(args).items() if key in ['eps', 'min_samples', 'min_cluster_size', 'n_clusters', 'n_components'] and value is not None}
    
    main(args.data_path, args.pattern, args.model_name, args.clustering_method, args.umap_n_neighbors, args.umap_n_components, args.optimize, args.subpatterns, args.break_condition, **clustering_kwargs)
