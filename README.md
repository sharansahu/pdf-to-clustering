# pdf-to-clustering

This project provides a script for clustering sentences extracted from a document. It supports multiple clustering methods and optimizes parameters using Bayesian optimization. The clustered sentences and their labels are saved to an Excel file, and the clusters are visualized using UMAP and Plotly.

## Features

- Extracts text from a PDF document using a regex pattern.
- Cleans the extracted text to remove unwanted patterns.
- Clusters sentences using various clustering algorithms:
  - DBSCAN
  - HDBSCAN
  - KMeans
  - Gaussian Mixture (EM)
  - Hierarchical (Agglomerative)
- Optimizes clustering parameters using Bayesian optimization with Hyperopt.
- Visualizes the clusters using UMAP and Plotly.
- Saves the clustered sentences and their labels to an Excel file.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/text-clustering.git
    cd text-clustering
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the spaCy language model:
    ```sh
    python -m spacy download en_core_web_sm
    ```

## Usage

To use the script, run the following command:

```sh
python text_clustering.py /path/to/your/document.pdf --clustering_method <clustering_method> [options]
```

### Arguments

- `data_path` (str): Path to the PDF document.
- `--pattern` (str, optional): Regex pattern for extracting text. Default is `r"(?<!UNCLASSIFIED//FOR OFFICIAL USE ONLY\n)(\d+\.\d+\.\d+\.\d+)\s(.*?)(?=\n\d+\.\d+\.\d+\.\d+\s|$)(?!\nUNCLASSIFIED//FOR OFFICIAL USE ONLY)"`.
- `--model_name` (str, optional): Sentence embedding model name. Default is `paraphrase-MiniLM-L6-v2`.
- `--clustering_method` (str): Clustering method to use. Choices are `dbscan`, `hdbscan`, `kmeans`, `gaussian_mixture`, `hierarchical`.
- `--optimize` (action): Optimize clustering parameters.
- `--eps` (float, optional): DBSCAN eps parameter.
- `--min_samples` (int, optional): DBSCAN min_samples parameter.
- `--min_cluster_size` (int, optional): HDBSCAN min_cluster_size parameter.
- `--n_clusters` (int, optional): KMeans and Hierarchical n_clusters parameter.
- `--n_components` (int, optional): Gaussian Mixture n_components parameter.

### Examples

#### Basic Usage
```sh
python text_clustering.py /path/to/your/document.pdf --clustering_method kmeans --n_clusters 5
```

#### Optimizing Clustering Parameters
```sh
python text_clustering.py /path/to/your/document.pdf --clustering_method hdbscan --optimize
```

## Output

The script will generate two outputs:
1. An Excel file named `clustered_sentences.xlsx` containing the sentences, their assigned clusters, and the extracted cluster labels.
2. A visualization of the clusters displayed in a Plotly scatter plot.

## License

This project is licensed under the MIT License.
```
