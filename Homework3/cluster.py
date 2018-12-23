import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
data_path = 'f:/data/Tweets.txt'

def pre():
    corpus, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line.strip())
            corpus.append(j['text'])
            labels.append(j['cluster'])
    return corpus, labels



def _kmeans(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    kmeans = KMeans(n_clusters=100, max_iter=50, n_init=10, init='k-means++')
    result = kmeans.fit_predict(X.toarray())
    print('K-means:', normalized_mutual_info_score(result, labels))



def _AffinityPropagation(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    affinity_propagation = AffinityPropagation(damping=.5, max_iter=200, convergence_iter=25, copy=False)
    result_affinity_propagation = affinity_propagation.fit_predict(X.toarray())
    print('AffinityPropagation:', normalized_mutual_info_score(result_affinity_propagation, labels))



def _mean_shift(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    mean_shift = MeanShift(bandwidth=0.65, bin_seeding=True)
    result_mean_shift = mean_shift.fit_predict(X.toarray())
    print('MeanShift:', normalized_mutual_info_score(result_mean_shift, labels))



def _SpectralClustering(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    spectral_clustering = SpectralClustering(n_clusters=100, n_init=10)
    result_spectral_clustering = spectral_clustering.fit_predict(X.toarray())
    print('SpectralClustering:', normalized_mutual_info_score(result_spectral_clustering, labels))



def _AgglomerativeClustering(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    agglomerative_clustering = AgglomerativeClustering(n_clusters=100)
    result_agglomerative_clustering = agglomerative_clustering.fit_predict(X.toarray())
    print('AgglomerativeClustering:', normalized_mutual_info_score(result_agglomerative_clustering, labels))


def _DBSCAN(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    dbscan = DBSCAN(eps=0.65, min_samples=1, leaf_size=35)
    result_dbscan = dbscan.fit_predict(X.toarray())
    print('DBSCAN:', normalized_mutual_info_score(result_dbscan, labels))


def GaussianMixture(text, labels):
    vectorizer = TfidfVectorizer()
    new= vectorizer.fit_transform(text)
    gaussian_mixture = GaussianMixture(n_components=89)
    result_gaussian_mixture = gaussian_mixture.fit_predict(new.toarray())
    print('GaussianMixture:', normalized_mutual_info_score(result_gaussian_mixture, labels))

if __name__ == '__main__':
    print("start cluster!")
    corpus, labels = pre()
    _kmeans(corpus, labels)
    _AffinityPropagation(corpus, labels)
    _mean_shift(corpus, labels)
    _SpectralClustering(corpus, labels)
    _AgglomerativeClustering(corpus, labels)
    _DBSCAN(corpus, labels)
    _GaussianMixture(corpus, labels) 
    print("end")
