import pandas as pd
import re
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
import warnings
warnings.filterwarnings('ignore')
import dblp
import umap
import numpy as np

resultsg1 = dblp.search(["Gradient"],2021)
resultsg2 = dblp.search(["Gradient"],2020)
resultsg3 = dblp.search(["Gradient"],2019)
resultsg4 = dblp.search(["Gradient"],2018)

resultso1 = dblp.search(["Optimization"],2021)
resultso2 = dblp.search(["Optimization"],2020)
resultso3 = dblp.search(["Optimization"],2019)
resultso4 = dblp.search(["Optimization"],2018)


document1 = resultsg1['Title']
document2 = resultsg2['Title']
document3 = resultsg3['Title']
document4 = resultsg4['Title']
document5 = resultso1['Title']
document6 = resultso2['Title']
document7 = resultso3['Title']
document8 = resultso4['Title']

totdoc = [document1, document2, document3, document4, document5, document6,document7,document8]
df = pd.DataFrame(columns = ['Title'])
df= pd.concat(totdoc, ignore_index=True)
fin_list = df.to_list()

# list for tokenized documents in loop
texts = []
#Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
# loop through title list
for i in fin_list:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # add tokens to list
    texts.append(stemmed_tokens)
corpus = []
for words in texts:
    corword = ' '.join(words)
    corword = re.sub("[^a-zA-Z 0-9]+", "", corword)
    corpus.append(corword)

print(corpus[:1])


# Creating the TF-IDF model
cv2 = TfidfVectorizer()
X2 = cv2.fit_transform(corpus).toarray()
print("\n TF-IDF Matrix-->\n",X2)

print(np.shape(X2))


##### PCA

ipca = IncrementalPCA(n_components=2)
ipca.fit(X2)
PCA_result = ipca.transform(X2)
print("ipca ouput \n", PCA_result)
plt.scatter(PCA_result[:,0], PCA_result[:,1],cmap='rainbow', alpha = 0.6 , s=10)
plt.title("PCA Output", weight = 'bold', fontsize  = '15')
plt.xlabel("X(Principal component 1)")
plt.ylabel("Y(Principal component 2)")
plt.savefig("PCA_Scatter_plot")
plt.show()
print(np.shape(PCA_result))


###### tsne

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X2)
print("tsne_results \n",tsne_results)
plt.scatter(tsne_results[:,0], tsne_results[:,1],cmap='rainbow', alpha = 0.6 , s=10)
plt.title("TSNE Output", weight = 'bold', fontsize  = '15')
plt.xlabel("X(Dimension 1)")
plt.ylabel("Y(Dimension 2)")
plt.savefig("TSNE_Scatter_plot")
plt.show()
print(np.shape(tsne_results))


###UMAP

fit = umap.UMAP()
u = fit.fit_transform(X2)
print("UMAP results \n",u)
plt.scatter(u[:,0], u[:,1],cmap='rainbow', alpha = 0.6 , s=10)
plt.title("UMAP Output", weight = 'bold', fontsize  = '15')
plt.xlabel("X(Dimension 1)")
plt.ylabel("Y(Dimension 2)")
plt.savefig("UMAP_Scatter_plot")
plt.show()
print(np.shape(u))

