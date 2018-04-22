from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import numpy as np
from itertools import compress
import os
from collections import Counter
from empath import Empath
from scipy.sparse import csr_matrix

''' Regex to extract tweet text '''
p = re.compile("\"text\": \"(.*?)\"")

''' m is the Dictionary of user tweets, e.g. m[user_id] = [tweet_1, tweet_2,...]'''
m = {}
for twh in os.listdir('timelines'):
    with open('timelines/'+str(twh)) as f:
        m[twh] = ' '.join(p.findall(f.read()))
        f.close()

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(list(m.values()))
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(x)
''' Reducing TFIDF to the words used by at least 20%% of users.'''
b = np.where(tfidf.toarray() > 0, 1, 0)
mask = sum(b) >= 108
''' Feature 1: TFIDF and BoW of aggregated users' tweets.'''
tfidf = tfidf[:,mask]
bow = x[:,mask]
feature_names = vectorizer.get_feature_names()
feature_names = list(compress(feature_names, list(mask)))

np.save('tfidf.npy', tfidf)
np.save('bow.npy', bow)

""" Feature 2: Word2vec clusters"""
with open('w2v-500') as w2v_file:
    w2v = w2v_file.read().splitlines()
    w2v_file.close()

word_list = [re.sub("[^\w]", " ", i).split() for i in w2v]
wds = [i[0] for i in word_list]
cltrs = [int(i[len(i)-1]) for i in word_list]

user_to_w2v = np.zeros([len(m),500])

j = 0
for tw in m.values():
    print(j)
    tw_words = re.sub("[^\w]", " ", tw).split()
    c = list((Counter(tw_words) & Counter(wds)).elements())
    for i in c:
        user_to_w2v[j, cltrs[wds.index(i)]-1] = user_to_w2v[j, cltrs[wds.index(i)]-1]+1
    j = j+1
np.save('user_to_w2v.npy', user_to_w2v)"""
""" Feature 3: """
"""j = 0
user_to_liwc = np.zeros([len(m),194])
lexicon = Empath()
for key, value in m.items():
    lx = lexicon.analyze(value, normalize=True)
    user_to_liwc[j,:] = np.array(list(lx.values()))
    j= j+1
np.save('user_to_liwc.npy', user_to_liwc)

""" Feature 4 """

users = os.listdir('timelines')
Y = np.zeros([len(users)])

with open('users-moderates17acl.txt') as f:
    data = f.read().splitlines()
    f.close()

datalist = [re.sub("[^\w]", " ", i).split() for i in data]
urs = [i[0] for i in datalist]
score = [i[3] for i in datalist]

""" c is observed users"""
c = list((Counter(urs) & Counter(users)).elements())
mask = np.full([len(users)], False)

for i in c:
    Y[users.index(i)] = score[urs.index(i)]
    mask[users.index(i)] = True
np.save('y.npy', Y)
np.save('mask.npy', mask)
