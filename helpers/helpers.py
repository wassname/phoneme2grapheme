import time
import re
import numpy as np
import nltk
import collections


# get short words
def cmudict_random_sample(n=100):
    '''get words shorter or equal to word_length'''
    cmudict = nltk.corpus.cmudict.dict()
    pairs = cmudict.items()
    # get random samples of cmudict of n=cutoff
    samples = (np.random.sample(n) * len(cmudict)).astype(int)
    pairs = [pairs[s] for s in samples]
    cmudict = collections.OrderedDict(pairs)
    return cmudict

# get short words


def cmudict_short_words(word_length=3):
    '''get words shorter or equal to word_length'''
    cmudict_raw = nltk.corpus.cmudict.dict()
    keys = np.array(cmudict_raw.keys())
    lens = np.array([len(k) for k in keys])
    vals = np.array(cmudict_raw.values())
    inds = np.argwhere(lens <= word_length)
    vals2 = vals[inds]
    keys2 = keys[inds]
    cmudict = dict(np.dstack((keys2.flatten(), vals2.flatten()))[0])  # must be a better way!
    return cmudict


def cmudict_short_phones(phone_length=3):
    '''get words shorter or equal to word_length'''
    cmudict_raw = nltk.corpus.cmudict.dict()
    vals = np.array(cmudict_raw.values())
    lens = np.array([len(p[0]) for p in vals])
    keys = np.array(cmudict_raw.keys())
    inds = np.argwhere(lens <= phone_length)
    vals2 = vals[inds]
    keys2 = keys[inds]
    cmudict = dict(np.dstack((keys2.flatten(), vals2.flatten()))[0])  # must be a better way!
    return cmudict


def unique(seq, idfun=None):
    '''get only unique items in list'''
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def split_string(word, inds):
    '''split string at indices'''
    splitword = []
    inds = np.sort(inds)
    for i in range(len(inds) - 1):
        j = inds[i]
        k = inds[i + 1]
        if k == 0 and j == len(word):
            continue
        part = word[j:k]
        splitword.append(part)
    return splitword


def remove_stress_arp(s):
    return re.sub('\d+', '', s)


def findall(pattern, target):
    inds = []
    while pattern in target:
        inds.append(target.index(pattern))
        target = target.replace(pattern, '_' * len(pattern), 1)
    return inds


def dist_in_letters(word, phones, phon_ind, match, lind):
    '''find distance between pheonom position and it's match
    provide:
        word e.g. 'baby'
        phones e.g. [u'B', u'EY1', u'B', u'IY0']
        phon_ind e.g. 2
        match e.g. B
        lind: positve of match in word e.g. 2
    '''
    dists = []
    letters_p_phon = 1.0 * len(word) / len(phones)
    phon_lind = phon_ind * letters_p_phon
    dist1 = abs(lind - phon_lind)
    dist2 = abs(lind + len(match) - phon_lind - len(match))
    dists.append(dist1)
    dists.append(dist2)

    return min(dists)


def write_cmudict_sample(n=200):
    import numpy as np
    cmudict = nltk.corpus.cmudict.dict()
    outf = 'cmudict_samples' + str(time.time()) + '.txt'
    fo = open(outf, 'w')
    samples = (np.random.random_sample(100) * len(cmudict)).astype(int)
    cmuk = cmudict.keys()
    for s in samples:
        k = cmuk[s]
        vs = cmudict[k]
        for v in vs:

            vstr = ' '.join(v)
            fo.write('{}  {}  {}\n'.format(k, k, vstr))
    fo.close()
    print("Please go to {} and place space between the letters that match the peonomes, this will be the positive test".format(outf))


def empty_tree(input_list):
    """Recursively iterate through values in nested lists."""
    if input_list:
        for item in input_list:
            if not isinstance(item, list) or not empty_tree(item):
                return False
    return True

import itertools
# based on https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py
# note: can't make sparse 3d matrices


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''

    def __init__(self, chars='', maxlen=None, null_char=' ', left_pad=False):
        self.chars = sorted(set([null_char] + list(chars)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen
        self.left_pad = left_pad
        self.null_char = null_char

    def fit(self, Cs, null_char=' '):
        """Determine chars and maxlen by fitting to data"""
        self.chars = sorted(set(itertools.chain([null_char], *Cs)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = max(len(c) for c in Cs)
        self.null_char = null_char

    def encode(self, Cs, maxlen=None):
        """Pass in an array of arrays to convert to integers"""
        maxlen = maxlen if maxlen else self.maxlen
        n = len(Cs)
        X = np.zeros((n, maxlen, len(self.chars)), dtype=np.bool)
        for j, C in enumerate(Cs):
            if self.left_pad:
                C = [self.null_char] * (maxlen - len(C)) + list(C)
            else:
                C = list(C) + [self.null_char] * (maxlen - len(C))
                for i, c in enumerate(C):
                    X[j, i, self.char_indices[c]] = True
        return X

    def decode(self, Xs, calc_argmax=True):
        if calc_argmax:
            Xs = Xs.argmax(axis=-1)
        return np.array(list([self.indices_char[x] for x in X] for X in Xs))


# show_results
from IPython.core.display import display, HTML
m=1.2
lighten=lambda x:1-(1/m-x/m)

def show_results(ytable, y_pred,y_test=None,X_test=None,xtable=None):
    """Show results which are darker when more confident"""
    html = '<table><tbody><thead>'
    html += '<tr><th>pronunciation</th><th>guess</th><th>spelling</th></tr>'
    html += '</thead>'
    p_pred = ytable.decode(y_pred)
    conf = y_pred.max(-1)
    for i in range(p_pred.shape[0]):
        html += '<tr>'

        if X_test is not None:
            p_test = xtable.decode(X_test)
            html+='<td>'
            for j in range(p_test.shape[1]):
                c=p_test[i][p_test.shape[1]-j-1]
                html+='<span style="color:rgba(0,0,0,{a:1.1f})">{c:}</span>'.format(c=c,a=1)
            html+='</td>'

        html+='<td>'
        for j in range(p_pred.shape[1]):
            c=p_pred[i][j]
            a=lighten(conf[i][j])
            html+='<span style="color:rgba(0,0,0,{a:1.1f})">{c:}</span>'.format(c=c,a=a)
        html+='</td>'

        if y_test is not None:
            html+='<td>'
            p_test = ytable.decode(y_test)
            for j in range(p_test.shape[1]):
                c=p_test[i][j]
                html+='<span style="color:rgba(0,0,0,{a:1.1f})">{c:}</span>'.format(c=c,a=1)
            html+='</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return HTML(html)

# test
# r=np.random.random((10,8,30))**20
# show_results(ytable, r)


class weighted_categorical_crossentropy(object):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        loss = weighted_categorical_crossentropy(weights).loss
        model.compile(loss=loss,optimizer='adam')
    """

    def __init__(self,weights):
        self.weights = K.variable(weights)

    def loss(self,y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= y_pred.sum(axis=-1, keepdims=True)
        # clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # calc
        loss = y_true*K.log(y_pred)*self.weights
        loss =-K.sum(loss,-1)
        return loss
