
import requests

from pysle import isletool
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
import re
import os

from .helpers import CharacterTable



def download_data_maybe(fname='ISLEdict.txt', url='http://isle.illinois.edu/sst/data/g2ps/English/ISLEdict.html', cache_subdir='datasets'):

    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print('Downloading data from ', url, 'to', fpath)
        r = requests.get(url)
        assert r.status_code == 200
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.content, 'lxml')
        with open(fpath, 'w') as fo:
            fo.write(soup.text.strip())
    return fpath


def get_data(seed=42, test_size=0.20, verbose=0, maxlen_x=None, maxlen_y=None, blacklist='()0123456789%.?"-_', max_phonemes=np.inf, max_chars=np.inf, phon_sep='', unique_graphemes=False, unique_phonemes=True):
    """Process ISLEDICT pronounciation dictionary to return unique phonemes two graphemes"""

    path = download_data_maybe()

    # load data
    isleDict = isletool.LexicalTool(path)
    X = []
    y = []
    for phrase in isleDict.data.keys():
        for pronounciation in zip(*isleDict.lookup(phrase)):
            xx = []
            for syllableList, stressedSyllableList, stressedPhoneList in pronounciation:
                xx += list(itertools.chain(*syllableList))
            y.append(phon_sep.join(xx))
            X.append(phrase)
    if verbose: print('loaded entries {}'.format(len(X)))

    # filter out duplicate X's
    if unique_phonemes:
        y, X = zip(*dict(zip(y, X)).items())
        if verbose: print('removed duplicate phonemes leaving {}'.format(len(X)))

    # filter out duplicates Y's
    if unique_graphemes:
        X, y = zip(*dict(zip(X, y)).items())
        if verbose: print('removed duplicate graphemes leaving {}'.format(len(X)))

    # split data (we must set asside test data before cleanign so it's always the same)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # filter out duplicate entries like 'HOUSE(2) or multi words CAT-DOG and CAT_DOG'
    p = re.compile('[%s]' % (re.escape(blacklist)))
    X_train, y_train = zip(*[(x, y) for x, y in zip(X_train, y_train) if not bool(p.findall(x))])
    X_test, y_test = zip(*[(x, y) for x, y in zip(X_test, y_test) if not bool(p.findall(x))])
    if verbose:
        print('removed blacklisted entries leaving {}'.format(len(X_train) + len(X_test)))

    # filter out complex entries if needed
    before_x = len(y_train)
    X_train, y_train = zip(*[(x, y) for x, y in zip(X_train, y_train) if len(y) <= max_phonemes and len(x) <= max_chars])
    X_test, y_test = zip(*[(x, y) for x, y in zip(X_test, y_test) if len(y) <= max_phonemes and len(x) <= max_chars])
    if verbose:
        print('restricted to less than {} phonemes leaving {} entries or {:2.2f}%'.format(max_phonemes, len(X_train) + len(X_test), len(X_train)/before_x*100))

    # FIXME it's slow in the next few lines
    # encode x and y and pad them
    xtable = CharacterTable()
    xtable.fit(X_test + X_train)
    if maxlen_x:
        xtable.maxlen = maxlen_x
    X_train = xtable.encode(X_train)
    X_test = xtable.encode(X_test)

    ytable = CharacterTable()
    ytable.fit(y_test + y_train)
    if maxlen_y:
        ytable.maxlen = maxlen_y
    y_train = ytable.encode(y_train)
    y_test = ytable.encode(y_test)

    if verbose:
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)

    return (X_train, y_train), (X_test, y_test), (xtable, ytable)
