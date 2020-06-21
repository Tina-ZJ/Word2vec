# -*-coding: utf8 -*-
import sys
import gensim
import logging
import os.path
import multiprocessing
import json

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def train():
    # check and process input arguments
    if len(sys.argv) < 3:
        print((globals()['__doc__'] % locals()))
        sys.exit(1)
    inp, outp1, outp_vec = sys.argv[1:4]
    model = Word2Vec(LineSentence(inp), size=100, window=3, min_count=1, workers=multiprocessing.cpu_count())
    # save
    model.save(outp1)
    model.wv.save_word2vec_format(outp_vec,binary=False)


def test(model_path):
    model = Word2Vec.load(model_path)
    # show most similar words
    result = model.most_similar('标价机')
    print(json.dumps(result, ensure_ascii=False))
    # test does not match
    result = model.doesnt_match(u"对讲机 电台 耳机".split())
    print(result)
    # get score
    result = model.similarity(u'对讲机', u"耳机")
    print(result)
    # computer a word vector
    result = model[u'耳机']
    print(result)


if __name__=='__main__':
    train()
    #test('w2v.model')

