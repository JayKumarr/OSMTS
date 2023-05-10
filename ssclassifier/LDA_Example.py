if __name__ == '__main__':
    import pandas as pd
    import gensim

    texts = [['kumar'], ['tharwani', 'Phd'], ['Jay', 'Phd']]

    dictionary = gensim.corpora.Dictionary(texts)
    bow = [dictionary.doc2bow(doc) for doc in texts]  # list[ doc1[tuple(word_id1, occurence),...], doc2[], ...  ]

    # lda_bow_tfidf = gensim.models.LdaMulticore(bow, num_topics=2, id2word=ttt, passes=2, workers=2)

    from gensim import corpora, models

    tfidf = models.TfidfModel(bow)
    corpus_tfidf = tfidf[bow]

    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=2, id2word=dictionary, passes=2, workers=2)


    sorted_by_topic_score = sorted(lda_model_tfidf[corpus_tfidf[0]], key=lambda tup: -1 * tup[1])



    a = 10