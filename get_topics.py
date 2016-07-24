
import numpy as np
import lda
import lda.datasets
import json
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from IPython import embed
from nltk.stem import *
import progressbar

progress = progressbar.ProgressBar()

# Return a nice list of review bodies
def load_reviews(filename, field):
    review_json = json.load(open(filename, 'r'))

    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()


    def process(review):
        cleaned = review[field].replace('\r\n', '')

        tokens = tokenizer.tokenize(cleaned.lower())

        return [stemmer.stem(token) for token in tokens]

    review_bodies = [process(s) for s in review_json]

    return review_bodies


# Filter out tokens we deem unworthy
def filter_tokens(tokens):

    #filter stop words
    def filter(word):
        return len(word) >= 3 and word not in get_stop_words('en')

    result = [token for token in tokens if filter(token)]

    return result



def generate_vocabulary(reviews):
    unique_tokens = set()

    for review in progress(reviews):
        for token in filter_tokens(review):
            unique_tokens.add(token)

    return tuple(unique_tokens)


def build_document_term_matrix(reviews, vocabulary):
    matrix = []

    progress = progressbar.ProgressBar()

    for review in progress(reviews):
        row = []

        for word in vocabulary:

            row.append(review.count(word))

        matrix.append(np.array(row))        

    return np.array(matrix)


if __name__ == "__main__":

    print "Load file"
    reviews = load_reviews('./review-titles-50000.json', 'title')

    print "Generate vocabulary"
    vocabulary = generate_vocabulary(reviews)

    print "Got a corupus of %d documents" % len(reviews)
    print "Got a vocab of %d words" % len(vocabulary)

    print "Building doc term thing..."
    doc_term_matrix = build_document_term_matrix(reviews, vocabulary)

    for num_topics in (20, 50, 100):
        print "Making model with %d topics" % num_topics
        model = lda.LDA(n_topics=num_topics, n_iter=500, random_state=1)
        model.fit(doc_term_matrix)

        topic_word = model.topic_word_  # model.components_ also works
        n_top_words = 8
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    