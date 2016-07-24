
import numpy as np
import lda
import lda.datasets
import json
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from IPython import embed
from nltk.stem import *
import progressbar
from scipy.sparse import lil_matrix


progress = progressbar.ProgressBar()

# Return a nice list of review bodies
def load_reviews(filename, field):
    review_json = json.load(open(filename, 'r'))

    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()


    # Given a list of words, return a map of word->count
    def word_count(words):
        result = {}
        for word in words:
            if word not in result:
                result[word] = 0
            result[word] += 1
        return result 


    def process(review):
        cleaned = review[field].replace('\r\n', '')

        # tokenize
        tokens = tokenizer.tokenize(cleaned.lower())

        def remove_stopwords(word):
            return len(word) >= 3 and word not in get_stop_words('en')

        # filter
        filtered_tokens = filter(remove_stopwords, tokens)

        # stem
        stemmed_tokens = map(stemmer.stem, filtered_tokens)

        # Turn into map
        word_counts = word_count(stemmed_tokens)

        return word_counts

    progress = progressbar.ProgressBar()

    review_bodies = [process(s) for s in progress(review_json)]

    return review_bodies



# Return map from words->index
def generate_vocabulary(reviews):
    unique_tokens = set()

    for review in progress(reviews):
        for token in review.keys():
            unique_tokens.add(token)

    return tuple(unique_tokens)

def build_document_term_matrix(reviews, vocabulary):

    progress = progressbar.ProgressBar()

    matrix = lil_matrix((len(reviews), len(vocabulary)), dtype=np.int32)

    for row in progress(range(len(reviews))):
        review = reviews[row]

        for word in review:
            column = vocabulary[word]
            matrix[row, column] = review[word]    

    return matrix


if __name__ == "__main__":

    print "Load file"
    reviews = load_reviews('./question-titles-md.json', 'title')

    print "Generate vocabulary"
    vocabulary = generate_vocabulary(reviews)

    print "Got a corupus of %d documents" % len(reviews)
    print "Got a vocab of %d words" % len(vocabulary)

    print "Building doc term thing..."
    vocabulary_with_index = {word: index for index, word in enumerate(vocabulary)}

    doc_term_matrix = build_document_term_matrix(reviews, vocabulary_with_index)

    for num_topics in (50, 100):
        print "Making model with %d topics" % num_topics
        model = lda.LDA(n_topics=num_topics, n_iter=500, random_state=1)
        model.fit(doc_term_matrix)

        topic_word = model.topic_word_  # model.components_ also works
        n_top_words = 8
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    