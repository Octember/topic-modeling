
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

    def clean(doc):
        return doc[field].replace('\r\n', '').lower()


    def process(document):
        cleaned = clean(document)

        # tokenize
        tokens = tokenizer.tokenize(cleaned)

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

    cleaned_docs = filter(clean, review_json)

    documents = [process(s) for s in progress(cleaned_docs)]

    # It is possible that the process() step generated some empty documents, 
    # remove those
    return filter(lambda x: len(x) > 0, documents), cleaned_docs

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
    word_counts, documents = load_reviews('./review-titles-lg.json', 'title')

    print "Generate vocabulary"
    vocabulary = generate_vocabulary(word_counts)

    print "Got a corupus of %d documents" % len(word_counts)
    print "Got a vocab of %d words" % len(vocabulary)

    print "Building doc term thing..."
    vocabulary_with_index = {word: index for index, word in enumerate(vocabulary)}

    doc_term_matrix = build_document_term_matrix(word_counts, vocabulary_with_index)

    for num_topics in (50,):
        print "Making model with %d topics" % num_topics
        model = lda.LDA(n_topics=num_topics, n_iter=500, random_state=1)
        model.fit(doc_term_matrix)

        topic_word = model.topic_word_  # model.components_ also works
        n_top_words = 8
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))


        # counts = {i: 0 for i in range(num_topics)}

        # for i, doc_topic in enumerate(model.doc_topic_):
        #     top_topic = doc_topic.argmax()
        #     # print("{} (top topic: {})".format(documents[i], doc_topic[i].argmax()))
        #     counts[top_topic] += 1

        # print counts
    