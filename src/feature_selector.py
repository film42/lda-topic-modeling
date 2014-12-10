import os
from tf_idf import TFIDF
from document import Document


class Data:
    def __init__(self, path):
        self.path = path
        self.corpus = {}
        self.documents = {}
        self.import_features()
        self.document_types = []

    def load_feature(self, name, words):
        # Save document
        self.documents[name] = words
        # Save the words for counting later
        for word in words:
            self.corpus[word] = self.corpus.get(word, 0) + 1

    def fill_topic_matrix(self, topic_matrix):
        result_matrix = list(xrange(len(topic_matrix)))
        words = self.corpus.keys()
        for i, vector in enumerate(topic_matrix):
            result_matrix[i] = [words[int(k)] for k in vector]

        return result_matrix

    def index_of(self, word):
        return self.corpus.keys().index(word)

    def import_features(self):
        for file_name in os.listdir(self.path):
            with open("%s/%s" % (self.path, file_name), 'r') as f:
                words = f.read().split("\n")[:-1]
                self.load_feature(file_name, words)

    def count_vectors(self):
        """
        The point of build is to load all documents into memory and generate a count vector for each document
        """
        count_vectors = []
        for document, words in self.documents.iteritems():
            count_vector = {}
            for corpus_word in self.corpus.keys():
                # Count the number of times the word in the corpus occurs in the document
                count_vector[corpus_word] = words.count(corpus_word)
            # Save the count vector
            # count_vectors.append((document, count_vector.values()))
            count_vectors.append(count_vector.values())
            # Save the document type
            document_type = document.split("/")[0]
            self.document_types.append(document_type)

        return count_vectors


class FeatureSelector:
    def __init__(self):
        self.table = TFIDF()

    def run(self, index_file):
        """
        Generate the features using Top N algorithm
        """
        with open(index_file) as f:
            lines = f.readlines()
            for line in lines:
                name = line[:-1]
                with open("../data/scoped/%s" % name, 'r') as d:
                    document = Document(d.read())
                    self.table.add_document(name, document.content_lower)

        new_data_set = self.table.top_n_words(10)
        for document_name, words in new_data_set.iteritems():

            with open("../data/scoped/%s" % document_name, 'r') as d:
                    document = Document(d.read())

            path_name = "../data/features/%s" % document_name

            with open("%s" % path_name, 'w') as f:
                for word in words:
                    for _ in xrange(document.count(word)):
                        f.write(word)
                        f.write("\n")


if __name__ == '__main__':
    # FeatureSelector().run("../data/scoped/index-small.txt")
    FeatureSelector().run("../data/scoped/index-medium.txt")
    # data = Data("../data/features")
    # data.import_features()
    # print data.corpus.keys()
    # print data.count_vectors()