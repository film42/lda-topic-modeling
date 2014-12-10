import math
import string
import operator
from itertools import islice
from textblob import TextBlob


def take(n, iterable):
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))


def sort_dict(a_dict):
    return sorted(a_dict.items(), key=operator.itemgetter(1), reverse=True)


def has_numbers(s):
    return any(char.isdigit() for char in s)


def is_punctuation(s):
    return all(c in string.punctuation for c in s)


def has_others(s):
    return ('.' in s) or ('\'' in s)


class TFIDF:
    def __init__(self):
        self.weighted = False
        self.documents = []
        self.corpus_dict = {}

    def add_document(self, doc_name, text):
        # building a dictionary
        doc_dict = {}
        for w in TextBlob(unicode(text, errors='ignore')).words:

            if is_punctuation(w) or has_numbers(w) or has_others(w) or len(w) < 2:
                continue

            doc_dict[w] = doc_dict.get(w, 0) + 1
            self.corpus_dict[w] = self.corpus_dict.get(w, 0) + 1

        # add the document to the corpus
        self.documents.append([doc_name, doc_dict])

    def top_n_words(self, n):
        """
        This returns a list of [document_name, words_in_document] that are found in the top n words
        of the corpus dictionary.
        """
        corpus_size = len(self.corpus_dict)
        weighted_documents = {}
        pool = {}
        # Delete words with crazy large counts
        marked_for_deletion = []
        for word, count in self.corpus_dict.iteritems():
            if count >= 30:
                marked_for_deletion.append(word)

        for word in marked_for_deletion:
            del self.corpus_dict[word]

        # Get the top N weighted words from a document
        for document, words in self.documents:
            document_weighted_words = {}
            for word, count in words.iteritems():
                # Check word's existence before checking its weight
                if not word in self.corpus_dict:
                    continue

                if count >= 1:
                    try:
                        weight = (1 + math.log(count)) * math.log(corpus_size / float(self.corpus_dict[word]))
                    except Exception, e:
                        print e
                        print "Error at word: %s in %s with count %d" % (word, document.title(), count)
                        continue
                else:
                    weight = 0
                # Assign the weight to the word
                document_weighted_words[word] = weight

            # Create pool for each document
            for word, count in take(n, sort_dict(document_weighted_words)):
                pool[word] = True

        # Filter document to only contain words found in the pool
        print "Pool size: %d" % len(pool)
        for document, words in self.documents:
            print "Working on document: %s" % document

            weighted_documents[document] = []
            for word, count in words.iteritems():
                if pool.get(word, False):
                    weighted_documents[document].append(word)

        return weighted_documents