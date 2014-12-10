from feature_selector import Data
from scipy.special import gammaln
import numpy as np


np.set_printoptions(threshold=np.nan, linewidth=10000)


def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K * alpha)


def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    # print vec
    # print vec.nonzero()[0]
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx


class LDA:
    def __init__(self, documents, alpha, beta, n_topics):
        self.documents = documents
        self.n_documents = len(documents)
        self.alpha = alpha
        self.beta = beta
        self.n_topics = 0
        self.vocab_size = len(self.documents[0])  # Normalized, so safe to ask here
        self.n_topics = n_topics

        # Cached matrices
        # self.t_assignments = np.zeros((self.n_topics, self.vocab_size))
        self.t_docs_count_matrix = np.zeros((self.n_topics, self.n_documents))
        self.t_word_count_matrix = np.zeros((self.n_topics, self.vocab_size))
        self.n_documents_vec = np.zeros(self.n_documents)
        self.n_topics_vec = np.zeros(self.n_topics)
        self.t_assignments = np.zeros((self.n_documents, self.vocab_size))

        # Init
        for d in xrange(self.n_documents):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(self.documents[d, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                self.t_docs_count_matrix[z, d] += 1
                self.n_documents_vec[d] += 1
                self.t_word_count_matrix[z, w] += 1
                self.n_topics_vec[z] += 1
                self.t_assignments[d, i] = z

    def sample(self, max_iterations=500):
        # Iterations
        for t in xrange(max_iterations):
            # Documents: n
            for d in xrange(self.n_documents):  # Check
                # Words in document d
                # print self.documents[d, :]
                for i, w in enumerate(word_indices(self.documents[d, :])):
                    z = self.t_assignments[d, i]
                    self.t_word_count_matrix[z, w] -= 1
                    self.n_documents_vec[d] -= 1
                    # print "%d, %d" % (z, w)
                    self.t_docs_count_matrix[z, d] -= 1
                    self.n_topics_vec[z] -= 1

                    # print "Iteration: %d" % it
                    # print z
                    # print self.nz
                    # print self.nzw
                    # print self.nm
                    # print self.nmz

                    pi_vector = self.create_pi_vector(w, d)

                    # print pi_vector

                    z = self.draw_from_multinomial(pi_vector)

                    self.t_docs_count_matrix[z, d] += 1
                    self.n_documents_vec[d] += 1
                    self.t_word_count_matrix[z, w] += 1
                    self.n_topics_vec[z] += 1
                    self.t_assignments[d, i] = z

                    # # Create the new pi vector
                    # pi_vector = self.create_pi_vector(z, w, d)
                    #
                    # print pi_vector
                    #
                    # # Draw from categorical
                    # assignment = pi_vector[self.draw_from_multinomial(pi_vector)]
                    # old_topic = z[w]
                    #
                    # self.t_word_count_matrix[old_topic, w] -= 1  # if not zero
                    # self.t_docs_count_matrix[old_topic, d] -= 1  # if not zero
                    #
                    # self.t_word_count_matrix[assignment, w] += 1
                    # self.t_docs_count_matrix[assignment, d] += 1
                    #
                    # z[w] = assignment

            # Yield iteration
            yield t

    def draw_from_multinomial(self, pi_vector):
        # Draw once from the multinomial distribution, return index
        return np.random.multinomial(1, pi_vector).argmax()

    def create_pi_vector(self, w, d):
        """
        Calculate the pi matrix, where each pi_j:
        pi_j = word with topic count * topic in doc count
        z: old z vector
        d: document index
        """
        # new_pi_vector = []
        # print self.n_topics_vec
        left = (self.t_word_count_matrix[:, w] + self.beta) / (self.n_topics_vec + self.beta * self.vocab_size)
        # print left
        right = (self.t_docs_count_matrix[:, d] + self.alpha) / (self.n_documents_vec[d] + self.alpha * self.n_topics)
        # print right
        pi_vector = left * right
        pi_vector /= np.sum(pi_vector)
        return pi_vector

        # for j in xrange(self.n_topics):
        #     numerator = self.t_word_count_matrix[j, w] + self.beta
        #     if z[w] is j:
        #         numerator -= 1
        #
        #     denominator = 0.
        #     for w2 in xrange(self.vocab_size):
        #         denominator += self.t_word_count_matrix[j, w2]
        #         if z[w] is j:
        #             denominator -= 1
        #     denominator += self.beta * self.vocab_size
        #
        #     t_doc_count = self.t_docs_count_matrix[j, d] + self.alpha
        #     if z[w] is j:
        #         t_doc_count -= 1
        #
        #     new_pi_vector.append((numerator / float(denominator)) * t_doc_count)
        #
        # print z
        #
        # print new_pi_vector
        #
        # print normalize(new_pi_vector)
        #
        # # Done. Return the new pi vector
        # return normalize(new_pi_vector)

    def log_likelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.t_word_count_matrix[z, :] + self.beta)
            lik -= log_multi_beta(self.beta, self.vocab_size)

        for d in xrange(self.n_documents):
            lik += log_multi_beta(self.t_docs_count_matrix[:, d] + self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

if __name__ == "__main__":
    N_TOPICS = 20

    data = Data("../data/features")
    data.import_features()
    count_matrix = np.array(data.count_vectors())

    # print matrix.shape
    sampler = LDA(count_matrix, alpha=0.1, beta=0.1, n_topics=N_TOPICS)

    print "Running %d Topics:" % N_TOPICS
    # print count_matrix
    # print count_matrix.shape

    for it, phi in enumerate(sampler.sample(max_iterations=800)):
        print "%d\t%f" % (it + 1, sampler.log_likelihood())

    # Take only the top 10 indices from each topic in topic_matrix
    topic_scoped_matrix = np.zeros((N_TOPICS, 10))
    for index, topic in enumerate(sampler.t_word_count_matrix):
        topic_scoped_matrix[index] = np.argsort(-topic)[:10]

    # print topic_scoped_matrix
    topic_word_matrix = data.fill_topic_matrix(topic_scoped_matrix)

    import sys
    for t, arr in enumerate(topic_word_matrix):
        sys.stdout.write(str(t) + '\t')
        for word in arr:
            index = data.index_of(word)
            sys.stdout.write(word + ' (' + str(int(sampler.t_word_count_matrix[t][index])) + ')\t')

        sys.stdout.write('\n')