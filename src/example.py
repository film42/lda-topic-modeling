"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in

Finding scientifc topics (Griffiths and Steyvers)
"""

import numpy as np
from scipy.special import gammaln
from feature_selector import Data


np.set_printoptions(threshold=np.nan, linewidth=10000)


def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1, p).argmax()


def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    # print "Vec word index"
    # print vec
    # print vec.nonzero()[0]
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx


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


class LdaSampler(object):
    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    def _initialize(self, matrix):
        n_docs, vocab_size = matrix.shape

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size))
        self.nm = np.zeros(n_docs)
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                self.nmz[m, z] += 1
                self.nm[m] += 1
                self.nzw[z, w] += 1
                self.nz[z] += 1
                self.topics[(m, i)] = z

    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:, w] + self.beta) / (self.nz + self.beta * vocab_size)
        right = (self.nmz[m, :] + self.alpha) / (self.nm[m] + self.alpha * self.n_topics)
        # print left
        # print right
        # print ""
        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

    def log_likelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z, :] + self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m, :] + self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.nzw.shape[1]
        num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def run(self, matrix, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape
        self._initialize(matrix)

        for it in xrange(maxiter):
            print "KSDF: "
            print n_docs
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m, i)]
                    self.nmz[m, z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z, w] -= 1
                    self.nz[z] -= 1
                    #
                    # print "Iteration: %d" % it
                    # print z
                    # print self.nz
                    # print self.nzw
                    # print self.nm
                    # print self.nmz

                    p_z = self._conditional_distribution(m, w)

                    exit()

                    z = sample_index(p_z)

                    self.nmz[m, z] += 1
                    self.nm[m] += 1
                    self.nzw[z, w] += 1
                    self.nz[z] += 1
                    self.topics[(m, i)] = z

            # FIXME: burn-in and lag!
            yield self.phi()


if __name__ == "__main__":
    N_TOPICS = 10
    DOCUMENT_LENGTH = 100

    data = Data("../data/features")
    data.import_features()
    count_matrix = np.array(data.count_vectors())

    # print matrix.shape
    sampler = LdaSampler(N_TOPICS)

    print N_TOPICS
    # print count_matrix
    # print count_matrix.shape

    for it, phi in enumerate(sampler.run(count_matrix, maxiter=1)):
        print "%d\t%f" % (it, sampler.log_likelihood())

    # print sampler.topics
    # Create the topic matrix T x V
    # topic_matrix = np.zeros((N_TOPICS, count_matrix.shape[1]))
    # for position, topic in sampler.topics.iteritems():
    #     document, word = position
    #     topic_matrix[topic - 1, word] += 1

    # print topic_matrix
    # print sampler.nzw

    # Take only the top 10 indices from each topic in topic_matrix
    topic_scoped_matrix = np.zeros((N_TOPICS, 10))
    for index, topic in enumerate(sampler.nzw):
        topic_scoped_matrix[index] = np.argsort(-topic)[:10]

    # print topic_scoped_matrix
    topic_word_matrix = data.fill_topic_matrix(topic_scoped_matrix)

    import sys
    for t, arr in enumerate(topic_word_matrix):
        sys.stdout.write(str(t) + '\t')
        for word in arr:
            sys.stdout.write(word + '\t')

        sys.stdout.write('\n')