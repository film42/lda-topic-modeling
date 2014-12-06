import numpy as np


class LDA:
    def __init__(self):
        self.documents = []
        self.alpha = 0.1
        self.beta = 0.1
        self.burn = 0
        self.length = 0
        self.n_documents = 0
        self.n_topics = 0
        self.vocab_size = 0
        self.t_word_count_matrix = np.zeros((self.n_topics, self.vocab_size))
        self.t_docs_count_matrix = np.zeros((self.n_topics, self.vocab_size))

    def gibbs_sampler(self):
        z = np.random.randint(self.n_topics)
        # Iterations
        for t in range(1, self.burn + self.length):
            # Documents: n
            for i in range(1, self.n_documents):
                # Create pi vector with (z, doc)
                pi_vector = self.create_pi_vector(z, i)
                # Draw from categorical
                new_z = pi_vector[self.draw_from_distribution(pi_vector)]

            # Yield iteration
            yield t

    def draw_from_distribution(self, pi_vector):
        # Normalize pi_vector
        pi_vec_sum = np.sum(pi_vector)
        norm_pi_vec = [pi_vector[p] / pi_vec_sum for p in xrange(self.n_topics)]
        # Draw once from the multinomial distribution, return index
        return np.random.multinomial(1, norm_pi_vec).argmax()

    def phi(self):
        pass

    def create_pi_vector(self, z, d):
        """
        Calculate the pi matrix, where each pi_j:
        pi_j = word with topic count * topic in doc count
        z: old z vector
        d: document index
        """
        derived_z_vec = [0]
        something = 0  # FIX
        document = self.documents[d]
        # For each topic generate a pi vector
        for j in range(1, self.n_topics + 1):
            # Find the somethings
            word_topic_count = (something + self.beta) / np.sum(something + (self.vocab_size * self.beta))
            topic_doc_count = (something + self.alpha) / np.sum(something + (self.n_topics * self.alpha))
            # Add to new z vector
            derived_z_vec.append(word_topic_count * topic_doc_count)

        return derived_z_vec

    def log_likelihood(self):
        return self.burn  # Fix