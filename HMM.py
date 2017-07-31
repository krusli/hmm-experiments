import scipy.special, scipy.stats, scipy.misc, math, copy, random
from pprint import pprint
from collections import defaultdict
import numpy as np

from multiprocessing import Pool
from datetime import datetime
import multiprocessing

class HMM:
    def __init__(self):
        self.emission_prob_cache = {}
        self.deltas = []
        self.deltas_went_up = 0  # no of times delta went up

        self.EPSILON = 0.0001
        self.DELTA_RISE_THRES = 2  # allow delta to go up only twice (usually because of NBD_MLE)

    def emission_prob(self, a, b, theta, k, i):  # k: state, i: item (observation - indices of the observed items) (array)
        """
        a[k], b[k]: shape and scale params of Gamma distribution -> params of NBD distribution
                    for the k-th state
        theta[i][k]: each column is a distribution (params of multinomial) -> preference of a class of users towards the items
        """
        # nbinom.pmf(k) = choose(k+n-1, n-1) * p**n * (1-p)**k
        # Pnbd(N; a, b) = choose(a+N-1, N) * (b/(b+1))**N * (1 - (b/(b+1)))**a

        # TODO:
        """Gotcha: a should not exceed 800k, overflows to inf (infinity)"""

        # negative binomial
        gamma = scipy.special.gamma  # gamma function
        factorial = scipy.misc.factorial
        comb = scipy.misc.comb
        gammaln = scipy.special.gammaln

        p = b[k] / (b[k] + 1)  # probability for NBD
        x = len(i)  # number of items in the observation

        def nbinom(x, a, b):
            assert a > 0
            # TODO: comb(x+a-1, a) results in 0 probabilities
            # NOTE: x (no of items in a time period) should be greater than 0 to ensure no 0 probabilities because of invalid inputs to comb(x+a, a)
            return comb(x+a-1, a) * (b/(b+1))**x * (1 - b/(b+1))**a

        # Multinomial
        item_probs = [row[k] for row in theta]  # param
        # print(item_probs)
        multinomial = scipy.stats.multinomial(n=len(i), p=item_probs)

        # parse observation from indices to counts of items
        # (index) [0, 4, 4, ...] => [1, 0, 0, 0, 2, 0, ...]
        observation_item_counts = [0 for i in range(len(theta))]  # theta has |I| (total no of items) rows
        for item_index in i:
            observation_item_counts[item_index] += 1

        if (k, tuple(i)) not in self.emission_prob_cache:
            # get the joint pdf
            if x > 0:
                # assert(nbinom(x, a[k], b[k]) * multinomial.pmf(observation_item_counts) > 0)
                retval = nbinom(x, a[k], b[k]) * multinomial.pmf(observation_item_counts)
            else:
                # assert(nbinom(x, a[k], b[k]) > 0)
                retval = nbinom(x, a[k], b[k]) * 1 # if user does not look at any items, simply the probability of looking at

            self.emission_prob_cache[(k, tuple(i))] = retval
        else:
            retval = self.emission_prob_cache[(k, tuple(i))]

        return retval
                                             # 0 items

    def forward(self, a, b, theta, pi, transition_prob, observation_seq):
        T = len(observation_seq)
        no_of_states = len(theta[0])
        scaling_factors = [1]
        # alphas = []
        alphas = np.zeros(shape=(T, no_of_states))

        # initialisation
        emissions = np.diag([self.emission_prob(a, b, theta, j, observation_seq[0]) for j in range(no_of_states)])
        alphas[0, :] = np.dot(emissions, pi)

        scaling_factor = alphas[0, :].sum()
        # print('scaling_factor: {}'.format(scaling_factor))
        if scaling_factor != 0:
            alphas[0, :] = alphas[0, :]/scaling_factor
        scaling_factors.append(scaling_factor)

        # recursion
        for t in range(1, T):
            emissions = np.diag([self.emission_prob(a, b, theta, j, observation_seq[t]) for j in range(no_of_states)])
            alphas[t, :] = np.dot(np.dot(emissions, np.transpose(transition_prob)), alphas[t-1, :])
            scaling_factor = alphas[t, :].sum()  # P(I_u^t | I_u^1:t-1)
            if scaling_factor != 0:
                alphas[t, :] = alphas[t, :]/scaling_factor
            scaling_factors.append(scaling_factor)

        # print('\nalphas')
        # pprint(alphas)

        return (alphas, scaling_factors)

    def backward(self, a, b, theta, transition_prob, observation_seq, scaling_factors):
        T = len(observation_seq)
        no_of_states = len(theta[0])
        # betas = [[] for i in range(T + 1)]
        betas = np.zeros(shape=(T+1, no_of_states))

        # initialisation
        # note that b_T:T is not scaled, only subsequent betas are
        # betas[T] = [1 for _ in range(no_of_states)]  # initial state assumed as given
        betas[T, :] = [1] * no_of_states

        # recursion
        for t in range(T-1, 0, -1):
            # indexing differences between betas and observation_seq
            emissions = np.diag([self.emission_prob(a, b, theta, j, observation_seq[t]) for j in range(no_of_states)])
            betas[t, :] = np.dot(np.dot(np.transpose(transition_prob), emissions), betas[t+1, :])

            # scaling
            scaling_factor = scaling_factors[t+1]
            if scaling_factor != 0:
                betas[t, :] = betas[t, :]/scaling_factor

        betas = betas[1:]  # remove first element, unneeded
        return betas

    def converged(self, a_old, a, b_old, b, theta_old, theta, pi_old, pi, transition_prob_old, transition_prob):
        delta = 0
        no_of_items = len(theta)
        no_of_states = len(theta[0])
        for k in range(no_of_states):
            delta += abs(a[k] - a_old[k]) + abs(b[k] - b_old[k]) + abs(pi[k] - pi_old[k])
            for l in range(no_of_states):
                delta += abs(transition_prob[k][l] - transition_prob_old[k][l])
            for i in range(no_of_items):
                delta += abs(theta[i][k] - theta_old[i][k])
        print('delta: {}'.format(delta))

        if len(self.deltas) > 1:
            if delta > self.deltas[-1]:
                self.deltas_went_up += 1
        self.deltas.append(delta)

        if delta < self.BAUM_WELCH_EPSILON:
            return True
        return False

    def baum_welch(self, a, b, theta, pi, transition_prob, observation_seqs, prior_weight_items, prior_weight_states,
                    baum_welch_epsilon=0.01, iterations=200):
        self.BAUM_WELCH_EPSILON = baum_welch_epsilon
        self.ITERATIONS = iterations

        T = len(observation_seqs[0])
        no_of_items = len(theta)        # |I|
        no_of_states = len(theta[0])    # K
            # TODO: assert len(theta[0]) == len(a) == len(b) == len(transition_prob) == len(transition_prob[0])
        no_of_users = len(observation_seqs)
        a = copy.deepcopy(a)
        b = copy.deepcopy(b)
        theta = copy.deepcopy(theta)
        transition_prob = copy.deepcopy(transition_prob)
        ALPHA_I = prior_weight_items
        ALPHA_K = prior_weight_states

        # for calculating theta. These values don't change, so don't repeatedly calculate them
        obsv_counts = np.zeros(shape=(T, no_of_items, no_of_users))  # obsv_counts[t][i]
        total_counts = np.zeros(shape=(T, no_of_users))  # total_counts[t]
        for t in range(T):
            for i in range(no_of_items):
                for u in range(no_of_users):
                    obsv_counts[t][i][u] = sum(1 for item in observation_seqs[u][t] if item == i)
            for u in range(no_of_users):
                total_counts[t][u] = len(observation_seqs[u][t])

        # EM algorithm
        for iteration in range(self.ITERATIONS):
            print('\nIteration {}'.format(iteration + 1))
            # input('Press ENTER to continue: ')

            # print('\nStarting probs')
            # pprint(pi)
            # print('\nTransition probs')
            # pprint(transition_prob)

            # print('\na')
            # pprint(a)
            # print('\nb')
            # pprint(b)
            # print('\nThetas')
            # pprint(theta)

            a_old = copy.deepcopy(a)
            b_old = copy.deepcopy(b)
            theta_old = copy.deepcopy(theta)
            pi_old = copy.deepcopy(pi)
            transition_prob_old = copy.deepcopy(transition_prob)
            alphas = np.zeros(shape=(no_of_users, T, no_of_states))
            scaling_factors = np.zeros(shape=(no_of_users, T+1))
            betas = np.zeros(shape=(no_of_users, T, no_of_states))

            # Expectation stage

            params = []
            u = 0
            for seq in observation_seqs:
                params.append( (a, b, theta, pi, transition_prob, seq, u) )
                u += 1

            print('forward-backward')
            print(datetime.now())
            try:
                pool = Pool(multiprocessing.cpu_count())
                results = pool.map(forward_backward_helper, params)
            finally:
                pool.close()
                pool.join()

            for result in results:
                u, alphas_u, scaling_factors_u, betas_u = result
                alphas[u] = alphas_u
                scaling_factors[u] = scaling_factors_u
                betas[u] = betas_u

            # for seq in observation_seqs:
            #     print('Forward')
            #     alphas_u, scaling_factors_u = self.forward(a, b, theta, pi, transition_prob, seq)
            #     print('Backward')
            #     betas_u = self.backward(a, b, theta, transition_prob, seq, scaling_factors_u)
            #     alphas.append(alphas_u)
            #     scaling_factors.append(scaling_factors_u)
            #     betas.append(betas_u)

            # print('\nalphas')
            # pprint(alphas)
            # print('\nbetas')
            # pprint(betas)

            # gammas = [[] for u in range(len(observation_seqs))]
            # xis = [[] for u in range(len(observation_seqs))]
            gammas = np.zeros(shape=(no_of_users, T, no_of_states))
            xis = np.zeros(shape=(no_of_users, T-1, no_of_states, no_of_states))

            print('Expectation')

            # for u in range(len(observation_seqs)):  # for user in users
            #     gamma = [[alphas[u][t][i] * betas[u][t][i]
            #                   for i in range(no_of_states)]
            #              for t in range(T)]
            #     xi = [[[alphas[u][t][i] * transition_prob[i][j] * self.emission_prob(a, b, theta, j, observation_seqs[u][t+1]) * betas[u][t+1][j] \
            #             / scaling_factors[u][t+2]  # different indexing in scaling_factors and alphas/betas
            #                 for j in range(no_of_states)]   # to-state
            #              for i in range(no_of_states)]  # from-state
            #           for t in range(T - 1)]
            #     gammas.append(gamma)
            #     xis.append(xi)

            print('gamma')
            print(datetime.now())
            params = []
            for u in range(len(observation_seqs)):
                params.append( (u, alphas, betas, T, no_of_states) )
            try:
                pool = Pool(multiprocessing.cpu_count())
                gammas_res = pool.map(calculate_gamma, params)
            finally:
                pool.close()
                pool.join()
            for result in gammas_res:
                # pprint(result)
                u, gamma_u = result
                gammas[u] = gamma_u

            print('xi')
            print(datetime.now())
            params = []
            for u in range(len(observation_seqs)):
                params.append( (u, a, b, theta, observation_seqs, alphas, betas, transition_prob, scaling_factors, T, no_of_states) )
            try:
                pool = Pool(multiprocessing.cpu_count())
                xis_res = pool.map(calculate_xi, params)
            finally:
                pool.close()
                pool.join()
            for result in xis_res:
                u, xi_u = result
                xis[u] = xi_u

            # for u in range(no_of_users):
            #     print('\nAlphas[{}]'.format(u))
            #     pprint(alphas[u])
            #     print('\nBetas[{}]'.format(u))
            #     pprint(betas[u])
            #     print('\nGammas[{}]'.format(u))
            #     pprint(gammas[u])
            #     print('\nXis[{}]'.format(u))
            #     pprint(xis[u])

            print('Maximisation')
            # Maximisation stage - update parameters to maximise likelihood
            # update starting probabilities - MAP estimate
            print('pi')
            print(datetime.now())
            for i in range(no_of_states):
                pi[i] = ( sum(gammas[u][0][i] for u in range(no_of_users)) + ALPHA_K/no_of_states - 1) \
                                            / ( sum(sum(gammas[u][0][k] for k in range(no_of_states))  for u in range(no_of_users)) + ALPHA_K - no_of_states)

            print('A')
            print(datetime.now())
            # update transition probabilities - MAP estimate
            for i in range(no_of_states):
                for j in range(no_of_states):
                    # transition_prob[i][j] = (sum(sum(xis[u][t][i][j] for t in range(T-1)) for u in range(no_of_users)) + ALPHA_K/no_of_states - 1 ) \
                    #                             / (sum(sum(gammas[u][t][i] for t in range(T-1)) for u in range(no_of_users)) + ALPHA_K - no_of_states)

                    xis_ = np.swapaxes(xis, 0, 2)
                    xis_ = np.swapaxes(xis_, 1, 3)
                    gammas_ = np.swapaxes(gammas, 0, 2)

                    transition_prob[i][j] = (xis_[i][j].sum() + ALPHA_K/no_of_states - 1 ) \
                                                / (gammas_[i][:T-1].sum() + ALPHA_K - no_of_states)

                    assert(transition_prob[i][j] >= 0)

            print('theta')
            print(datetime.now())
            # update emission probabilities: multinomial
            # for i in range(no_of_items):
            #     for k in range(no_of_states):
            #         gammas_ = np.swapaxes(gammas, 0, 2)  # gammas[k][t][u]
            #         numerator = 0
            #         denominator = 0
            #         for t in range(T):
            #             numerator += (gammas_[k][t] * [sum(1 for item in observation_seqs[u][t] if item == i) for u in range(no_of_users)]).sum()
            #             denominator += (gammas_[k][t] * [len(observation_seqs[u][t]) for u in range(no_of_users)]).sum()
            #         numerator += ALPHA_I/no_of_items - 1
            #         denominator += ALPHA_I - no_of_items
            #
            #         assert(numerator / denominator >= 0)
            #         theta[i][k] = numerator/denominator

            # print('\nStarting probs')
            # pprint(pi)
            # print('\nTransition probs')
            # pprint(transition_prob)

            params = []
            for i in range(no_of_items):
                for k in range(no_of_states):
                    params.append( (i, k, no_of_items, no_of_states, no_of_users, gammas, T, observation_seqs, obsv_counts, total_counts, ALPHA_I) )
            try:
                pool = Pool(multiprocessing.cpu_count())
                theta_res = pool.map(calculate_theta, params)
            finally:
                pool.close()
                pool.join()
            for result in theta_res:
                i, k, theta_ik = result
                theta[i][k] = theta_ik

            if self.deltas_went_up < self.DELTA_RISE_THRES:  # let delta go up only twice.
                """
                Explanation: if the counts in the dataset doesn't fit results from an actual NBD, the algorithm for NBD will converge to
                progressively higher values of a and b (which ARE incorrect and even further from the "truth").

                We can demonstrate this using the example test case - just replace the empty observations with one random item - the algorithm
                will behave like this.

                By no longer maximising a and b, we can get a "good enough" estimate for a and b (MLE of NBD) which seems to be quite
                unstable (ESPECIALLY if the data does not fit emissions from a negative binomial distribution), yet still maximise
                the rest of the model to fit the users' observation data well, hopefully getting a "good enough" result.
                """
                # TODO: replace with whenever delta jumps up
                print('a, b')
                print(datetime.now())
                # MLE estimate of weighted NBD (NBD mixing weights of Multinomial distribution)
                a, b = self.NBD_MLE(a, b, gammas, observation_seqs)


            # print('\na')
            # pprint(a)
            # print('\nb')
            # pprint(b)
            # print('\nThetas')
            # pprint(theta)

            if self.converged(a_old, a, b_old, b, theta_old, theta, pi_old, pi, transition_prob_old, transition_prob):
                return (a, b, theta, pi, transition_prob, alphas)

            # clear out the emission prob cache
            self.emission_prob_cache = {}

        return (a, b, theta, pi, transition_prob, alphas)

    def NBD_MLE(self, a, b, gammas, observation_seqs):
        # TODO: might not converge if not actually a NBD

        """Section 2.1 of Minka 2002"""
        # print('@ NBD_MLE')
        counts = copy.deepcopy(observation_seqs)
        no_of_users = len(counts)
        no_of_states = len(gammas[0][0])

        log = np.log
        polygamma = scipy.special.polygamma

        T = len(observation_seqs[0])
        for u in range(len(counts)):  # count number of items selected for each user at each time
            for t in range(T):
                counts[u][t] = len(counts[u][t])

        # print('\nCounts')
        # pprint(counts)

        average_counts = []
        for k in range(len(a)):  # K states
            numerator = 0
            denominator = 0
            for u in range(len(counts)):  # for each user
                for t in range(T):
                    # gammas: "mixing weights"
                    numerator += gammas[u][t][k] * ( (counts[u][t] + a[k]) * b[k] / (b[k] + 1) )
                    denominator += gammas[u][t][k]  # "number of observations"
            average_counts.append(numerator / denominator)

        average_log_counts = []
        for k in range(len(a)):
            numerator = 0
            denominator = 0
            for u in range(len(counts)):  # for each user
                for t in range(T):
                    if counts[u][t] == 0:
                        numerator = 0
                    else:
                        numerator += gammas[u][t][k] * ( polygamma(0, counts[u][t]+a[k]) + log(b[k]/(b[k]+1)) )
                    denominator += gammas[u][t][k]
            average_log_counts.append(numerator / denominator)


        # print('\nAverage counts')
        # pprint(average_counts)  # average counts (no of items) bought for each state,
        #                         # averaged over all users and the probability the user is in a state at a time t for all classes

        # print('\nAverage log counts')
        # pprint(average_log_counts)  # average counts (no of items) for each state

        # TODO: change "counts" variable names to mean lambda (hidden variable) as in Minka 2002

        # maximisation
        for k in range(len(a)):
            # print('a[{}]: {}, b[{}]: {}'.format(k, a[k], k, b[k]))
            b[k] = average_counts[k] / a[k]  # Minka 2002 (3)
            if average_counts != 0:
                a[k] = 0.5 / (log(average_counts[k]) - average_log_counts[k])

            # starting point
            if average_counts != 0:
                a[k] = 0.5 / (log(average_counts[k]) - average_log_counts[k])

            while True:  # NOTE: *should* converge in four iterations
                a_old = a[k]

                # print(average_counts[k], a[k])
                if average_counts[k] > 0 and a[k] > 0:
                    # f = sum(sum(gammas[u][t][k] * (polygamma(0, a[k] + average_counts[k]) - polygamma(0, a[k]) - log(average_counts[k]/a[k] + 1)) for t in range(T)) for u in range(no_of_users))
                    # df = sum(sum(gammas[u][t][k] * (polygamma(1, a[k] + average_counts[k]) - polygamma(1, a[k]) - 1/(average_counts[k] + a[k]) + 1/a[k]) for t in range(t)) for u in range(no_of_users))
                    a_new_inv = 1/a[k] + (average_log_counts[k] - log(average_counts[k]) + log(a[k]) - polygamma(0, a[k])) / (a[k]**2 * (1/a[k] - polygamma(1, a[k])))
                    a[k] = 1/a_new_inv

                    # a[k] = a[k] - f/df

                assert(a[k] >= 0)
                # print('a_old: {}, a_updated: {}'.format(a_old, a[k]))

                if a[k] - a_old < self.EPSILON:
                    break

            # print('a[{}]: {}, b[{}]: {}'.format(k, a[k], k, b[k]))
        return (a, b)

    def item_rank(self, u, a, b, alphas, theta, transition_prob):
        """
        Returns a list of relevant items for a user in order of relevance.
        """
        no_of_states = len(alphas[u][0])

        # calculate distribution over the states for the user at time t+1
        p_t_plus_1 = []
        for k in range(no_of_states):
            total = 0
            for l in range(no_of_states):
                total += alphas[u][-1][k] * transition_prob[l][k]
            p_t_plus_1.append(total)
#         # print('\np_t_plus_u1')
        # pprint(p_t_plus_1)

        item_rank = defaultdict(float)
        for i in range(len(theta)):  # for each item
            item_rank[i] = -sum(p_t_plus_1[k] * (1 + b[k] * theta[i][k])**(-a[k]) for k in range(no_of_states))

#         # print(item_rank)
        items = sorted(item_rank, key=item_rank.__getitem__, reverse=True)

        return items

hmm = HMM()

def forward_backward_helper(params):
    a, b, theta, pi, transition_prob, seq, u = params
    # u: user index
    # print('Forward {}'.format(u))
    alphas_u, scaling_factors_u = hmm.forward(a, b, theta, pi, transition_prob, seq)
    # print('Backward {}'.format(u))
    betas_u = hmm.backward(a, b, theta, transition_prob, seq, scaling_factors_u)

    return u, alphas_u, scaling_factors_u, betas_u

def calculate_gamma(params):
    u, alphas, betas, T, no_of_states = params
    # print('Gamma {}'.format(u))
    gamma = alphas[u] * betas[u]
    return (u, gamma)

def calculate_xi(params):
    u, a, b, theta, observation_seqs, alphas, betas, transition_prob, scaling_factors, T, no_of_states = params
    # print('Xi {}'.format(u))

    xi = np.zeros(shape=(T-1, no_of_states, no_of_states))
    for t in range(T-1):
        emissions = np.array([hmm.emission_prob(a, b, theta, j, observation_seqs[u][t+1]) for j in range(no_of_states)])
        for i in range(no_of_states):
            xi[t][i] = np.multiply(np.multiply(np.multiply(alphas[u][t][i], transition_prob[i]), emissions), betas[u][t+1])
        if scaling_factors[u][t+2] != 0:
            xi[t] = xi[t] / scaling_factors[u][t+2]
    return (u, xi)

def calculate_theta(params):
    i, k, no_of_items, no_of_states, no_of_users, gammas, T, observation_seqs, obsv_counts, total_counts, ALPHA_I = params
    gammas_ = np.swapaxes(gammas, 0, 2)  # gammas[k][t][u]
    numerator = 0
    denominator = 0
    for t in range(T):
        # gammas for all users (dot) counts for all users
        numerator += (gammas_[k][t] * obsv_counts[t][i]).sum()
        denominator += (gammas_[k][t] * total_counts[t]).sum()
    numerator += ALPHA_I/no_of_items - 1
    denominator += ALPHA_I - no_of_items

    assert(numerator / denominator >= 0)
    theta_ik = numerator/denominator

    return (i, k, theta_ik)

hmm = HMM()


"""
Let's say we have 15 items.
Imagine:
index 0 to 7: "tech"
index 8 to 12: "fashion"
index 13 to 14: "power tools"

Assume we have 5 users
observation_seq[u][t]
-> u: the u-th user
-> t: time step, assume it's a day?
    each observation:
    [0, 3, 2, 7, ...]
    -> entries: indices of the items

"""

if __name__ == '__main__':
    # Test case for speed tests
    # pi = [1/5, 3/5, 1/5]  # distribution of the starting state of the users
    # A = [  # transition probability
    #     [3/5, 1/5, 1/5],
    #     [2/5, 1/5, 2/5],
    #     [2/5, 1/5, 2/5],
    # ]
    observation_seqs = [
        [[0, 3, 2, 7], [1, 4], [2, 5, 6], [0, 0, 2, 3]],  # user 1, only tech
        [[0, 1, 2, 8, 9, 2], [3, 1, 4, 1, 5, 9], [8, 10, 12, 7], [1, 2, 1, 1]], # user 2, mixture of tech and fashion. Heavy user.
        [[0, 1], [2], [3], []],  # user 3, light user, mainly tech
        [[13], [14], [], [0, 1]],  # user 4, power tools, also browsed tech
        [[8, 9, 10], [9, 10, 11], [10, 11, 12], [8, 8, 9]]  # only fashion
        # [[i for i in range(6)]*3 for i in range(4)]
        # [[1], [1], [1]],
        # [[2], [1, 7], [1]]
    ]

    # a = [1/1000]*3
    # b = [1/1000]*3
    # theta = [
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15],
    #             [1/15, 1/15, 1/15]
    #         ]

    K = 3
    I = 15

    pi = [random.uniform(0, 1) for k in range(K)]
    total = sum(pi)
    pi = [entry/total for entry in pi]

    A = []
    for i in range(K):
        A_i = [random.uniform(0, 1) for k in range(K)]
        total = sum(pi)
        A.append([entry/total for entry in A_i])

    a = [random.uniform(0, 10) for k in range(K)]
    b = [random.uniform(0, 10) for k in range(K)]

    theta_per_state = []
    for k in range(K):
        theta_i_K = [random.uniform(0, 1) for i in range(I)]
        total = sum(theta_i_K)
        theta_per_state.append([entry/total for entry in theta_i_K])

    theta = np.zeros(shape=(I, K))
    for i in range(I):
        for k in range(K):
            theta[i][k] = theta_per_state[k][i]

    # NOTE: IIRC prior weight has to be > number of items or number of states
    a, b, theta, pi, A, alphas = hmm.baum_welch(a, b, theta, pi, A, observation_seqs, prior_weight_items=len(theta)+1,
                                        prior_weight_states=(len(theta[0])-1)*len(observation_seqs)+1)
                                        # prior_weight_states=100*5)
                                        # (no of states - 1) * no of users + 1
                                        # this is the lowest possible prior before running into
                                        # probabilities >1 and <0 (sums still to 1)

    # print(a, b, theta, pi, A, alphas)
    # pprint(A)
    # pprint(theta)

    pprint(alphas)

    print('Recommendations')
    for u in range(len(observation_seqs)):
        print('user {}: {}'.format(u, hmm.item_rank(u, a, b, alphas, theta, A)))
    # print(hmm.item_rank(1, a, b, alphas, theta, A))
    # print(hmm.item_rank(2, a, b, alphas, theta, A))
    # print(hmm.item_rank(3, a, b, alphas, theta, A))
    # print(hmm.item_rank(4, a, b, alphas, theta, A))
