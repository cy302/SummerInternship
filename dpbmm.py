import numpy as np
import scipy as sp
import copy
from scipy.stats import beta
from scipy.stats import norm
import random


def dirichletrnd(alpha):
    '''
    generating dirchlet distributed random variable, generated sample size depends on the input parameter dimension
    alpha: the parameter vector for the dirichlet distribution Dir(alpha)
    '''
    s = np.random.gamma(shape=alpha, scale=1, size=len(alpha))
    x = s/np.sum(s)
    return np.array(x)


def calculate_m_s(s, c=None):
    '''
    compute the number of candidates in each cluster given the cluster assignment vector s
    s: the cluster assignment vector
    c: the candidate to be ommited
    '''
    k = len(np.unique(s))
    m_s = np.zeros(k)
    for i in range(k):
        m_s[i] = len(np.where(s == i)[0])
    if c is not None:
        m_s[s[c]] -= 1
    return np.array(m_s), k


def rearrange_s(s):
    '''
    function to make sure that the cluster assignment is at most the number of unique elements in s
    s: cluster assignment vector
    '''
    k = len(np.unique(s))
    if np.max(s) > (k-1):
        ind = np.where(s > (k-1))[0]
        s[ind] = k-1
    return np.array(s), k


def exp_trans(alpha_val, beta_val):
    L_alpha_val = np.exp(np.abs(alpha_val))
    L_beta_val = np.exp(np.abs(beta_val))
    return L_alpha_val, L_beta_val


class params(object):
    def __init__(self, mu_a, mu_b, sigma2_a, sigma2_b, k, m_s, s, tau, pi=None):
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.sigma2_a = sigma2_a
        self.sigma2_b = sigma2_b
        self.k = k
        self.m_s = m_s
        self.s = s
        self.tau = tau
        self.pi = pi

    def param_val(self, C, G):
        alpha_val = np.random.normal(loc=self.mu_a, scale=self.sigma2_a, size=(G, C))
        beta_val = np.random.normal(loc=self.mu_b, scale=self.sigma2_b, size=(G, C))
        self.alpha_val = alpha_val
        self.beta_val = beta_val


def dpbmm(data, num_iter, param=None):
    # Beta value Dirichlet process mixture model, with no gap algorithm
    s_data = np.shape(data)
    G = copy.copy(s_data[0]) ## number of genes
    C = copy.copy(s_data[1]) ## number of samples
    count_sum = 100

    # parameters for G0, Gamma(a, b)
    a = 6
    b = 5

    if param is None:
        param = params(mu_a=3.2, mu_b=2.2, sigma2_a=5, sigma2_b=2, k=C, m_s=np.ones(C), s=np.arange(C), tau=5)
        param.param_val(C, G)

    for i in range(num_iter):
        new_param = copy.deepcopy(param)
        ## Gibbs sampling
        ## step 1: resample s

        s = copy.copy(new_param.s)
        k = copy.copy(new_param.k)
        m_s = copy.copy(new_param.m_s)
        alpha_val = copy.copy(new_param.alpha_val)
        beta_val = copy.copy(new_param.beta_val)
        tau = copy.copy(new_param.tau)
        mu_a = copy.copy(new_param.mu_a)
        mu_b = copy.copy(new_param.mu_b)
        sigma2a = copy.copy(new_param.sigma2_a)
        sigma2b = copy.copy(new_param.sigma2_b)

        for j in range(C):
            s, k = rearrange_s(s)
            m_s, k = calculate_m_s(s)
            if m_s[s[j]] == 1:
                u = np.random.rand()
                if u < (k-1)/k:
                    continue
                ind = np.where(s == (k-1))[0]
                tmp = copy.copy(s[j])
                s[ind] = copy.copy(tmp)
                s[j] = k-1

                tmp_alpha_val = copy.copy(alpha_val[:, tmp])
                alpha_val[:, tmp] = copy.copy(alpha_val[:, k-1])
                alpha_val[:, k-1] = copy.copy(tmp_alpha_val)
                tmp_beta_val = copy.copy(beta_val[:, tmp])
                beta_val[:, tmp] = copy.copy(beta_val[:, k-1])
                beta_val[:, k-1] = copy.copy(tmp_beta_val)

                m_s, k = calculate_m_s(s, c=j)
                p_x = np.zeros(k)

                for l in range(k):
                    L_alpha_val, L_beta_val = exp_trans(alpha_val[:, l], beta_val[:, l])
                    p_x[l] = np.prod(beta.pdf(data[:, j], L_alpha_val, L_beta_val))

                w = m_s/(tau+j) * p_x

                if k < C:
                    L_alpha_val, L_beta_val = exp_trans(alpha_val[:, k], beta_val[:, k])
                    # w[k+1] = tau/(tau+j-1) * np.prod(beta(data[:, j], L_alpha_val, L_beta_val))
                    temp1 = np.prod(beta.pdf(data[:, j], L_alpha_val, L_beta_val))
                    p_x = np.append(p_x, temp1)
                    w = np.append(w, tau/(tau+j)*temp1)
                    population = np.arange(k+1)

                else:
                    population = np.arange(k)
                w = w/np.sum(w)
                s[j] = np.random.choice(population, size=1, p=w)
            else:
                # del m_s
                # del p_x
                # del w
                m_s, k = calculate_m_s(s, c=j)
                p_x = np.zeros(k)

                for l in range(k):
                    L_alpha_val, L_beta_val = exp_trans(alpha_val[:, l], beta_val[:, l])
                    p_x[l] = np.prod(beta.pdf(data[:, j], L_alpha_val, L_beta_val))

                w = m_s/(tau+j) * p_x

                if k < C:
                    L_alpha_val, L_beta_val = exp_trans(alpha_val[:, k], beta_val[:, k])
                    p_x = np.append(p_x, np.prod(beta.pdf(data[:, j], L_alpha_val, L_beta_val)))
                    w = np.append(w, p_x[k]*tau/(tau+j))
                    population = np.arange(k+1)
                else:
                    population = np.arange(k)
                w = w / np.sum(w)
                s[j] = np.random.choice(population, size=1, p=w)
        s, k = rearrange_s(s)
        m_s, k = calculate_m_s(s, c=0)

        # new_alpha_val = np.zeros([G, k])
        # new_beta_val = np.zeros([G, k])
        for g in range(G):

            for j in np.arange(k, C):
                alpha_val[g, j] = np.random.normal(loc=mu_a, scale=sigma2a, size=1)
                beta_val[g, j] = np.random.normal(loc=mu_b, scale=sigma2b, size=1)

            V_a = sigma2a * np.ones([G, G])
            V_b = sigma2b * np.ones([G, G])

            for j in range(k):
                new_alpha_val = np.random.multivariate_normal(mean=alpha_val[:, j], cov=V_a)
                new_beta_val = np.random.multivariate_normal(mean=beta_val[:, j], cov=V_b)

                count = 0
                accept = 0
                c_1 = 0
                ind = np.where(s == j)

                while accept == 0:
                    for count_ind in range(count_sum):
                        p_xi = np.zeros(len(ind))
                        p_xi_t = np.zeros(len(ind))
                        new_alpha_val[g] = np.random.normal(loc=alpha_val[g, j], scale=sigma2a, size=1)
                        # new_beta_val[g, i] = np.random.normal(loc=beta_val[g, i], scale=sigma2b, size=1)
                        # ind = np.where(s == j)
                        L_alpha_val, L_beta_val = exp_trans(new_alpha_val, beta_val[:, j])
                        L_alpha_val_t, L_beta_val_t = exp_trans(alpha_val[:, j], beta_val[:, j])

                        for m in range(len(ind)):
                            p_xi[m] = np.sum(np.log(beta.pdf(data[:, ind[m]], L_alpha_val, L_beta_val)))
                            p_xi_t[m] = np.sum(np.log(beta.pdf(data[:, ind[m]], L_alpha_val_t, L_beta_val_t)))

                        sum_p_xi = np.sum(p_xi)
                        sum_p_xi_t = np.sum(p_xi_t)

                        fx = np.dot(norm.pdf(alpha_val[:, j], np.zeros(G), sigma2a),
                                    norm.pdf(beta_val[:, j], np.zeros(G), sigma2b))
                        fx_t = np.dot(norm.pdf(new_alpha_val, np.zeros(G), sigma2a),
                                      norm.pdf(beta_val[:, j], np.zeros(G), sigma2b))

                        # Metropolis Hastings sampling
                        if sum_p_xi == sum_p_xi_t and sum_p_xi == -np.inf:
                            if fx == fx_t and fx == 0:
                                tmp = 0
                            else:
                                tmp = np.exp(np.log(fx) - np.log(fx_t))
                        else:
                            if fx == fx_t and fx == 0:
                                tmp = np.exp(sum_p_xi - sum_p_xi_t)
                            else:
                                if fx == 0 and sum_p_xi_t == -np.inf:
                                    tmp = np.exp(sum_p_xi - np.log(fx_t))
                                elif fx_t == 0 and sum_p_xi == -np.inf:
                                    tmp = np.exp(fx - sum_p_xi_t)
                                else:
                                    tmp = np.exp(np.log(fx) + sum_p_xi - np.log(fx_t) - sum_p_xi_t)

                        u = np.random.rand()
                        # count_ind += 1
                        if u < tmp:
                            alpha_val[g, j] = new_alpha_val[g]
                            count += 1

                    c_1 += 1

                    if count >= 3 or c_1 > 4:
                        accept = 1
                    else:
                        sigma2a *= 0.98
                        if sigma2a < 0.01:
                            sigma2a = 0.01
                        if count > 20:
                            sigma2a /= 0.98

                accept = 0
                count = 0
                c_2 = 0
                while accept == 0:
                    for count_ind in range(count_sum):
                        new_beta_val[g] = np.random.normal(loc=beta_val[g, j], scale=sigma2b, size=1)

                        L_alpha_val, L_beta_val = exp_trans(alpha_val[:, j], new_beta_val)
                        L_alpha_val_t, L_beta_val_t = exp_trans(alpha_val[:, j], beta_val[:, j])

                        p_xi = np.zeros(len(ind))
                        p_xi_t = np.zeros(len(ind))
                        for m in range(len(ind)):
                            p_xi[m] = np.sum(np.log(beta.pdf(data[:, ind[m]], L_alpha_val, L_beta_val)))
                            p_xi_t = np.sum(np.log(beta.pdf(data[:, ind[m]], L_alpha_val_t, L_beta_val_t)))

                        sum_p_xi = np.sum(p_xi)
                        sum_p_xi_t = np.sum(p_xi_t)

                        fx = np.dot(norm.pdf(alpha_val[:, j], np.zeros(G), sigma2a),
                                    norm.pdf(beta_val[:, j], np.zeros(G), sigma2b))
                        fx_t = np.dot(norm.pdf(new_alpha_val, np.zeros(G), sigma2a),
                                      norm.pdf(new_beta_val, np.zeros(G), sigma2b))

                        # Metropolis Hastings sampling
                        if sum_p_xi == sum_p_xi_t and sum_p_xi == -np.inf:
                            if fx == fx_t and fx == 0:
                                tmp = 0
                            else:
                                tmp = np.exp(np.log(fx) - np.log(fx_t))
                        else:
                            if fx == fx_t and fx == 0:
                                tmp = np.exp(sum_p_xi - sum_p_xi_t)
                            else:
                                if fx == 0 and sum_p_xi_t == -np.inf:
                                    tmp = np.exp(sum_p_xi - np.log(fx_t))
                                elif fx_t == 0 and sum_p_xi == -np.inf:
                                    tmp = np.exp(fx - sum_p_xi_t)
                                else:
                                    tmp = np.exp(np.log(fx) + sum_p_xi - np.log(fx_t) - sum_p_xi_t)

                        u = np.random.rand()
                        if u < tmp:
                            beta_val[g, j] = new_beta_val[g]
                            count += 1

                    c_2 += 1

                    if count >= 3 or c_2 > 4:
                        accept = 1
                    else:
                        sigma2b *= 0.98
                        if sigma2b < 0.01:
                            sigma2b = 0.01
                        if count > 20:
                            sigma2b /= sigma2b
        
        # step 3: resampling mixture weights pi
        if k == 1:
            continue
        m_s, k = calculate_m_s(s)
        pi = dirichletrnd(m_s+tau/k)
        
        # step 4: resampling concentration parameter tau
        r = np.random.beta(a=tau+1, b=C, size=1)
        eta_r = 1/(C*(b-np.log(r))/(a+k-1)+1)
        tmp = np.random.rand()
        if tmp < eta_r:
            tau_new = np.random.gamma(shape=a+k, scale=b-np.log(r), size=1)
        else:
            tau_new = np.random.gamma(shape=a+k-1, scale=b-np.log(r), size=1)
        tau = tau_new
        
        # step 5: update parameters for the usage in the new iteration
        new_param.s = s
        new_param.k = k
        new_param.m_s = m_s
        new_param.alpha_val = alpha_val
        new_param.beta_val = beta_val
        new_param.tau = tau
        new_param.pi = pi
        new_param.sigma2_a = sigma2a
        new_param.sigma2_b = sigma2b

        param = copy.deepcopy(new_param)

        file.write(str(i) + " iterations done")

    print(param.s)
    return param

np.random.seed(1998)


def gen_simulated_data(N, pi, u, v):
    D = np.zeros([N, np.shape(u)[1]])
    a = np.arange(N)
    l = np.random.choice(a, size=N, replace=False)
    lengths = np.round(np.cumsum(pi) * N)
    lengths = np.insert(lengths, 0, 0)
    cl = [[l[np.arange(lengths[i-1], lengths[i], dtype=np.int)]] for i in range(1, len(lengths))]

    for i in range(len(pi)):
        for j in range(np.shape(u)[1]):
            D[cl[i][0], j] = np.random.beta(u[i, j], v[i, j], len(cl[i][0]))
    return D


pi = np.array([0.4, 0.6])
u = np.array([[3, 5], [4, 9]])
v = np.array([[1, 6], [4, 8]])
D = gen_simulated_data(100, pi, u, v)
p = dpbmm(data=np.transpose(D), num_iter=2000)
