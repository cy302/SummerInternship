import numpy as np
import scipy as sp
import copy
from scipy.stats import beta
from scipy.stats import norm


def dirichletrnd(alpha):
    '''
    generating dirchlet distributed random variable, generated sample size depends on the input parameter dimension
    alpha: the parameter vector for the dirichlet distribution Dir(alpha)
    '''
    s = np.random.gamma(shape=alpha, scale=1, size=len(alpha))
    x = s/np.sum(s)
    return x


def calculate_m_s(c, s):
    '''
    compute the number of candidates in each cluster given the cluster assignment vector s
    s: the cluster assignment vector
    c: the candidate to be ommited
    '''
    s_unique = np.unique(s)
    k = len(s_unique)
    m_s = np.zeros(k)
    for i in range(k):
        m_s[i] = np.sum(s == i)
    if c != 0:
        m_s[s[c]] -= 1
    return m_s, k


def rearrange_s(s):
    '''
    function to make sure that the cluster assignment is at most the number of unique elements in s
    s: cluster assignment vector
    '''
    s_unique = np.unique(s)
    k = len(s_unique)
    if np.max(s) > k:
        ind = np.where(s>k)
        s[ind] = k
    return s, k


def exp_trans(alpha_val, beta_val):
    L_alpha_val = np.diag(sp.linalg.expm(np.diag(abs(alpha_val))))
    L_beta_val = np.diag(sp.linalg.expm(np.diag(abs(beta_val))))
    return L_alpha_val, L_beta_val


class params(object):
    def __init__(self, mu_a, mu_b, sigma2_a, sigma2_b, k, m_s, s, tau):
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.sigma2_a = sigma2_a
        self.sigma2_b = sigma2_b
        self.k = k
        self.m_s = m_s
        self.s = s
        self.tau = tau

    def param_val(self, C, G):
        alpha_val = np.random.normal(loc=self.mu_a, scale=self.sigma2_a, size=(G, C))
        beta_val = np.random.normal(loc=self.mu_b, scale=self.sigma2_b, size=(G, C))
        self.alpha_val = alpha_val
        self.beta_val = beta_val


def dpbmm(data, num_iter, param=None, debug=False):
    # Beta value Dirichlet process mixture model, with no gap algorithm
    s_data = np.shape(data)
    G = s_data[0]
    C = s_data[1]
    count_sum = 100

    # parameters for G0, Gamma(a, b)
    a = 6
    b = 5

    if param is None:
        param = params(mu_a=3.2, mu_b=2.2, sigma2_a=5, sigma2_b=2, k=C, m_s=np.ones(C), s=list(range(C)), tau=5)
        param.param_val(C, G)

    for i in range(num_iter):
        new_param = copy.deepcopy(param)
        if debug:
            print("resampling s")

        s = new_param.s
        k = new_param.k
        m_s = new_param.m_s
        alpha_val = new_param.alpha_val
        beta_val = new_param.beta_val
        tau = new_param.tau
        mu_a = new_param.mu_a
        mu_b = new_param.mu_b
        sigma2a = new_param.sigma2_a
        sigma2b = new_param.sigma2_b

        for j in range(C):
            s, k = rearrange_s(s)
            m_s, k = calculate_m_s(0, s)
            if m_s[s[j]] == 1:
                u = np.random.rand()
                if u < (k-1)/k:
                    continue
                ind = np.where(s == k)
                tmp = s[j].copy()
                s[ind] = tmp.copy()
                s[j] = k.copy()

                tmp_alpha_val = alpha_val[:, tmp].copy()
                alpha_val[:, tmp] = alpha_val[:, k].copy()
                alpha_val[:, k] = tmp_alpha_val.copy()
                tmp_beta_val = beta_val[:, tmp].copy()
                beta_val[:, tmp] = beta_val[:, k].copy()
                beta_val[:, k] = tmp_beta_val.copy()

                m_s, k = calculate_m_s(j, s)
                p_x = []
                for l in range(k):
                    L_alpha_val, L_beta_val = exp_trans(alpha_val[:, l], beta_val[:, l])
                    p_x = np.append(p_x, np.prod(beta(data[:, j], L_alpha_val, L_beta_val)))

                w = m_s/(tau+j-1) * p_x
                if k < C:
                    L_alpha_val, L_beta_val = exp_trans(alpha_val[:, k+1], beta_val[:, k+1])
                    # w[k+1] = tau/(tau+j-1) * np.prod(beta(data[:, j], L_alpha_val, L_beta_val))
                    temp1 = np.prod(beta(data[:, j], L_alpha_val, L_beta_val))
                    p_x = np.append(p_x, temp1)
                    w = np.append(w, tau/(tau+j-1)*temp1)
                    population = list(range(k+1))
                    if debug:
                        nan_p_x = np.where(np.isnan(p_x) == 1)
                        if len(nan_p_x) > 0:
                            print("likelihood not a number {} p_x {}".format(nan_p_x, p_x))
                        nan_w = np.where(np.isnan(w) == 1)
                        if len(nan_w) > 0:
                            print("w is not a number {} w {}".format(nan_w, w))

                else:
                    population = list(range(k))
                s[j] = np.random.choice(population, size=1, p=p_x)
            else:
                m_s, k = calculate_m_s(j, s)
                p_x = []
                for l in range(k):
                    L_alpha_val, L_beta_val = exp_trans(alpha_val[:, l], beta_val[:, l])
                    p_x = np.append(p_x, beta(data[:, j], L_alpha_val, L_beta_val))
                if len(m_s) != k:
                    print(len(m_s))
                w = m_s/(tau+j-1) * p_x
                if k < C:
                    L_alpha_val, L_beta_val = exp_trans(alpha_val[:, k+1], beta_val[:, k+1])
                    temp1 = np.prod(beta(data[:, j], L_alpha_val, L_beta_val))
                    temp2 = tau/(tau+j-1) * temp1
                    w = np.append(w, temp2)
                    p_x = np.append(p_x, temp1)
                    population = list(range(k+1))
                else:
                    population = list(range(k))
                if debug:
                    nan_p_x = np.where(np.isnan(p_x) == 1)
                    if len(nan_p_x) > 0:
                        print("likelihood not a number {} p_x {}".format(nan_p_x, p_x))
                    nan_w = np.where(np.isnan(w) == 1)
                    if len(nan_w) > 0:
                        print("w is not a number {} w {}".format(nan_w, w))
                s[j] = np.random.choice(population, size=1, p=p_x)
        s, k = rearrange_s(s)
        m_s, k = calculate_m_s(0, s)

        new_alpha_val = np.zeros([G, k])
        new_beta_val = np.zeros([G, k])
        for g in range(G):
            if debug:
                print("resampling Phi_E")

            for j in range(k+1, C+1):
                alpha_val[g, j] = np.random.normal(loc=mu_a, scale=sigma2a, size=1)
                beta_val[g, j] = np.random.normal(loc=mu_b, scale=sigma2b, size=1)

            if debug:
                print("resampling Phi_F")

            for j in range(k):
                V_a = sigma2a * np.ones([G, G, C])
                V_b = sigma2b * np.ones([G, G, C])
                new_alpha_val[:, j] = np.random.multivariate_normal(mean=alpha_val[:, j], cov=V_a[:, :, j])
                new_beta_val[:, j] = np.random.multivariate_normal(mean=beta_val[:, j], cov=V_b[:, :, j])

                count = 0
                accept = 0
                while accept == 0:
                    for count_ind in range(count_sum):
                        p_xi = []
                        p_xi_t = []
                        new_alpha_val[g, j] = np.random.normal(loc=alpha_val[g, j], scale=sigma2a, size=1)
                        # new_beta_val[g, i] = np.random.normal(loc=beta_val[g, i], scale=sigma2b, size=1)
                        ind = np.where(s == j)
                        L_alpha_val, L_beta_val = exp_trans(new_alpha_val[:, i], beta_val[:, i])
                        L_alpha_val_t, L_beta_val_t = exp_trans(alpha_val[:, i], beta_val[:, i])

                        for m in range(len(ind)):
                            p_xi = np.append(p_xi, np.sum(np.log(beta(data[:, ind[m]], L_alpha_val, L_beta_val))))
                            p_xi_t = np.append(p_xi_t, np.sum(np.log(beta(data[:, ind[m], L_alpha_val_t,
                                                                          L_beta_val_t]))))
                        sum_p_xi = np.sum(p_xi)
                        sum_p_xi_t = np.sum(p_xi_t)

                        fx = np.matmul(np.transpose(norm(alpha_val[:, j], np.zeros(G), np.diag(V_a[:, :, j]))),
                                       norm(beta_val[:, j], np.zeros(G), np.diag(V_b[:, :, j])))
                        fx_t = np.matmul(np.transpose(norm(new_alpha_val[:, j], np.zeros(G), np.diag(V_a[:, :, j]))),
                                         norm(beta_val[:, j], np.zeros(G), np.diag(V_b[:, :, j])))

                        ## Metropolis Hastings sampling
                        tmp = np.log(fx) + sum_p_xi - np.log(fx_t) - sum_p_xi_t
                        tmp = np.exp(tmp)
                        u = np.random.rand()
                        # count_ind += 1
                        if u < tmp:
                            alpha_val[g, i] = new_alpha_val[g, i].copy()
                            count += 1

                    if count >= 3:
                        accept = 1
                    else:
                        sigma2a *= 0.98
                        if sigma2a < 0.01:
                            sigma2a = 1
                        if count > 20:
                            sigma2a /= 0.98

                accept = 0
                count = 0
                while accept == 0:
                    for count_ind in range(count_sum):
                        new_beta_val[g, j] = norm(loc=beta_val[g, i], scale=sigma2b, size=1)
                        ind = np.where(s == j)

                        L_alpha_val, L_beta_val = exp_trans(alpha_val[:, j], new_beta_val[:, j])
                        L_alpha_val_t, L_beta_val_t = exp_trans(alpha_val[:, j], beta_val[:, j])

                        p_xi = []
                        p_xi_t = []
                        for m in range(len(ind)):
                            p_xi = np.append(p_xi, np.sum(np.log(beta(data[:, ind[m]], L_alpha_val, L_beta_val))))
                            p_xi_t = np.append(p_xi_t, np.sum(np.log(beta(data[:, ind[m]], L_alpha_val_t,
                                                                        L_beta_val_t))))
                        p_x = np.sum(p_xi)
                        p_x_t = np.sum(p_xi_t)

                        fx = np.matmul(np.transpose(norm(alpha_val[:, j], np.zeros(G), np.diag(V_a[:, :, j]))),
                                       norm(beta_val[:, j], np.zeros(G), np.diag(V_b[:, :, j])))
                        fx_t = np.matmul(np.transpose(norm(new_alpha_val[:, j], np.zeros(G), np.diag(V_a[:, :, j]))),
                                         norm(beta_val[:, j], np.zeros(G), np.diag(V_b[:, :, j])))

                        tmp = np.log(fx) + p_x - np.log(fx_t) - p_x_t
                        tmp = np.exp(tmp)
                        u = np.random.rand()
                        if u < tmp:
                            beta_val[g, i] = new_beta_val[g, i]
                            count += 1

                    if count >= 3:
                        accept = 1
                    else:
                        sigma2b *= 0.98
                        if sigma2b < 0.01:
                            sigma2b = 0.01
                        if count > 20:
                            sigma2b /= sigma2b
        
        ## step 3: resampling mixture weights pi
        if debug:
            print("resampling pi")
        if k == 1:
            continue
        m_s, k = calculate_m_s(0, s)
        pi = dirichletrnd(m_s+tau/k)
        
        ## step 4: resampling concentration parameter tau
        if debug:
            print("resampling tau")
        r = np.random.beta(a=tau+1, b=C, size=1)
        eta_r = 1/(C*(b-np.log(r))/(a+k-1)+1)
        tmp = np.random.rand()
        if tmp < eta_r:
            tau_new = np.random.gamma(shape=a+k, scale=b-np.log(r), size=1)
        else:
            tau_new = np.random.gamma(shape=a+k-1, scale=b-np.log(r), size=1)
        tau = tau_new.copy()
        
        ## step 5: update parameters for the usage in the new iteration
        if debug:
            print("parameters update")
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

    return param.s
