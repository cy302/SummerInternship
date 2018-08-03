library("MASS")
library(readxl)
library(reshape2)
library(ggplot2)
library(grid)
library(ggdendro)
library(betareg)

setwd("~/Desktop/MPhilComputationalBiology/Internship/")

rdirichlet <- function(alpha){
  y <- rgamma(length(alpha), shape=alpha)
  y <- y/sum(y)
  return(y)
}

exp_trans <- function(alpha, beta){
  return(list("alpha"=exp(abs(alpha)), "beta"=exp(abs(beta))))
}

rearrange_s <- function(s){
  k <- length(unique(s))
  s[which(s>k)] <- k
  return(s)
}

calc_m_s <- function(s, c=0){
  k <- length(unique(s))
  m_s <- rep(0, k)
  for (i in 1:k){
    m_s[i] <- length(which(s==i))
  }
  if (c != 0){
    m_s[s[c]] <- m_s[s[c]] - 1
  }
  return(m_s)
}

bdpmm <- function(dat, num_its, params=NA){
  N <- dim(dat)[1]; G <- dim(dat)[2];
  ## N: number of samples, L: number of genes
  
  a <- 6; b <- 5;
  count_num <- 100
  
  if (is.na(params)){
    params <- list()
    params$mu_a <- 3.2; params$mu_b <- 2.2; 
    sigma2a <- 5; sigma2b <- 2;
    params$alpha <- matrix(rnorm(N*G, mean=params$mu_a, sd=sigma2a), G, N)
    params$beta <- matrix(rnorm(N*G, mean=params$mu_b, sd=sigma2b), G, N)
    params$k <- N; ## initial number of clusters
    params$m_s <- rep(1, N) ## number of samples in each cluster
    params$s <- 1:N ## cluster assignment variable
    params$tau <- 5 ## concentration parameter for DP
    params$sigma2a <- sigma2a; params$sigma2b <- sigma2b;
  }
  
  for (it in 1:num_its){
    s <- params$s; k <- params$k; m_s <- params$m_s; 
    alpha <- params$alpha; beta <- params$beta;
    tau <- params$tau; mu_a <- params$mu_a; mu_b <- params$mu_b;
    sigma2a <- params$sigma2a; sigma2b <- params$sigma2b;
    
    for (i in 1:N){
      s <- rearrange_s(s)
      m_s <- calc_m_s(s)
      k <- max(s)
      if (m_s[s[i]]==1){
        u <- runif(1)
        if (u < (k-1)/k){
          next
        }
        ind <- which(s==k)
        temp <- s[i]; s[ind] <- temp; s[i] <- k;
        
        alpha <- alpha[, replace(1:N, c(temp, k), (1:N)[c(k, temp)])]
        beta <- beta[, replace(1:N, c(temp, k), (1:N)[c(k, temp)])]
        
        m_s <- calc_m_s(c=i, s)
        L <- exp_trans(alpha, beta)
        L_alpha <- L$alpha; L_beta <- L$beta;
        d <- dat[i, ]
        p_xi <- sapply(1:k, function(j) prod(dbeta(x=d, L_alpha[, j], 
                                                   L_beta[, j])))
        w <- m_s/(tau+i-1)*p_xi;
        if (k < N){
          L <- exp_trans(alpha[, k+1], beta[, k+1])
          L_alpha <- L$alpha; L_beta <- L$beta;
          # w <- c(w, tau/(tau+i-1)*prod(dbeta(d, L_alpha, L_beta)))
          p_xi <- c(p_xi, prod(dbeta(d, L_alpha, L_beta)))
          w <- c(w, tau/(tau+i-1)*p_xi[k+1])
          population <- 1:(k+1)
        }
        else{
          population <- 1:k
        }
        s[i] <- sample(population, 1, prob=w)
      }
      else{
        m_s <- calc_m_s(c=i, s)
        L <- exp_trans(alpha, beta)
        L_alpha <- L$alpha; L_beta <- L$beta;
        p_xi <- sapply(1:k, function(j) prod(dbeta(x=d, L_alpha[, j], 
                                                   L_beta[, j])))
        w <- m_s/(tau+i-1)*p_xi
        
        if (k < N){
          L <- exp_trans(alpha[, k+1], beta[, k+1])
          L_alpha <- L$alpha; L_beta <- L$beta;
          # w <- c(w, tau/(tau+i-1)*prod(dbeta(d, L_alpha, L_beta)))
          p_xi <- c(p_xi, prod(dbeta(d, L_alpha, L_beta)))
          w <- c(w, tau/(tau+i-1)*p_xi[k+1])
          population <- 1:(k+1)
        }
        else{
          population <- 1:k
        }
        s[i] <- sample(population, 1, prob=w)
      }
    }
    s <-rearrange_s(s)
    m_s <- calc_m_s(s)
    k <- length(m_s)
    
    ## resampling Phi
    for (g in 1:G){
      if (k < N){
        alpha[g, (k+1):N] <- rnorm((N-k), mu_a, sigma2a)
        beta[g, (k+1):N] <- rnorm((N-k), mu_b, sigma2b)
      }
      
      Va <- matrix(sigma2a, G, G); Vb <- matrix(sigma2b, G, G)
      # alpha_new <- beta_new <- c()
      for (i in 1:k){
        # alpha_new <- cbind(alpha_new, mvrnorm(mu=alpha[, i], Va))
        # beta_new <- cbind(beta_new mvrnorm(mu=beta[, i], Vb))
        alpha_new <- mvrnorm(mu=alpha[, i], Sigma=Va)
        beta_new <- mvrnorm(mu=beta[, i], Sigma=Vb)
        
        count_1 <- accept_1 <- c_1 <- 0
        ind <- which(s==i)
        while (accept_1 == 0){
          for (count_id_1 in 1:count_num){
            alpha_new[g] <- rnorm(1, alpha[g, i], sigma2a)
            
            L_1 <- exp_trans(alpha_new, beta[, i])
            L_t_1 <- exp_trans(alpha[, i], beta[, i])
            
            p_xi_1 <- p_xi_t_1 <- rep(0, length(ind))
            for (m in 1:length(ind)){
              p_xi_1[m] <- sum(log(dbeta(dat[ind[m], ], L_1$alpha, L_1$beta)))
              p_xi_t_1[m] <- sum(log(dbeta(dat[ind[m], ], L_t_1$alpha, 
                                           L_t_1$beta)))
            }
            # p_xi_1 <- sapply(ind, function(m) sum(log(dbeta(dat[ind[m], ], 
            #                                               L_1$alpha, L_1$beta))))
            # p_xi_t_1 <- sapply(ind, function(m) sum(log(dbeta(dat[ind[m], ], 
            #                                                 L_t_1$alpha, 
            #                                                 L_t_1$beta))))
            p_x_1 <- sum(p_xi_1)
            p_x_t_1 <- sum(p_xi_t_1)
            
            fx_1 <- t(dnorm(alpha[, i], rep(0, G), sigma2a))%*%
              dnorm(beta[, i], rep(0, G), sigma2b)
            fx_t_1 <- t(dnorm(alpha_new, rep(0, G), sigma2a))%*%
              dnorm(beta[, i], rep(0, G), sigma2b)
            
            # Metropolis Hasting updates
            if (p_x_1 == p_x_t_1 & p_x_1 == -Inf){
              if (fx_1 == fx_t_1 & fx_1 == 0){
                temp_1 <- 0
              }
              else{
                temp_1 <- exp(log(fx_1) - log(fx_t_1))
              }
            }
            else{
              if (fx_1 == fx_t_1 & fx_1 == 0){
                temp_1 <- exp(p_x_1-p_x_t_1)
              }
              else{
                if (fx_1 == 0 & p_x_t_1 == -Inf){
                  temp_1 <- exp(p_x_1 - log(fx_t_1))
                }
                else if(fx_t_1 == 0 & p_x_1 == -Inf){
                  temp_1 <- exp(log(fx_1) - p_x_t_1)
                }
                else{
                  temp_1 <- exp(log(fx_1) + p_x_1 - log(fx_t_1) - p_x_t_1)
                }
              }
            }
            u_1 <- runif(1)
            # count <- count + 1
            if (u_1 < temp_1){
              alpha[g, i] <- alpha_new[g]
              count_1 <- count_1 + 1
            }
          }
          c_1 <- c_1 + 1
          if (count_1 >= 3 | c_1 > 5){
            accept_1 <- 1
          }
          else{
            sigma2a <- sigma2a * 0.98
            sigma2a <- ifelse(sigma2a < 0.01, 0.01, sigma2a)
            sigma2a <- ifelse(count_1 > 20, sigma2a/0.98, sigma2a)
          }
        }
        
        accept_2 <- count_2 <- c_2 <- 0
        while (accept_2 == 0){
          
          for (count_id_2 in 1:count_num){
            beta_new[g] <- rnorm(1, beta[g, i], sigma2b)
            ind <- which(s==i)
            L_2 <- exp_trans(alpha[, i], beta_new)
            L_t_2 <- exp_trans(alpha[, i], beta[, i])
            
            p_xi_2 <- p_xi_t_2 <- rep(0, length(ind))
            for (m in 1:length(ind)){
              p_xi_2[m] <- sum(log(dbeta(dat[ind[m], ], L_2$alpha, L_2$beta)))
              p_xi_t_2[m] <- sum(log(dbeta(dat[ind[m], ], L_t_2$alpha, 
                                           L_t_2$beta)))
            }
            
            # p_xi_2 <- sapply(ind, function(m) sum(log(dbeta(dat[, ind[m]], 
            #                                               L_2$alpha, L_2$beta))))
            # p_xi_t_2 <- sapply(ind, function(m) sum(log(dbeta(dat[, ind[m]], 
            #                                                 L_t_2$alpha, 
            #                                                 L_t_2$beta))))
            
            p_x_2 <- sum(p_xi_2); p_x_t_2 <- sum(p_xi_t_2);
            
            fx_2 <- t(dnorm(alpha_new, rep(0, G), sigma2a))%*%
              dnorm(beta[, i], rep(0, G), sigma2b)
            fx_t_2 <- t(dnorm(alpha_new, rep(0, G), sigma2a)) %*% 
              dnorm(beta_new, rep(0, G), sigma2b)
            
            if (p_x_2 == p_x_t_2 & p_x_2 == -Inf){
              if (fx_2 == fx_t_2 & fx_2 == 0){
                temp_2 <- 0
              }
              else{
                temp_2 <- exp(log(fx_2)-log(fx_t_2))
              }
            }
            else{
              if (fx_2 == fx_t_2 & fx_2 == 0){
                temp_2 <- exp(p_x_2 - p_x_t_2)
              }
              else{
                if (fx_2 == 0 & p_x_t_2 == -Inf){
                  temp_2 <- exp(p_x_2 - log(fx_t_2))
                }
                else if(fx_t_2 == 0 & p_x_2 == -Inf){
                  temp_2 <- exp(log(fx_2) - p_x_t_2)
                }
                else{
                  temp_2 <- exp(log(fx_2) + p_x_2 - log(fx_t_2) - p_x_t_2)
                }
              }
            }
            
            u_2 <- runif(1)
            if (u_2 < temp_2){
              count_2 <- count_2 + 1
              beta[g, i] <- beta_new[g]
            }
          }
          c_2 <- c_2 + 1
          if (count_2 >= 3 || c_2 > 5){
            accept_2 <- 1
          }
          else{
            sigma2b <- sigma2b*0.98
            sigma2b <- ifelse(sigma2b < 0.01, 0.01, sigma2b)
            sigma2b <- ifelse(count_2 > 20, sigma2b/0.98, sigma2b)
          }
        }
      }
    }
    ## resample pi
    if (k == 1){
      next
    }
    m_s <- calc_m_s(s)
    pi <- rdirichlet(m_s+tau/k)
    
    ## resample tau from gamma distribution
    r <- rbeta(1, tau+1, N)
    eta_r <- 1/(N*(b-log(r))/(a+k-1) + 1)
    temp <- runif(1)
    tau_new <- ifelse(temp < eta_r, rgamma(a+k, b-log(r)), 
                      rgamma(a+k-1, b-log(r)))
    tau <- tau_new
    
    k <- length(m_s)
    params$s <- s; params$k <- k; params$m_s <- m_s; 
    params$alpha <- alpha; params$beta <- beta;
    params$tau <- tau; params$pi <- pi; 
    params$sigma2a <- sigma2a; params$sigma2b <- sigma2b;
    print(paste(it, "iterations done"))
  }
  print(params$s)
  return(params)
}

sim_data <- generate_simulation(N=100, pi=c(0.4, 0.6), 
                                u=matrix(c(15, 12, 6, 10), 2, 2, byrow=TRUE), 
                                v=matrix(c(6, 20, 12, 4), 2, 2, byrow=TRUE))
p <- bdpmm(sim_data, 2000)


df.Mono <- read_excel("subsetgenes_MonoBvalues.xlsx")
d1 <- data.frame(df.Mono[,-1])
mat.Mono <- matrix(unlist(df.Mono[, 4:ncol(df.Mono)]), nrow(df.Mono), 
                   ncol(df.Mono)-3)

df.Neutro <- read_excel("subsetgenes_NeutroBvalues.xlsx")
d2 <- data.frame(df.Neutro[,-1])
mat.Neutro <- matrix(unlist(df.Neutro[, 4:ncol(df.Neutro)]), nrow(df.Neutro), 
                     ncol(df.Mono)-3)

p_mono <- bdpmm(t(mat.Mono), 4000)
p_neutral <- bdpmm(t(mat.Neutro), 4000)

heatmap_plot <- function(dat, cluster, )
