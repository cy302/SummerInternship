setwd("~/Desktop/MPhilComputationalBiology/Internship/")
library(readxl)

df.Mono <- read_excel("subsetgenes_MonoBvalues.xlsx")
d1 <- data.frame(df.Mono[,-1])
mat.Mono <- matrix(unlist(df.Mono[, 4:ncol(df.Mono)]), nrow(df.Mono), 
                   ncol(df.Mono)-3)
colnames(d1)[-c(1, 2)] <- sapply(colnames(d1)[-c(1, 2)],
                                 function(x){substr(x, 1, 6)})

df.Neutro <- read_excel("subsetgenes_NeutroBvalues.xlsx")
d2 <- data.frame(df.Neutro[,-1])
d2 <- data.frame(df.Neutro[,-1])
mat.Neutro <- matrix(unlist(df.Neutro[, 4:ncol(df.Neutro)]), nrow(df.Neutro), 
                     ncol(df.Mono)-3)
colnames(d2)[-c(1, 2)] <- sapply(colnames(d2)[-c(1, 2)],
                                 function(x){substr(x, 1, 6)})

## N samples, each sample is D-dimensional, M clusters

get_ratio <- function(dm, nclusters, num_iter){
  ratio_mat <- matrix(0, nrow(dm), nclusters)
  for (i in 1:num_iter){
    k <- as.vector(kmeans(dm, centers=nclusters)$cluster)
    for (j in 1:nclusters){
      ratio_mat[which(k==j), j] <- ratio_mat[which(k==j), j] + 1
    }
  }
  ratio_mat <- ratio_mat/num_iter
  return(ratio_mat)
}

generate_simulation <- function(N, pi, u, v){
  dat <- matrix(0, N, ncol(u))
  a <- 1:N
  l <- list()
  for (i in 1:(length(pi)-1)){
    l[[i]] <- sample(a, round(pi[i]*N))
    a <- a[-l[[i]]]
  }
  l[[i+1]] <- a
  for (i in 1:length(pi)){
    for (j in 1:ncol(u)){
      dat[l[[i]], j] <- rbeta(length(l[[i]]), u[i, j], v[i, j])
    }
  }
  return(dat)
}

sim_data <- generate_simulation(N=1000, pi=c(0.5, 0.5), 
                    u=matrix(c(15, 12, 6, 10), 2, 2, byrow=TRUE), 
                    v=matrix(c(6, 20, 12, 4), 2, 2, byrow=TRUE))
r1 <- get_ratio(sim_data, 15, 100)

EVI <- function(M=15, c_init=0.001, alpha_init=0.1, beta_init=0.2, mu_init=0.62, 
                eta_init=0.82, max_iter=10000, data.mat){
  N <- ncol(data.mat)
  D <- nrow(data.mat)
  
  # initialisation
  Alpha <- matrix(alpha_init, D, M)
  Beta <- matrix(beta_init, D, M)
  Mu <- matrix(mu_init, D, M)
  Eta <- matrix(eta_init, D, M)
  C <- rep(c_init, M)
  R <- get_ratio(dm=t(data.mat), nclusters=M, num_iter=1)
  U_bar <- Mu/Alpha; V_bar <- Eta/Beta
  
  con <- FALSE
  L_B_old <- Inf
  ct <- 0
  
  E_log_v <- digamma(Eta) - log(Beta)
  E_log_u <- digamma(Mu) - log(Alpha)
  
  # Alpha_ast <- Alpha; Beta_ast <- Beta; Mu_ast <- Mu; Eta_ast <- Eta;
  while (con!=TRUE & ct < max_iter){
    # updating hyperparameter
    
    E_z <- R
    C_ast <- C + colSums(E_z)
    
    Mu_ast <- Mu + matrix(rep(colSums(E_z), D), D, M, byrow=TRUE)*
      U_bar*(digamma(U_bar+V_bar)-digamma(U_bar)+ 
               V_bar*psigamma(U_bar+V_bar, deriv=1)* (E_log_v - log(V_bar)))
    Alpha_ast <- Alpha - log(data.mat) %*% E_z
    Eta_ast <- Eta + matrix(rep(colSums(E_z), D), D, M, byrow=TRUE)*
      V_bar*(digamma(U_bar+V_bar)-digamma(V_bar)+
               U_bar*psigamma(U_bar+V_bar, deriv=1)* (E_log_u - log(U_bar)))
    Beta_ast <- Beta - log(1 - data.mat) %*% E_z
    C <- C_ast; Mu <- Mu_ast; Eta <- Eta_ast; Alpha <- Alpha_ast; Beta <- Beta_ast;
    U_bar <- Mu/Alpha; V_bar <- Eta/Beta
    
    E_log_v <- digamma(Eta) - log(Beta)
    E_log_u <- digamma(Mu) - log(Alpha)
    
    log_rho <- matrix(rep(digamma(C)-digamma(sum(C)), N), N, M, byrow=TRUE) + 
      t(log(data.mat))%*%(U_bar-1) + t(log(1-data.mat))%*%(V_bar-1)
    
    # s <- matrix(0, N, M)
    # for (j in 1:N){
    #   for (k in 1:M){
    #     ss <- 0
    #     for (l in 1:D){
    #       ss <- ss+(U_bar[l, k]-1)*log(data.mat[l, j])+(V_bar[l, k]-1)*
    #         log(1-data.mat[l, j])
    #     }
    #     s[j, k] <- ss
    #   }
    # }
    
    a <- colSums(log(gamma(U_bar+V_bar)/gamma(U_bar)+ gamma(V_bar))+
                U_bar*(digamma(U_bar+V_bar)-digamma(U_bar))*(E_log_u-log(U_bar))+
                V_bar*(digamma(U_bar+V_bar)-digamma(V_bar))*(E_log_v-log(V_bar))+
                0.5*(U_bar^2)*(psigamma(U_bar+V_bar, deriv=1)-psigamma(U_bar, deriv=1))*((digamma(Mu)-log(Mu))^2+psigamma(Mu, deriv=1))+
                0.5*(V_bar^2)*(psigamma(U_bar+V_bar, deriv=1)-psigamma(V_bar, deriv=1))*((digamma(Eta)-log(Eta))^2+psigamma(Eta, deriv=1))+
                U_bar*V_bar*psigamma(U_bar+V_bar, deriv=1)*(E_log_u-log(U_bar))*(E_log_v-log(V_bar)))
    
    log_rho <- t(apply(log_rho, 1, function(x) x+a))
    
    log_rho[which(log_rho>100)] <- log_rho[which(log_rho>100)]/10
    log_rho[which(log_rho< -100)] <- log_rho[which(log_rho< -100)]/10
    
    rho <- exp(log_rho)
    rs_rho <- matrix(rep(rowSums(rho), M), N, M)
    R <- rho/rs_rho
    L_B <- sum(log(1/beta(U_bar, V_bar)))
    con <- all.equal(L_B, L_B_old, tolerance=1e-3)
    
    L_B_old <- L_B
    ct <- ct + 1
  }
  # p <- 1 + colSums(E_z)
  # q <- rep(0, M-2)
  # for (i in 1:(M-2)){
  #   q[i] <- C[i] + sum(E_z[, (i+1):M])
  # }
  # q <- c(q, C[14] + sum(E_z[, M]))
  # lambda <- head(p, M-1)/(head(p, M-1)+q)
  # lambda <- c(lambda, 1)
  # pi <- rep(0, M)
  # for (i in 1:M){
  #   pi[i] <- lambda[i]*prod(head(1-lambda, i-1))
  # }
  # M_correct <- which(pi > 1e-3)
  E_z <- R
  Z <- apply(E_z, 1, which.max)
  pi <- sapply(1:M, FUN = function(x){length(which(Z==x))})/N
  return(list("mixing_prob"=pi, "latent_var"=Z, "num_iter"=ct))
}
sim_data <- generate_simulation(N=1000, pi=c(0.2, 0.8), 
                                u=matrix(c(15, 12, 6, 10), 2, 2, byrow=TRUE), 
                                v=matrix(c(6, 20, 12, 4), 2, 2, byrow=TRUE))
p_vb <- EVI(data.mat=t(sim_data), max_iter=5000)

p_mono <- EVI(data.mat=mat.Mono)
p_neutral <- EVI(data.mat=t(mat.Neutro))

Z_mono <- p_mono$latent_var
l_Z_mono <- length(unique(Z_mono))
label_Z_mono <- unique(Z_mono)
for (i in 1:l_Z_mono){
  Z_mono[which(Z_mono==label_Z_mono[i])] <- i
}
Z_mono[which(Z_mono==4)] <- 2

rownames(mat.Mono) <- d1$GENE_NAME
Mono.dendro <- as.dendrogram(hclust(d = dist(x = mat.Mono)), labels=FALSE, 
                             leaf_labels = FALSE)
Mono.dendro.plot <- ggdendrogram(data = Mono.dendro, rotate = TRUE)
Mono.dendro.plot <- Mono.dendro.plot + 
  theme(axis.text.y = element_text(size = 3))
print(Mono.dendro.plot)

Mono.order <- order.dendrogram(Mono.dendro)
d1.1$GENE_NAME <- factor(x = d1.1$GENE_NAME,
                         levels = d1.1$GENE_NAME[Mono.order], 
                         ordered = TRUE)

data.mat <- t(mat.Mono)
data.mat <- t(sim_data)





