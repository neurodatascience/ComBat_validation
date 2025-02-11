library(gtools)
set.seed(666)
#sample size is 5000
N=5000
#the number of features is 1000.
G=10
#10 sites
I=10
####################
alpha=sample(1,20,I,replace=T)
p=gtools::rdirichlet(1,alpha)

library(nimble)
library(LaplacesDemon)

n_i=round(p*N)

alpha_g=runif(G,0,0.5)

Y_i=runif(I,0,0.1)
tau_i=nimble::rinvgamma(n = I, shape=2, scale = 0.5)
gamma_ig=sapply(1:I,function(i) rnorm(G,mean=Y_i[i],sd=sqrt(tau_i[i])))

lambda_i=rgamma(I,shape=50,scale=50)
v_i=rgamma(I,shape=50,scale=1)
delta_ig=sapply(1:I, function(i) rgamma(G, shape = lambda_i[i] * v_i[i], scale = v_i[i]))

sigma_g=LaplacesDemon::rhalfcauchy(G,scale =0.2)
epsilon_ijg=lapply(1:G, function(g) rnorm(n_i[g], mean = 0, sd = sigma_g[g]))
 



