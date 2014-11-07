# Code for simulating parasite mortality

# Load fxns
source("analysis_fxns.R")

# Parameters
num_hosts = 1000
k = 1
mu = 20
a = 6  # LD50 a
b = -2  # LD50 b

survival_prob = function(x, a, b){

    res = exp(a + b * log(x)) / (1 + exp(a + b * log(x)))
    return(res)

}

# 1. Equilibrium is established
sim_data = rnbinom(num_hosts, size=k, mu=mu)

# 2. Parasite induced mortality occurs
surv_probs = survival_prob(sim_data, a, b)
surv_probs[is.na(surv_probs)] = 1

# Determine if an individual dies or not
rands = runif(length(sim_data))
surv = rands < surv_probs
alive_hosts = sim_data[surv]

true_death = 1 - sum(surv) / length(surv)

par(mfrow=c(2,2))

# Plot survival fxn
vals = seq(1, max(sim_data), 1)
plot(log(vals), survival_prob(vals, a, b), 'l', xlab="log Num Parasites",
            ylab="Survival prob.")

ests1 = estimate_mortality(alive_hosts, plot=TRUE, weights=FALSE)
ests2 = estimate_mortality_bin(alive_hosts, plot=TRUE, weights=FALSE)

plot(ecdf(sim_data), xlab="Num, Parasites", ylab='ECDF')
lines(ecdf(alive_hosts), pch=19, col='red')

# How much death does this estimate estimate


#hist(sim_data, col='blue', freq=FALSE)
#hist(alive_hosts, col='red', add=TRUE, freq=FALSE)


