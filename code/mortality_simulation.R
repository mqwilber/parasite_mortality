# Code for simulating parasite mortality

# Load fxns
source("analysis_fxns.R")

# Parameters
num_hosts = 100
k = .2
mu = 15
a = 6  # LD50 a
b = -2  # LD50 b
LD50 = exp(a / abs(b))

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

trun_val = .5 * LD50
ests1 = estimate_mortality(alive_hosts, plot=TRUE, weights=TRUE,
            trun=trun_val)

#ests2 = estimate_mortality_bin(alive_hosts, plot=TRUE, weights=TRUE,
#    trun=trun_val)

#plot(ecdf(sim_data), xlab="Num, Parasites", ylab='ECDF')
#lines(ecdf(alive_hosts), pch=19, col='red')

# How much death does this estimate estimate
tcol = "#0000ff60"
hist(sim_data, col='red', freq=FALSE)
hist(alive_hosts, col=tcol, add=TRUE, freq=FALSE)

pred = ests1$pred
obs = ests1$obs
para_num = ests1$para_num

plot(para_num, pred, col='red')
points(para_num, obs)

legend("right", c("Predicted", "Observed"), fill=c("red", "black"))


