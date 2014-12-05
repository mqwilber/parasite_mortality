# Code for simulating parasite mortality

# Load fxns
source("analysis_fxns.R")

# Parameters
num_hosts = 10000
k = 1
mu = 15
a = 6 # LD50 a
b = -2  # LD50 b
LD50 = exp(a / abs(b))

survival_prob = function(x, a, b){

    res = exp(a + b * log(x)) / (1 + exp(a + b * log(x)))
    return(res)

}

# 1. Equilibrium is established
sim_data = rnbinom(num_hosts, size=k, mu=mu)
sorted_sim = as.data.frame(table(sim_data))
sim_para = as.numeric(levels(sorted_sim$sim_data)[as.integer(sorted_sim$sim_data)])
sim_freq = sorted_sim$Freq

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
max_para =
vals = seq(1, max(sim_data), 1)
plot(log(vals), survival_prob(vals, a, b), 'l', xlab="Log Num. Parasites",
            ylab="Prob. Survival of Host", main="True Survival Curve")


trun_val = .25 * LD50

# How much death does this estimate estimate
tcol = "#0000ff60"
hist(log(sim_data), col='red', breaks=15, freq=T, xlab="Log Num. Parasites",
        ylab="Num. Hosts", main="Pre vs Post Mortality", xlim=c(0,7))
hist(log(alive_hosts), col=tcol, breaks=15, add=TRUE, freq=T)
abline(v=log(LD50), col='grey', lwd=4, lty='dashed')
abline(v=log(trun_val), col='black', lwd=4, lty='dashed')

legend("topright", c("Pre-mortality", "Observed"), fill=c("red", tcol),
    cex=0.6)

ests1 = estimate_mortality(alive_hosts, plot=TRUE, weights=T,
            trun=trun_val, force_it=T, save=FALSE,
            title="Observed Survival Curve")

#ests2 = estimate_mortality_bin(alive_hosts, plot=TRUE, weights=TRUE,
#    trun=trun_val)

#plot(ecdf(sim_data), xlab="Num, Parasites", ylab='ECDF')
#lines(ecdf(alive_hosts), pch=19, col='red')

pred = ests1$pred
obs = ests1$obs
para_num = ests1$para_num

max_pt = max(c(pred, obs, sim_freq[2:length(sim_freq)]))
plot(log(para_num), pred, col='red', xlab="Log Num. of Parasites",
            ylab="Num. Hosts",
            main="Parasites per host", ylim=c(0, max_pt))
points(log(para_num), obs)
points(log(sim_para), sim_freq, col='blue')


legend("topright", c("Predicted", "Observed", "Simulated"),
        fill=c("red", "black", "blue"), cex=0.6)


