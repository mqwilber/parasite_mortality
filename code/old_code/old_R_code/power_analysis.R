# Code for simulating parasite mortality...should put these all into functions

# Load fxns
source("analysis_fxns.R")

# Parameters
num_hosts = 10000
k = .5
mu = 15
a = 6 # LD50 a
b = -2  # LD50 b
LD50 = exp(a / abs(b))

survival_prob = function(x, a, b){

    res = exp(a + b * log(x)) / (1 + exp(a + b * log(x)))
    return(res)

}

ks = seq(0.01, 1, length=50)
trun_vals = seq(0.1, 1, length=50)
a_vals = matrix(NA, nrow=length(ks), ncol=length(trun_vals))
b_vals = matrix(NA, nrow=length(ks), ncol=length(trun_vals))
mort_ratio = matrix(NA, nrow=length(ks), ncol=length(trun_vals))

mort_diff = matrix(NA, nrow=length(ks), ncol=length(trun_vals))

for(i in 1:length(ks)){
    for(j in 1:length(trun_vals)){

        # 1. Equilibrium is established
        sim_data = rnbinom(num_hosts, size=ks[i], mu=mu)
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

        trun_val = trun_vals[j] * LD50

        ests1 = estimate_mortality(alive_hosts, plot=F, weights=T,
                    trun=trun_val, force_it=T, save=FALSE,
                        title="Observed Survival Curve")

        a = ests1$fits$a
        b = ests1$fits$b

        a_vals[i, j] = a
        b_vals[i, j] = b
        mort_ratio[i, j] = ests1$fits$Z0 / true_death
        mort_diff[i, j] = ests1$fits$Z0 - true_death


    }
}

write.matrix(a_vals, "a_vals.txt")
write.matrix(b_vals, "b_vals.txt")
write.matrix(ks, "ks.txt")
write.matrix(trun_vals, "trun_vals.txt")
write.matrix(mort_ratio, "mort_ratio.txt")
write.matrix(mort_diff, "mort_diff.txt")





