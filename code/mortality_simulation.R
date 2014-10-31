# Code for simulating parasite mortality

# Parameters
num_hosts = 100
k = 1
mu = 20
ld50a = 1
ld50b = 1


survival_prob = function(x, a, b){

    res = exp(a + b * log(x)) / (1 + exp(a + b * log(x)))
    return(res)

}

# 1. Equilibrium is established
sim_data = rnbinom(num_hosts, size=k, mu=mu)

# 2. Parasite induced mortality occurs

