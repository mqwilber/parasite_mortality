library(MASS)

# file = "../data/worm_data.csv"
# data=read.csv(file)
# bp <- data$adBPtot

file = "../data/all_raccoon_data.csv"
data=read.csv(file)
bp <- data$worms

# Count number of racs with a given parasite
count_data = as.data.frame(table(bp))
full_data = data.frame(list("bp" = seq(0, max(bp), 1), "Freq" = rep(0, max(bp) + 1)))
full_data[as.numeric(as.vector(count_data$bp)) + 1, "Freq"] = count_data$Freq
obs = full_data$Freq


# Get predicted numbers from negative binomial
fits = fitdistr(bp, "negative binomial", list(size=.5, mu= 15), lower=.001)
vec = seq(0, max(bp), 1)
probs = dnbinom(vec, size=fits$estimate["size"], mu=fits$estimate["mu"])
pred = probs * length(bp)  # Maybe set fractional raccoons to zero?


# # Observed can't be greater than predicted...seems like a hack-
# #is binomial the best assumed distribution- cant have obs >pred

parasite_nums = seq(0, max(bp), 1)
new_pred = round(pred)
new_obs = obs
ind = obs > new_pred
new_obs[ind] = new_pred[ind]

# Drop where pred is zero- removes fraction individuals
# start at i=2 because dont include 0 class as should have 100%surv
ind_nozeros = new_pred != 0
trun_obs = new_obs[ind_nozeros][2:sum(ind_nozeros)]
trun_pred = new_pred[ind_nozeros][2:sum(ind_nozeros)]
trun_x = parasite_nums[ind_nozeros][2:sum(ind_nozeros)]


# # Maximum likelihood function

logistregfun = function(params, x, N, y){

	a = params[1]
	b = params[2]
	p.pred = exp(a + b *log(x)) / (1 + exp(a + b * log(x)))
	-sum(dbinom(y, size=N, prob=p.pred, log=TRUE))
}

opt1 = optim(fn = logistregfun, par = c(a=.5,b=.5),
				x = trun_x, N=trun_pred, y=trun_obs,
				method="BFGS")

#a=opt1$par[1]
#b=opt1$par[2]

### GLM in R: Should give the same results at the direct optimization
fits_glm = glm(cbind(trun_obs, trun_pred - trun_obs) ~
            log(trun_x), family=binomial, weights=trun_pred)

a = coef(fits_glm)[1]
b = coef(fits_glm)[2]


hx<-trun_obs / trun_pred

# # Plotting MLE and data

x1=seq(0.0001, 75, 0.0001)
y<-exp(a + b * log(x1)) / (1 + exp(a + b * log(x1)))

plot(log(trun_x), hx, xlab="Num. parasites", ylab="Prob. Survival")
lines(log(x1), y)


# # Loss of hosts due to parasites

k=fits$estimate[1]
mu=fits$estimate[2]
num_racoons = length(bp)

#norm_factor = 0.79899 # To account for the fact that we see mortality at one
hx_func = function(x){
	exp(a + b *log(x)) / (1 + exp(a + b *log( x))) #/ norm_factor
}

Fx_func = function(x){
	dnbinom(x, size=k, mu=mu) * num_racoons
}

# # Calculating host mortality due to parasites
xs = seq(1, max(trun_x), 1)
numerator = c(Fx_func(0), Fx_func(xs)) * c(1, hx_func(xs))
sum_numerator = sum(numerator)

denom = c(Fx_func(seq(0, max(trun_x), 1)))
sum_denom = sum(denom)

Z0 = 1 - sum_numerator / sum_denom

Z1 = 1 - (sum(hx_func(xs) * Fx_func(xs)) / sum(Fx_func(xs)))

Z2 = 1 - (sum(xs * hx_func(xs) * Fx_func(xs)) / sum(xs * Fx_func(xs)))


# What are some of the problems with this method
# 1.  If you have juvenile mortality for reasons other than parasite mortality
# you will overestimate the the probability of death caused by parasites.
# 2.  Can we come up with a way to control for other factors of mortality?
# 3. What about the relationship betwen age structure and parasite load?
# Or hetergeneity and parasite load?  There is a distribution of parasites
# across hosts that differes by age class.  This is where you will begin to
# loose the power to detect these changes if you only have one sample.
#  4.
# This is not incorporated.  There is a distribution of parasites across