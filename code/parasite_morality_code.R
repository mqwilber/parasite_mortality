library(MASS)

file = "/Users/mqwilber/Desktop/worm_data.csv"
data=read.csv(file)

bp <- data$adBPtot

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

# parasite_nums = seq(0, max(bp), 1)
# new_pred = round(pred)

# new_obs = obs
# ind = obs > new_pred
# new_obs[ind] = new_pred[ind]

# Drop where pred is zero- removes fraction individuals
# start at i=2 because dont include 0 class as should have 100%surv
# ind_nozeros = new_pred != 0
# trun_obs = new_obs[ind_nozeros][2:sum(ind_nozeros)]
# trun_pred = new_pred[ind_nozeros][2:sum(ind_nozeros)]
# trun_x = parasite_nums[ind_nozeros][2:sum(ind_nozeros)]


# # Maximum likelihood function

# logistregfun = function(params, x, N, y){

# 	a = params[1]
# 	b = params[2]
# 	p.pred = exp(a + b *log( x)) / (1 + exp(a + b * log(x)))
# 	-sum(dbinom(y, size=N, prob=p.pred, log=TRUE))
# }

# opt1 = optim(fn = logistregfun, par = c(a=.5,b=.5),
# 				x = trun_x, N=trun_pred, y=trun_obs,
# 				method="BFGS")

# hx<-trun_obs / trun_pred

# # Plotting MLE and data

# a=opt1$par[1]
# b=opt1$par[2]
# x1=seq(1, 75,1)
# y<-exp(a + b * log(x1)) / (1 + exp(a + b * log(x1)))
# plot(y~x1, type="l", ylim=c(0,1))
# points(hx~log(trun_x))

# plot(hx~trun_x, log="x")
# points(y~(x1), type="l", col="blue")

# ?barplot()
# barplot(trun_obs)

# points(trun_pred)
# dev.new()

# # Loss of hosts due to parasites

# k=fits$estimate[1]
# mu=fits$estimate[2]
# num_racoons = length(bp)

# hx_func = function(x){
# 	exp(a + b *log( x)) / (1 + exp(a + b *log( x)))
# }

# Fx_func = function(x){
# 	dnbinom(x, size=k, mu=mu)* num_racoons
# }

# # Calculating host mortality due to parasites
# xs = seq(0, max(trun_x), 1)
# Z0 = 1 - (sum(hx_func(xs) * Fx_func(xs)) / sum(Fx_func(xs)))
# #something not quite right with this now that it is log

# xs = seq(1, max(trun_x), 1)
# Z1 = 1 - (sum(hx_func(xs) * Fx_func(xs)) / sum(Fx_func(xs)))
# Z1


# These values don't make sense, try log 2 binning to account for large parasite numbers


#hist(data$adBPtot, col="black", xlab="Adult Parasites", main="",
#	cex.lab=1.5, cex.axis=2, freq=FALSE, breaks=20)