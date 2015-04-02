library(MASS)

#file = "../data/raccoon_data10_24.csv"
#data=read.csv(file)
#bp <- data$Weinstein
#obs = bp[!is.na(bp)]
# file = "../data/ze_adults_enzootic.csv"
# data = read.csv(file)
# obs = data$ZE1
# obs = round(obs)
file = "../data/all_raccoon_data.csv"
data=read.csv(file)
obs <- data$worms

# Bin observe data
log_base = 2
start = 1
max_num = ceiling(log(max(obs), base=log_base))
bins = c(-0.1, log_base^(start:max_num))
cuts = cut(obs, bins)
df = data.frame(list("data"=obs, "bins"=cuts))
observed = aggregate(obs~bins, df, length)

# # Get predicted numbers from negative binomial
trun_obs = obs
fits = fitdistr(trun_obs, "negative binomial", list(size=.5, mu= 15), lower=.001)
vec = seq(0, max(obs), 1)
probs = dnbinom(vec, size=fits$estimate["size"], mu=fits$estimate["mu"])
pred = probs * length(obs)

tbins = 0:max(obs)
new_cuts = cut(tbins, bins)
pred_df = data.frame(list("data"=pred, "bins"=new_cuts))
predicted = aggregate(pred~bins, pred_df, sum)

# # Round these estimates?
predicted$pred = round(predicted$pred)

# Observed can't be bigger than predicted
ind = observed$obs > predicted$pred
observed$obs[ind] = predicted$pred[ind]

# Set observed parasites to upper bound of the group
group_upper = log_base^(start:max_num)

# # # Maximum likelihood function

# logistregfun = function(params, x, N, y){

#     a = params[1]
#     b = params[2]
#     p.pred = exp(a + b *log(x)) / (1 + exp(a + b * log(x)))
#     -sum(dbinom(y, size=N, prob=p.pred, log=TRUE))
# }

# opt1 = optim(fn = logistregfun, par = c(a=.5,b=.5),
#                 x = group_upper, N=predicted$pred, y=observed$obs,
#                 method="BFGS")

# a = opt1$par[1]
# b = opt1$par[2]

hx = observed$obs / predicted$pred

### GLM in R: Should give the same results at the direct optimization when you
### drop the weights
fits_glm = glm(cbind(observed$obs, predicted$pred - observed$obs) ~
            log(group_upper), family=binomial, weights=predicted$pred)

# Weights estimates
a = coef(fits_glm)[1]
b = coef(fits_glm)[2]

# # # Plotting MLE and data
x1 = seq(0.0001, max(obs), 1)
y = exp(a + b * log(x1)) / (1 + exp(a + b * log(x1)))

plot(log(group_upper), hx, xlab="Log num. parasites", ylab="Prob. Survival",
    ylim=c(0,1))
lines(log(x1), y)


# # # Loss of hosts due to parasites

k=fits$estimate[1]
mu=fits$estimate[2]
num_racoons = length(obs)

# norm_factor = 0.79899 # To account for the fact that we see mortality at one
hx_func = function(x){
    exp(a + b * log(x)) / (1 + exp(a + b *log( x)))
}

Fx_func = function(x){
    dnbinom(x, size=k, mu=mu) * num_racoons
}

# # # Calculating host mortality due to parasites
xs = seq(1, max(group_upper), 1)
numerator = c(Fx_func(0), Fx_func(xs)) * c(1, hx_func(xs))
sum_numerator = sum(numerator)

denom = c(Fx_func(seq(0, max(group_upper), 1)))
sum_denom = sum(denom)

Z0 = 1 - sum_numerator / sum_denom

Z1 = 1 - (sum(hx_func(xs) * Fx_func(xs)) / sum(Fx_func(xs)))

Z2 = 1 - (sum(xs * hx_func(xs) * Fx_func(xs)) / sum(xs * Fx_func(xs)))


