library(MASS)

## Defines analysis functions for analysing mortality data

hx_func = function(x, a, b){
    # Survival fxn
    exp(a + b *log(x)) / (1 + exp(a + b *log( x)))
}


Fx_func = function(x, k, mu, num){
    # NBD x number
    dnbinom(x, size=k, mu=mu) * num
}

fit_model = function(obs, pred, para_number, num, k, mu,
        plot=FALSE, weights=FALSE, title="Default", save=TRUE){

    # obs : Observed hosts with i parasites
    # pred : Predicted hosts with i parasites, should match obs
    # num : Total number of host in the sample
    # para_number : vector with the number of parasites in each bin
    # k : k of the NBD fir to data
    # mu : mu of the NBD fit to data
    # plot : if True plot the survival fxn
    # weights : if True, run glm with observations wiegthed by sample size
    #
    # Returns : list of Z0 (prop, of population dead), Z1 (prop, of infected
    #                population dead), a (parameter of surv. fxn),
    #                b (parameter of surv fxn)

    if(weights){
        fits_glm = glm(cbind(obs, pred - obs) ~
                log(para_number), family=binomial, weights=pred)
    }
    else{
        fits_glm = glm(cbind(obs, pred - obs) ~
                log(para_number), family=binomial)
    }

    a = coef(fits_glm)[1]
    b = coef(fits_glm)[2]

    # # Calculating host mortality due to parasites
    all_parasites = seq(1, max(para_number), 1)
    numerator = c(Fx_func(0, k, mu, num), Fx_func(all_parasites, k, mu, num)) *
                                c(1, hx_func(all_parasites, a, b))
    sum_numerator = sum(numerator)

    x_with_0 = 0:max(para_number)
    denom = c(Fx_func(x_with_0, k, mu, num))
    sum_denom = sum(denom)

    Z0 = 1 - sum_numerator / sum_denom

    Z1 = 1 - (sum(hx_func(all_parasites, a, b) * Fx_func(all_parasites, k, mu, num)) /
                        sum(Fx_func(all_parasites, k, mu, num)))

    if(plot){

        if(save){
            pdf(paste(gsub(" ", "_", title), ".pdf", sep=""), width=7, height=5)
        }

        hx = obs / pred
        vals = seq(0.0001, max(para_number), 1)
        plot(log(para_number), hx, xlab="Log Num. Parasites",
                ylab="Prob. Survival of Host", ylim=c(0,1), main=title)
        lines(log(vals), hx_func(vals, a, b))

        if(save){
            dev.off()
        }
    }

    return(list("Z0" = Z0, "Z1" = Z1, 'a'=a, 'b'=b))

}

estimate_mortality = function(para_data, plot=FALSE, weights=FALSE, trun=1000,
    title="Default", force_it=FALSE, save=TRUE){
    # If only you had a docstring...
    # This function computes and returns the proportion of the population lost
    # to parasites and the the proportion of the infected population lost to
    # parasites as well as the a and b parameters of the survival function.

    # Fit data
    count_data = as.data.frame(table(para_data))
    full_data = data.frame(list("para_data" = seq(0, max(para_data), 1),
                    "Freq" = rep(0, max(para_data) + 1)))
    full_data[as.numeric(as.vector(count_data$para_data)) + 1, "Freq"] = count_data$Freq
    obs = full_data$Freq
    num = length(para_data)

    # Get predicted numbers from negative binomial
    trun_data = para_data[para_data < trun]

    fits = mle_fit_nbd(round(para_data), 1, trun)
    k = fits[1]
    mu = fits[2]
    #fits = fitdistr(trun_data, "negative binomial", list(size=.5, mu= 15), lower=.001)
    #notrun_fits = fitdistr(para_data, "negative binomial", list(size=.5, mu= 15), lower=.001)
    #k = fits$estimate["size"]
    #notrun_k = notrun_fits$estimate["size"]

    #print(paste("Length of trun data" , length(trun_data)))
    #print(paste("k of truncated fit", k, ", k no truncated fit", notrun_k))


    vec = seq(0, max(para_data), 1)
    probs = dnbinom(vec, size=k, mu=mu)
    pred = probs * length(para_data)  # Maybe set fractional raccoons to zero?

    # # Observed can't be greater than predicted...seems like a hack-
    # # is binomial the best assumed distribution- cant have obs >pred
    parasite_nums = seq(0, max(para_data), 1)
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

    if(force_it){
        # Finally, make sure the first group has 100% survival
        trun_obs[1] = trun_pred[1]
    }

    # Drop all values where predicted is only 1...
    # ind1 = trun_pred != 1
    # trun_obs = trun_obs[ind1]
    # trun_pred = trun_pred[ind1]
    # trun_x = trun_x[ind1]

    return(list("fits"=fit_model(trun_obs, trun_pred, trun_x, num, k, mu,
                    plot=plot, weights=weights, title=title, save=save), "obs"=trun_obs,
                    "pred"=trun_pred, "para_num"=trun_x))

}


estimate_mortality_bin = function(para_data, log_base=2, start=1,
    plot=FALSE, weights=FALSE, trun=1000, title="Default", force_it=FALSE){

    # Bin observe data
    max_num = ceiling(log(max(para_data), base=log_base))
    bins = c(-0.1, log_base^(start:max_num))
    cuts = cut(para_data, bins)
    df = data.frame(list("data"=para_data, "bins"=cuts))
    observed = aggregate(para_data~bins, df, length)

    # Get predicted numbers from negative binomial
    trun_obs = para_data[para_data < trun]
    trun_obs = round(trun_obs)

    fits = mle_fit_nbd(round(para_data), 1, trun)
    k = fits[1]
    mu = fits[2]
    #fits = fitdistr(trun_obs, "negative binomial", list(size=.01, mu=mean(trun_obs)), lower=.001)
    # k = fits$estimate["size"]
    # mu = fits$estimate["mu"]
    vec = seq(0, max(para_data), 1)
    probs = dnbinom(vec, size=k, mu=mu)
    pred = probs * length(para_data)

    tbins = 0:max(para_data)
    new_cuts = cut(tbins, bins)
    pred_df = data.frame(list("data"=pred, "bins"=new_cuts))
    predicted = aggregate(pred~bins, pred_df, sum)
    #return(list("obs"=observed, "pred"=predicted))

    # # Round these estimates?
    predicted$pred = round(predicted$pred)

    # Observed can't be bigger than predicted
    ind = observed$para_data > predicted$pred
    observed$para_data[ind] = predicted$pred[ind]

    if(force_it){
        # The first group should have 100% survival. Force it so it does.
        # Another hack that is required
        observed$para_data[1] = predicted$pred[1]
    }


    # Set observed parasites to upper bound of the group
    group_upper = log_base^(start:max_num)

    #return(list(observed, predicted, group_upper))
    return(list("fits"=fit_model(observed$para_data, predicted$pred, group_upper,
            length(para_data), k, mu, plot=plot, weights=weights, title=title),
            "obs"=observed$para_data, "pred"=predicted$pred,
            "para_num"=group_upper))

}


mle_fit_nbd = function(data, guess_k, trunc){
    # NBD optimization of k with a fixed mean

    fixed_mu = mean(data)
    trun_data = data[data < trunc]

    mle_func = function(k, x, mu){

        -sum(dnbinom(x, size=k, mu=mu, log=TRUE))
    }

    opt1 = optim(fn = mle_func, par = c(guess_k),
                x = trun_data, mu=fixed_mu, method="L-BFGS-B", lower=1e-5)

    return(c(opt1$par[1], fixed_mu))

}

