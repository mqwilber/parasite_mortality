
source("analysis_fxns.R")

## 1. Run analysis on full raccon data

# Load in data
raccoon_data = read.csv("../data/all_raccoon_data.csv")
pdf("full_raccoon_hist.pdf", width=7, height=5)
hist(raccoon_data$worms, breaks=30, col='black', border='white',
                xlab="Log Num. Parasites", ylab="Num. Raccoons",
                xlim=c(0, 150), main="")
dev.off()

#raccoon_results = estimate_mortality(raccoon_data$worms, plot=T, weights=F,
#            title="Raccoon Mortality, No Truncation", force_it=FALSE)

# raccoon_results_force = estimate_mortality(raccoon_data$worms, plot=T, weights=T,
#             title="Raccoon Mortality, No Truncation, with force", force_it=T)

raccoon_results_trun = estimate_mortality(raccoon_data$worms, plot=T, weights=T,
             title="Raccoon Mortality, Truncation", force_it=F, trun=120)

# # # Plot observed vs predicted
# # # How much death does this estimate estimate
# # tcol = "#0000ff60"
# # barplot(raccoon_results$obs, width=1, col='red', ylim=c(0, 50))
# # points(1:length(raccoon_results$obs), raccoon_results$pred)

# # hist(raccoon_results$observed, col=tcol, add=TRUE, freq=FALSE)

# # # Truncated analysis. Truncated at 40
# raccoon_results_trun = estimate_mortality(raccoon_data$worms, plot=T,
#                 weights=T, trun=40, title="Raccoon Mortality, Truncation")

# ## 2. Run analysis of chytrid data.  Try it with binned and unbinned

# Load chytrid data; epizootic and enzootic
# epi_data = read.csv("../data/ze_epizootic.csv")
# en_data = read.csv("../data/ze_enzootic.csv")

# en_results = estimate_mortality_bin(en_data$ZE1, plot=T, weights=T, trun=5000,
#                 title="Enzootic Chtyrid Mortality")

# en_results_force = estimate_mortality_bin(en_data$ZE1, plot=T, weights=T,
#             trun=5000, title="Enzootic Chtyrid Mortality, Forced",
#             force_it=TRUE)








