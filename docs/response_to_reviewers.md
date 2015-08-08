## Response to Reviewers

### Reviewer 1

We thank Reviewer 1 for the helpful suggestions. We agree with Reviewer 1 that
host-parasite distributions are dynamic and that it would be interesting to
apply the methods developed in this paper to a stochastic host-parasite model. However, we believe that this approach is unnecessary/problematic to include in the current manuscript for a few reasons.  

First, stochastic host-parasite models can range from relatively simple models
with an analytical solutions at any given point in time (e.g. a simple death-
immigration process) to complex models with clumped infections, host
heterogeneity, and acquired immunity that can only be explored via simulation.
Given the space alloted in this manuscript and the time alloted for these
revisions, in would be difficult to provide a comprehensive treatment of how our
method works across the numerous different assumptions that have been included
in stochastic host-parasite models. We believe that this exploration would be
better suited for a follow-up study.

However, we can gain some insight into whether our model might be successful in
identifying PIHM in a simple birth-death-immigration process in which the number
of parasites increase by one, decrease by one, or stay the same in a given time
step.  The stationary distribution for a BDI process in which birth rate < death
rate and there is no parasite-induced host mortality is a negative binomial
distribution with $k =$ immigration rate / birth rate. This result is consistent
with our assumption that the pre-mortality distribution of parasites across
hosts is negative binomial. If we were to include parasite-induced host
mortality into this stochastic process, we know from extensive literature that
the variance to mean ratio would decrease and the parasite distribution with
PIHM would be less aggregated than the distribution without PIHM.  If the
immigration and birth rate (or self-reinfection rate) was known for the system,
our model could be applied in which the null model was a negative binomial with
a fixed $k = $ immigration / birth and the alternative was model was defined by
equation X in the manuscript.  If the immigration and birth rate was not known, our method could be applied as given in the manuscript.  The key criteria necessary for our method to work would be that the resulting distribution post mortality took a form significantly different than a negative binomial distribution. We have included an additional Appendix (Appendix X) in which we simulate a birth-death-immigration process with host mortality and test the applicability of our method.  We have also added the following the the Discussion to address Reviewer 1's comments.

### Reviewer 2

We thank Reviewer 2 for their helpful feedback and address each of their points in turn. 

1. Normally I would be asking authors to insert caveats into conclusions when they over spin their work, but in your case I do think you are way too negative about your own work when you talk about the caveats in the discussion and your conclusion / statements in the abstract.

We thank Reviewer 2 for this comment and have made changes in the Abstract and Discussion to cast our method in a more positive light.  In particular, we changed the the following sentences in the abstract from 

*However, we stress that this method, and all techniques that estimate parasite- induced mortality from intensity data alone, have several critical assumptions that limit their applicability in the real world. Due to these limitations, these methods should only be used as an exploratory tool to inform more rigorous studies of parasite-induced host mortality.*

to 

**However, this method, and all techniques that estimate parasite-induced
mortality from intensity data alone, have several important assumptions that
must be scrutinized before applying them to real-world data. Given that these
assumptions are met, our method is new exploratory tool that can help inform
more rigorous studies of parasite-induced host mortality.**

2. Pg 8-9 - to me it could be made clearer why we should care about the Type I error - it is just stated on Pg 9 with no justification. I am not saying the authors are incorrect. I would get this justification in at the start of the section - ie lines 150~ Pg 8. In addition, I would insert here the "by estimating power"after "pre-mortality parameters" (Line 154) and then expand a bit.  Just needs for me a couple of sentences setting up Type 1 and power and why done for the reader.  The more detailed description that follows then flows.

To address these comments, we added the following justification for using Type I error and power at the beginning the subsection "Evaluating the Adjei and Lieklihood Methods"

**We used statistical power and Type I error to test the ability of the Adjei and the Likelihood Methods to correctly identify the presence of PIHM on simulated data with known pre-mortality parameters. The power of a method is the probability of correctly detecting PIHM given that it is occurring and the Type I error is the probability of incorrectly identifying PIHM given that it is not occurring. If a method has low Type I error we can be confident that when we detect PIHM it is actually occurring.  If one method has higher power for detecting PIHM than another, we will need to sample fewer hosts to detect PIHM.**

3. Pg 8 Line 169-175 - this may be V2 of your work and not for this paper, but I wonder what would be the impact of non-random mortality. I am thinking here of say where mortality is a combination of burden and some host factor - eg sex. As said this maybe beyond the scope of the paper, but could be mentioned in the discussion if agree?

This is a great point by Reviewer 2. 

4. lines 201-202 - I assume that any confidence intervals around the average no of surviving hosts are nice and symmetrical - ie that we are happy to just take the mean and do not need to worry about some horrible non linear type distribution of surviving hosts.

Reviewer 2 is correct in assuming that the distribution of surviving hosts was generally quite symmetrical and the standard deviation was small relative to the mean. We made this clear in the manuscript by adding the following line

**The distribution of surviving hosts over the 150 simulations was generally symmetrical and the standard deviation was small compared to the mean (maximum coefficient of variation was approximately 0.06 across all parameter combinations), suggesting that the mean number of surviving hosts was an adequate summary statistic of the number of hosts sampled post-mortality.**

5. Pg 13 - lines 295-302 - is it of interest that the variation your ML method observed in the LD50 seemed a lot less than the Adjei method?  Given this is the same host, same parasite and similar habitats I would be a little concerned at the marked variation in LD50 in the Adjei method

6. Pg 13-14 - lines 316-> - see comments above about being more positive about your work; Pg 15 - I think the conclusions need to be more positive

We addressed these concerns in above section.

7. Figure 2, 3, supplementary figures - distinction between ML and Adeji not always very clear.  Would suggest (a) reordering the legend to ML grad, mode,steep then the Adjei equivalents; (b) having the lines in the legend box longer; (c) ideally some colour to distinguish between ML and Adjei, if not put into the figure legends themselves Adjei=gray, ML=black; (d) can the lines be any thicker on the graphs?

We changed Figures 2, 3 and corresponding supplementary figures according the the suggestions by the reviewer. We reordered the legends, lengthened the lines in the legend box, colored the Adjei Lines green, and made all the lines thicker.  We believe that this makes the figures much more readable.

8. Figure 4 - I would recommend having vertical lines coming down from the steep and moderate to see what sample size is associated with 80% power

Added the recommended vertical lines to the Figure. 

9. Figure S4 missing from my download

Seems to be present in the version that we are looking at.

10. Pg 2 lines 13-15 ref for sentence would be good

Added a citation AndersonandMay1978

11. Pg 5 line 84 - an unnecessary "an"

Deleted the extra "an"

12. Pg 12 - Figure 4 referred to in text before Figure 3 - suggest swapping order of figures

Swapped the order of the figures






