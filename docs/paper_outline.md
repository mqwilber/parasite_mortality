## Paper Outline

Kevin's 5 components
1. Explain the logic behind the method []

2. Give examples of people that have used it [Easy for Crofton Method.  Not
many people have used Adjei's method]

3. Explain the logic behind why it does not work - including critiques by others
[Crofton's method works fine.  But Adjei's method does not work]

4. Show the simulations [got some preliminary sims]

5. Give a list of alternative ways to estimate parasite-induced mortality [Show
our alternative ]

Journal Targets: Journal of Parasitology, International journal of parasitology.

Title: {Functional form of parasite induced host mortality cannot be estimated
using deviations from the negative binomial distribution.}

Authors: Mark Wilber and Sara Weinstein 

## Outline

1. Intro (SW)
    * Disease has major impacts on animal populations.
       * Change population dynamics/stability (grouse/nematodes), alter predator prey interactions (Isle royal echinococcus/moose),  and cause species decline/extinction (amphibian chytrid, distemper in African predators)
       * Parasites can potentially regulate host populations by reducing fecundity or increasing mortality.
       * Accurate estimates of parasite induced host mortality in wild animals are important to understand what regulate both host and parasite populations and to make predictions about disease transmission in natural systems.
    * Explain the logic behind the method: Parasite distribution  suggested as a method of estimating parasite induced host mort.(KP 1)
       * Parasite distributions in hosts can typically be modeled using the negative binomial distribution (shaw paper).
       * Assume parasite induced mortality is intensity dependent then we expect to see a reduction in the number of infected hosts with high worm burdens, truncating the tail of the distribution.     
    * History of use (KP2)
       * First proposed by Crofton (1971),
       * Used by Royce and Rossignol (1990) with honey bee mites (with issues)
       * Lanciani and Boyett (1980) with water mites on mosquitos
       * adjei 1986 with lizard fish with larval tapeworms.
       *ferguson paper fish
       * Comparing the observed frequency of independent infection events and calculating the probability of co-occurrence, as done to estimate the mortality caused by  flounder eye copepods (author, 1964?).  From field data calculated the probability of each eye being infected alone, and assume there is no mortality associated with only one eye infection.  Use these probabilities to calculate the expected frequency of double infections, compare to observed frequency. 
          *  a simple version of the truncated binomial idea, but parasite intensity is limited to 0,1,2- the flounder study didn't work (they say sample size), but it might be worth testing with the simulations to see if this may be an instance where this method could work.
    *  deviations from the negative binomial distribution is still discussed and suggested amongst parasitologists as a method for estimating parasite induced host mortality.
       * Our goal: use simulations to test for when this method can be used.
2. The method/model (MW)
    * Description of Crofton's method (Crofton 1971, Royce 1990, Fergunson et al
    2011, Lanciani 1989)
    * Lanciani (1989) point out that Crofton's method can't be used to detect
    linear effects of parasite induced mortality.  It can be used to detect
    non-linear ones.  However, can this functional form be recovered?
    * Adjei proposed a method to recover the functional form of non-linear
    parasite induced mortality.  Their method proceeded in two steps. 1) They
    used the Crofton Method to estimate the negative binomial before parasite
    induced mortality. 2) They used a GLM approach to calculate the the
    functional form of parasite induced mortality: a and b
    * This method relies on a few critical assumptions. 1) They show that 
    * Outline the traditional way to estimate parasite induced-mortality
    * Start with the historical approach.
        * Assumptions        
3. Results (MW + SW)
    * Simulation results (KP 4)
        * You can get the right answer for the wrong reasons.  You might think the method works but it doesn't.
    * 
4. Discussion (SW and MW)
    * Distinction between the assumptions and the limitations
    * Why is it not working?
    * Some new fangled approaches (Bayesian/Full Likelihood approach) 
        * Show why these also have some problems
    * Any other suggestions for estimating pi-mortality from distributional data alone?
    * Logic behind why it doesn't work- method fundamentally flawed (address historical critiques with age-intensity)(KP 3)  
    *  Other methods for estimating parasite induced mortality (KP 5)
       * Age -intensity data
          * When hosts can be classified by age, a decrease in average infection intensity (or decrease in the variance/mean ratio (Gordon to Rau 1982, fish and metacercaria)) in older hosts can be due to parasite induced mortality (critical assumption) and used to estimate loss due to infection.  
          * Examples include Henricson (1977) pleurocercoids in char 
          * Perrin and Power (1980) Crassicauda nematodes in dolphins (also see Balbueno et al 2014 for reanalysis of data), 
          * Cystidicola nematode in char (knudson 2002). 
          * this age intensity method has issues  (assumptions, etc)
    * (historical critiques)Both these intensity/distribution methods have been criticized previously because the observed patterns can be generated be a variety of mechanisms, not just parasite induced host mortality (McCallum 2000)
          * Rousett 1996 for simulations and others, eg pacala Dobson 1988…)
          * Anderson and Gordon 1982 for simulations of age impacts.  
          * Sampling error- older host groups frequently under sampled ()
          * Other mechanisms: host behavior impacts transmission, acquired immunity, etc
          * No way to check that estimates are correct from wild population data.
    * Other, (acceptable) ways to estimate parasite induced mortality: 
       * when wild animal populations can’t be manipulated, marked, or followed.
          * If infection induced mortality has been calculated in the lab, field data on parasite infections may be used to estimate wild animal mortality (Burreson 1984, Tiner 1954, more recent others)
          * Long term data sets that correlate host mortality and worm burden (eg Hudson et al 1992b).
          * Autopsy data on wild animals presumed to have died from the parasite can provide some information, but should only be considered if comparing these worm burdens to parasite loads in a random sample of the host population (lester 1984, Grenfell
    * Best ways to get at parasite induced host mortality
       * good: if wild populations can’t be manipulated but individuals can be followed, parasite impact on survival can be estimated by monitoring the survival of animals with a range of known infection intensities (Grenfell, examples).
       * better: treat a subset of wild animals in an infected pop and then monitor the survival differences between (presumably) parasitized and unparasitized hosts (Mc 2000, Grenfell, examples)
       * Best: In the absence of logistical/ethical constraints, the best method is to would be to capture animals from an uninfected population, infect (and sham infect the controls) and release them back into the wild and then use standard survival analyses to compare the infected and control groups (Mccallum 2000, Jaenike et al 1995).  
       


 


**Targets and TODOs**
1. Flesh out outline for each of our respective sections.  Saturday December 13th. 

