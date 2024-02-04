# PySintonAnalysis

To make sure it is all clear, here are all the main equations that define how these parameters are analysed.
### See more on
- RS Bonilla, et al (2017), Phys. Status Solidi A, 214: 1700293
- A. Kimmerle, J. Greulich, and A. Wolf, Sol. Energy Mater. Sol. Cells 142, 116–122 (2015).
- H. M€ackel and K. Varner, Prog. Photovolt. Res. Appl. 21, 850–866 (2013).
- B. Hammann et al., IEEE Journal of Photovoltaics, vol. 13, no. 4, pp. 535-546, July 2023.)

Starting from the effective lifetime measured in a Sinton tester 

$\frac{1}{\tau_{eff}}=\frac{1}{\tau_{bulk_int}}+\frac{1}{\tau_{SRH}}+\frac{1}{\tau_{surf}} $.

Effective recombinaiton velocity is: $S_{eff}=\sqrt{D(\frac{1}{\tau_{surf}})}*\tan(W/2\sqrt{(\frac{1}{D\tau_{surf}})})$.  Which is often approximated to $S_{eff}=W/2(\frac{1}{\tau_{surf}})$

Then, J0 can be estimate from the average of the flat region of a plot of: $\frac{\partial}{\partial \Delta n}(n_i^2(S_{eff}))$ which requires the algorithm implemented for getting $J_{0S-avg}$, since we need to calculate an SRH recombination that makes  $J_{0s-avg}=\frac{qn_i^2S_{eff}}{N_{dop}+\Delta n}$ a true equation. Here $J_0$ would be a single value parameter, but it is of course possible to simply calculate S and then obtain a J0 as a function of minority carriers. 

Some useful advice for extracting J0:
 
- Start by letting the software fit the tn, tp, and E_t values.
- Check that the fit makes sense in the figure. Remember this code does not fit lifetime, it finds the SRH term that is most compatible with a carrier-independent J_0s average term, as per Kimmerle's method.
- If you are doing many samples at the same time, and you cannot check all of them, start on the assumption that they all have the same bulk lifetime.
- This means setting a strict value of the SRH bulk via tn, tp, and E_t.
- Often you can find out how much defect mediated recombination sample have based on a sample with reasonable passivation (J0s<10 fA/Cm2)
- Then use this to fit the best possible SRH, and analyse all samples with a single value of SRH bulk.
- Conversely, if your processes aim to change the bulk, but you have a good confidence of the surface passivation, then set the value of J0s and allow the fitted to find the best recombination parameters it can find. 
