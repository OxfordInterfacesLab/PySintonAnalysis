import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def SRHfunction(taun, taup, Et_Ev, DeltaN, ni, n_0, p_0):
    """
    Calculate the SRH recombination rate.

    Parameters:
    taun (float): SRH lifetime for electrons
    taup (float): SRH lifetime for holes
    Et_Ev (float): Energy level of the trap relative to the valence band maximum
    DeltaN (float or array): Excess carrier density
    ni (float): Intrinsic carrier concentration
    n_0 (float): Equilibrium electron concentration
    p_0 (float): Equilibrium hole concentration

    Returns:
    F (float or array): SRH recombination rate
    """
    q = 1.6e-19
    Vt = (1.38e-23 * 300) / q

    Et = -(1.1 / 2 - Et_Ev)
    n_1 = ni * np.exp(Et / Vt)
    p_1 = ni * np.exp(-Et / Vt)
    F = (taup * (n_0 + n_1 + DeltaN) + taun * (p_0 + p_1 + DeltaN)) / (p_0 + n_0 + DeltaN)

    return F

def IntBulkLifetime(Ndop, Delta_n, waferT, ni, Temp):
    

    # Niewelt model
    # Excess concentration in stable region
    if Ndop > 0:
        p_0 = Ndop
        n_0 = ni**2 / p_0
    else:
        n_0 = -Ndop
        p_0 = ni**2 / n_0

    if n_0 > p_0:
        n_d = n_0 + Delta_n
        p_d = 0 + Delta_n
    else:
        n_d = 0 + Delta_n
        p_d = p_0 + Delta_n

    Blow = 4.76e-15  # [cm^3s^-1] Nguyen
    # Values from Altermatt in NUSOD'2005
    Bmin = 0.2 + (0 - 0.2) / (1 + (Temp / 320)**2.5)
    b1 = 2 * (1.5e18 + (1e7 - 1.5e18) / (1 + (Temp / 550)**3.0))
    b3 = 2 * (4e18 + (1e9 - 4e18) / (1 + (Temp / 365)**3.54))
    Brel = Bmin + (1.00 - Bmin) / (1 + ((n_d + p_d) / b1)**0.54 + ((n_d + p_d) / b3)**1.25)

    # Recalculating for bandgap narrowing
    nieff = ni**2 / Brel
    if n_0 > p_0:
        n0eff = n_0
        p0eff = nieff / n_0
    else:
        p0eff = p_0
        n0eff = nieff / p_0
        
    n_deff = p0eff + Delta_n
    p_deff = n0eff + Delta_n

    B = Brel * Blow  # [cm^3s^-1] Radiative recombination probability

    # Photon recycling
    fPR = 0.9835 + 0.006841 * np.log10(waferT) - 4.554e-9 * Delta_n**0.4612

    geeh = 1 + (4.38 - 1) / (1 + ((n_d + p_d) / 4e17)**2)
    gehh = 1 + (4.88 - 1) / (1 + ((n_d + p_d) / 4e17)**2)

    R_Auger = (3.41e-31 * geeh * ((n_d**2 * p_d) - (n0eff**2 * p0eff)) +
               1.17e-31 * gehh * ((n_d * p_d**2) - (n0eff * p0eff**2)))

    t_Aug = Delta_n / R_Auger

    R_rad = ((n_d * p_d - nieff) * (B * (1 - fPR)))

    t_Rad = Delta_n / R_rad

    return t_Aug, t_Rad, np.sqrt(nieff), p0eff, n0eff, n_deff, p_deff 


def extract_srv(df, plot_option, srh_fit_options):
    """
    This Function extracts all the recombination parameters from measurements
    of effective lifetime using a Sinton Instrument, and plots these
    parameters as a function of minority carrier density
    """
    
    q=1.6e-19  #e charge in C
    Temp = 300  # K Temperature
    Vt=(1.38e-23*Temp)/q
    ni=9.65409e9;#1/cm3 Semicon intrinsic carrier consentration,Altermatt JAP 2003 
    
    
    silicon=sp.io.loadmat('SiliconData.mat')
    
    # get all the values
    
    
    waf_thick=df.iloc[0,0]
    resistivity=df.iloc[1,0]
    si_type=df.iloc[2,0]
    delta_n=df.iloc[3,0]
    tau_eff=df.iloc[4,0]
    sample_name=df.columns[0]
    
    # srh_fit_options=np.array([0,0,0,0])# tn,tp,Et-Ev,J0
    
    if si_type == 'n-type':
        res_table = silicon['res_n']
        diffusion_table = silicon['diff_n']
    elif si_type == 'p-type':
        res_table = silicon['res_p']
        diffusion_table = silicon['diff_p']
    else:
        raise ValueError('Type of substrate not well defined')
        
    ndop = np.interp(resistivity, np.flip(res_table[:, 1]), np.flip((res_table[:, 0])))
    
    
    #first smooth the data
    tau_eff_sm =sp.signal.savgol_filter(tau_eff,50,4);
    
    #then make the Teff-Dn data set smaller, len 40
    delta_n2 = np.logspace(np.log10(np.min(delta_n)), np.log10(np.max(delta_n)), 40)
    tau_eff2 = sp.interpolate.interp1d(delta_n, tau_eff_sm, kind='linear',fill_value="extrapolate")(delta_n2)
    
    t_Aug, t_Rad, nieff, p_0, n_0, n_d, p_d  = IntBulkLifetime(ndop, delta_n2, waf_thick, ni, Temp)
    t_bulk=1/(1/t_Rad + 1/t_Aug) #bulk lifetime
    
    # Rearrange the table with all diffusivities
    dummy = np.transpose(diffusion_table, (0, 2, 1))
    
    # Interpolation to find the ambipolar diffusion coefficient as f of Delta_n
    # (('Dn','Ndop_table'),'Diff_Coef_table',('delta_n2','ndop'))
    Damb=sp.interpolate.interpn((diffusion_table[1:202, 0, 0],dummy[0, :, 0]),dummy[1:202, :, 8],(delta_n2, np.abs(ndop)))
    
    # SRV with infinite SRH lifetime
    Seff_1 = np.sqrt(Damb * (1/tau_eff2 - 1/t_bulk)) * np.tan((waf_thick/2) * np.sqrt((1/Damb) * (1/tau_eff2 - 1/t_Rad - 1/t_Aug)))
    
    # Approx 6 Mackel pip.2167
    # Calculating the gradient
    Joe6= waf_thick * q * nieff**2 * np.gradient(1/tau_eff2 - 1/t_Rad - 1/t_Aug,delta_n2)
    
    # now we will start the algorigthm to obtain more accurate values of J0 and S
    #Joe calculated from Kimmerle SOLMAT 142 (2015) 116–122
    
    
    # We need to fit an SRH term. variables: TauN, TauP, Et-Ev
    SRH_0 = np.max(1 / (1/tau_eff2 - 1/t_bulk)) * np.array([10, 10, 0]) + np.array([0, 0, 0.56])  # Starting point 10x above tsurf
    SRH_lb = np.max(1 / (1/tau_eff2 - 1/t_bulk))*np.array([1,1, 0]) # Lower boundary
    SRH_ub = np.max(1 / (1/t_bulk))*np.array([1.2, 1.2, 0])+ np.array([0, 0, 1.12])   # Upper boundary
    
    if srh_fit_options[0] !=0: SRH_lb[0]=0.99*srh_fit_options[0];SRH_ub[0]=1.01*srh_fit_options[0];SRH_0[0]=srh_fit_options[0];
    if srh_fit_options[1] !=0: SRH_lb[1]=0.99*srh_fit_options[1];SRH_ub[1]=1.01*srh_fit_options[1];SRH_0[1]=srh_fit_options[1];
    if srh_fit_options[2] !=0: SRH_lb[2]=0.99*srh_fit_options[2];SRH_ub[2]=1.01*srh_fit_options[2];SRH_0[2]=srh_fit_options[2];
    
    
    
    # Iterate over the algorithm n times
    for i in range(5):  # 5 iterations
    
        # Calculate Joe2 with intrinsic bulk lifetime and a very low SRH recomb
        tau_surf=1/(1/tau_eff2 - 1/t_bulk - 1 / SRHfunction(SRH_0[0], SRH_0[1], SRH_0[2], delta_n2, nieff, n_0, p_0))
            
        Joe2 = q * np.gradient(nieff**2 * np.sqrt(Damb * (1/tau_surf)) 
            * np.tan((waf_thick / 2) * np.sqrt((1 / Damb) * (1/tau_surf))),delta_n2)
       
        Joe2_pos = Joe2[Joe2 > 0]
        delta_n2_pos = delta_n2[Joe2 > 0]   
        
        # Check if DeltaN_pos is empty
        if len(delta_n2_pos) == 0:  
            
            print('Surface recombination is so high J_0 must be calculated from Seff - Error')  
            # Recalculate Joe2 from the easier expression without derivative
            Joe2 = np.sqrt(Damb * (1/tau_surf)) * np.tan((waf_thick/2) * np.sqrt((1/Damb) * (1/tau_surf))) * (q * nieff**2) / (np.abs(ndop) + delta_n2)
            # Update Joe2_pos and DeltaN_pos to only account for positive values of Joe2
            Joe2_pos = Joe2[Joe2 > 0]
            delta_n2_pos = delta_n2[Joe2 > 0]
            
        # calculate grad(J0) and only analise deta for deltaN>0.5Ndop
        GradJ0e2=np.abs(np.gradient(Joe2_pos[delta_n2_pos>0.5*np.abs(ndop)],delta_n2_pos[delta_n2_pos>0.5*np.abs(ndop)]))
        sub_Joe2_pos=Joe2_pos[delta_n2_pos>0.5*np.abs(ndop)]
        Joe_avg=np.mean(sub_Joe2_pos[np.where(GradJ0e2<np.min(GradJ0e2)*1.2)])#take the mean of the lowest 20% values
        
        if srh_fit_options[3] !=0: Joe_avg=srh_fit_options[3]; # this is in case the options defined a static Joe
            
        def ObjectiveFun(x,delta_n2,tau_eff2,t_Rad,t_Aug,nieff,n_0, p_0,ndop,Joe_avg):
            tau_surf=1/ (1/tau_eff2 - 1/t_bulk - 1 /(SRHfunction(x[0], x[1], x[2], delta_n2, nieff, n_0, p_0)))
            term1 = (Joe_avg * (np.abs(ndop) + delta_n2)) / (q * nieff**2)
            term2 = np.sqrt(Damb * (1/tau_surf)) * np.tan((waf_thick/2) * np.sqrt((1/Damb) * (1/tau_surf)))
            return term1 - term2
    
        # Perform the optimization
        result = sp.optimize.least_squares(ObjectiveFun, SRH_0, bounds=(SRH_lb, SRH_ub),loss='linear', method='trf',
                                           args=(delta_n2,tau_eff2,t_Rad,t_Aug,nieff,n_0, p_0,ndop,Joe_avg))
        
        SRH_I = result.x
        #now run the algorithm again, but refresh the starting point of SRH
        SRH_0=SRH_I
    
    print(sample_name)
    print(result.message)
    print('The values of tn [s], tp [s], Et-Ev [eV], J0 [A/cm2] are')
    print([result.x,Joe_avg])
 
    #calculate the SRH bulk lifetime of the optimally found values
    tau_SRH=SRHfunction(SRH_0[0], SRH_0[1], SRH_0[2], delta_n2, nieff, n_0, p_0)
    #calculate the iVoc of from the lifetime
    iVoc = Vt * np.log((delta_n2 * (np.abs(ndop) + delta_n2)) / nieff**2)
    
    mse = ((tau_eff2 - (1/(1/t_bulk + (2*Joe_avg*(np.abs(ndop)+delta_n2)/(waf_thick*q*nieff**2)) + 1/tau_SRH)))**2).mean(axis=0)
    print('Mean squared error of fit is: '+ "{:10.4e}".format(mse) + '\n')
    
    
    # SRV with estimated SRH lifetime using algorithm
    tau_surf=1/(1/tau_eff2 - 1/t_bulk - 1 / SRHfunction(SRH_0[0], SRH_0[1], SRH_0[2], delta_n2, nieff, n_0, p_0))
    Seff_2 = np.sqrt(Damb * (1/tau_surf)) * np.tan((waf_thick/2) * np.sqrt((1/Damb) * (1/tau_surf)))
    Seff_3 = Joe_avg * (np.abs(ndop) + delta_n2) / (q * nieff**2)
    Jeo1 = Seff_1 * (q * nieff**2) / (np.abs(ndop) + delta_n2)
    
    
    #some ploting to check it's all good
    if plot_option==1:
 
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))  # Adjust the figsize as needed

        # Subplot 1
        axs[0, 0].loglog(delta_n, tau_eff, 'o', delta_n2, t_bulk, '-')
        axs[0, 0].loglog(delta_n2, 1/(1/t_bulk + (2*Joe_avg*(np.abs(ndop)+delta_n2)/(waf_thick*q*nieff**2)) + 1/tau_SRH), '-')
        axs[0, 0].loglog(delta_n2, tau_SRH, '-')
        axs[0, 0].loglog(delta_n2, 1/(2*Joe_avg*(np.abs(ndop)+delta_n2)/(waf_thick*q*nieff**2)), '-')
        axs[0, 0].loglog([np.abs(ndop), np.abs(ndop)], [np.min(tau_eff), np.max(tau_eff)], '--')
        axs[0, 0].legend(['Exp Data', 'Tau_intrinsic', 'Tau_best fit', 'Tau_SRH', 'Tau_SurfaceJ_0S', 'N_doping'],
                         bbox_to_anchor=(1.05,1), loc='upper left')
        axs[0, 0].set_xlabel(r'$\Delta n\ (cm^{-3})$')
        axs[0, 0].set_ylabel(r'$\tau\ (s)$')
        axs[0, 0].set_xlim([1e13, 2e16])
        
        # Subplot 2
        axs[0, 1].loglog(delta_n2, Seff_1, 'o', delta_n2, Seff_2, '-', delta_n2, Seff_3, '.-')
        axs[0, 1].legend(['S_eff1 from Tau_eff (Tau_SRH=∞)', 'S_eff2 when finite tau_SRH', 'S_eff3 from J_0s-avg'],
                         bbox_to_anchor=(1.05,1), loc='upper left')
        axs[0, 1].set_xlabel(r'$\Delta n\ (cm^{-3})$')
        axs[0, 1].set_ylabel(r'$S_{eff}  (cm/s)$')
        
        # Subplot 3
        axs[1, 0].loglog(delta_n2, 1e15*Joe6, 'o', delta_n2, 1e15*Joe2, '^', delta_n2, 1e15*Jeo1)
        axs[1, 0].loglog([1e13, 2e16], 1e15*np.array([Joe_avg, Joe_avg]))
        axs[1, 0].legend(['J_0s-6 Mackel', 'J_0s-2 Kimmerle', 'J_0s from S_eff1', 'J_0s Avg'],
                         bbox_to_anchor=(1.05,1), loc='upper left')
        axs[1, 0].set_xlabel(r'$\Delta n\ (cm^{-3})$')
        axs[1, 0].set_ylabel(r'$J_{0s}  (fA/cm^2)$')
        
        # Subplot 4
        axs[1, 1].semilogx(delta_n2, iVoc, 'o')
        axs[1, 1].set_xlabel(r'$\Delta n\ (cm^{-3})$')
        axs[1, 1].set_ylabel(r'$iV_{oc} (V)$')
        
        plt.tight_layout()
        plt.show()
    
    
    df.loc[5,sample_name]=delta_n2
    df.loc[6,sample_name]=t_bulk
    df.loc[7,sample_name]=nieff
    df.loc[8,sample_name]=ndop
    df.loc[9,sample_name]=Joe_avg
    df.loc[10,sample_name]=tau_SRH
    df.loc[11,sample_name]=[[Seff_1],[Seff_2],[Seff_3]]
    df.loc[12,sample_name]=[[Joe6],[Joe2],[Jeo1]]
    df.loc[13,sample_name]=iVoc
    df.loc[14,sample_name]=tau_eff2
    df.loc[15,sample_name]=SRH_0
    df.loc[16,sample_name]=mse
    
    
    return df


def save_file(results_df, file_name):
    
    
    # extracting all variables and defining by name
    sample_name=results_df.columns[0]
    delta_n=results_df.iloc[3,0]
    tau_eff=results_df.iloc[4,0]
    waf_thick=results_df.iloc[0,0]
    delta_n2=results_df.loc[5,sample_name]
    t_int=results_df.loc[6,sample_name]
    nieff=results_df.loc[7,sample_name]
    ndop=results_df.loc[8,sample_name]
    Joe_avg=results_df.loc[9,sample_name]
    tau_SRH=results_df.loc[10,sample_name]
    [[Seff_1],[Seff_2],[Seff_3]]=results_df.loc[11,sample_name]
    [[Joe6],[Joe2],[Joe1]]=results_df.loc[12,sample_name]
    iVoc=results_df.loc[13,sample_name]
    SRH_parms=results_df.loc[15,sample_name]
    mse=results_df.loc[16,sample_name]

    f=open(file_name,'w')

    f.write("delta_n, tau_eff \n");np.savetxt(f, delta_n, newline=", ");f.write("\n")
    np.savetxt(f, tau_eff, newline=", ");f.write("\n")
    f.write("waf_thick,");f.write(np.array2string(waf_thick));f.write("\n")
    f.write("Ndop,");f.write(np.array2string(ndop));f.write("\n")
    f.write("J0 average,");f.write(np.array2string(Joe_avg));f.write("\n")

    f.write("delta_n2 \n");np.savetxt(f, delta_n2, newline=", ");f.write("\n")
    f.write("t_intrinsic \n");np.savetxt(f, t_int, newline=", ");f.write("\n")
    f.write("tau_SRH \n");np.savetxt(f, tau_SRH, newline=", ");f.write("\n")
    f.write("Seff_1 \n");np.savetxt(f, Seff_1, newline=", ");f.write("\n")
    f.write("Seff_2 \n");np.savetxt(f, Seff_2, newline=", ");f.write("\n")
    f.write("Seff_3 \n");np.savetxt(f, Seff_3, newline=", ");f.write("\n")
    f.write("Joe1 \n");np.savetxt(f, Joe1, newline=", ");f.write("\n")
    f.write("Joe2 \n");np.savetxt(f, Joe2, newline=", ");f.write("\n")
    f.write("Joe6 \n");np.savetxt(f, Joe6, newline=", ");f.write("\n")
    f.write("iVoc \n");np.savetxt(f, iVoc, newline=", ");f.write("\n")
    f.write("SRH parameters (tn, tp, E_t \n");np.savetxt(f, SRH_parms, newline=", ");f.write("\n")
    f.write("Mean Squared Error of best fit \n");np.savetxt(f, [[mse]], newline=", ");f.write("\n")


    f.close()
    

    """DataFrame: index is sample name, then each row is
    0 - wafer thickness in cm
    1 - resistiity in ohm cm
    2 - silicon type
    3 - Delta_n
    4 - Teff from Sinton
    5 - delta_n2
    6 - t_int
    7 - nieff
    8 - ndop
    9 - Joe_avg
    10 - tau_SRH
    11 - [[Seff_1],[Seff_2],[Seff_3]]
    12 - [[Joe6],[Joe2],[Jeo1]]
    13 - iVoc
    14 - teff_2 (reduced and filtered lifetime)
    """

    return


    


    

# matplotlib.pyplot.loglog(delta_n,tau_eff,"o-")
# matplotlib.pyplot.loglog(delta_n2,tau_eff2,"o-")

# matplotlib.pyplot.loglog(delta_n2,tau_eff2,"o-");matplotlib.pyplot.loglog(delta_n2,1/(1/t_Aug+1/t_Rad),"o-")
# matplotlib.pyplot.loglog(delta_n2,Joe6,"o-");matplotlib.pyplot.loglog(delta_n2,Joe2,"o-")



