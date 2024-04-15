from HSC_x_DESI_RCR.hsc_desi_utils import *


wavelength_ratio = 0.01334 # 4000 km/s / c

wavelength_ratio_fitting = 0.01334*1.0 # 4,000 km/s / c


def make_line_dictionary(data, line_dictionary = line_dictionary):
        
    keys = [i for i in line_dictionary.keys()]

    dict = {}

    if len(data) == len(keys):
        for i in np.arange(len(data)):
            dict[keys[i]] = data[i]

        return dict 

    else:
        print('Input list not the same length as the default line dictionary.')



def cut_hiEW_line(waves, fluxes, key):
    
    wavelength = line_dictionary[key]

    lims = [np.min(waves), np.max(waves)]

    if in_wavelength_range(wavelength, lims):

        delta = wavelength_ratio * wavelength
        idx = ((waves < (wavelength - delta) ) | (waves > (wavelength + delta) )) 

        return waves[idx], fluxes[idx]
    
    else: 
        return waves,fluxes


def cut_all_hiEW_lines(waves, fluxes):

    lines = [i for i in line_dictionary.keys()]

    for i in lines:
        waves, fluxes = cut_hiEW_line(waves, fluxes, i)
    return waves, fluxes




def power_law_fit(x,y, sig = None):
    def func_powerlaw(x, m, c):
        return  (x**m * c)
    
    #result = model.fit(y, bins=binpoints, A = 1.)
    
    pars, cov = curve_fit(f=func_powerlaw, xdata=x, ydata=y, sigma= sig, p0=[-2, 100],maxfev=10000, absolute_sigma = True)

    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    res = y - func_powerlaw(x, *pars)
    
    return pars, stdevs, res



def fit_hiEW_line_sigma(waves, fluxes, key):

    def gauss_func(x, a, x0, sigma, C): 
        return a*np.exp(-(x-x0)**2/(2*sigma**2)) + C
  
    
    wavelength = line_dictionary[key]

    lims = [np.min(waves), np.max(waves)]

    if in_wavelength_range(wavelength, lims):

        delta = wavelength_ratio_fitting * wavelength

        idx = ((waves > (wavelength - delta) ) * (waves < (wavelength + delta) )) 

        popt = curve_fit(gauss_func, waves[idx], fluxes[idx], 
                         p0 = [1,wavelength,1, 0], maxfev = 10000)[0] # Grabbing only the parameters, second index = 1 is the covariance 
        # print(popt)
        # plt.figure()
        # plt.scatter(waves[idx], fluxes[idx])
        # ym = gauss_func(waves[idx], popt[0], popt[1], popt[2],popt[3] ) 
        # plt.plot(waves[idx], ym, c='r', label='Best fit') 

        sigma = np.abs(popt[2]) * 2.99e5/wavelength # convert to km/s

        return sigma
    
    else: 
        return 0.
    
    

def fit_all_hiEW_line_sigma(waves,fluxes):

    lines = [i for i in line_dictionary.keys()]

    sigmas = np.array([fit_hiEW_line_sigma(waves, fluxes, i) for i in lines ])

    sigmas = make_line_dictionary(sigmas)

    return sigmas




def fit_spectra_gamma(spec, tabrow):


    waves, fluxes = combine_spec_arms(spec)
    

    z = tabrow['Z']

    # make into restframe
    waves= waves / (1+z)

    # Keep a full version of the spectra with the spectral lines for later
    waves_full = np.copy(waves)
    fluxes_full = np.copy(fluxes)

    # Cut the high EW lines 
    waves, fluxes = cut_all_hiEW_lines(waves, fluxes)

    # Normalize at 3000 Angstroms (Or the closest we can get)
    norm_ind = np.where(np.abs(waves-3000) == np.min(np.abs(waves-3000)))[0]
    waves_n =  waves/waves[norm_ind]
    fluxes_n = fluxes/fluxes[norm_ind]

    # Fit a power law to the data: 
    alpha, A = power_law_fit(waves_n,fluxes_n)[0]


    # Now fit the spectral lines and get sigmas for the gaussians:
        # first we subtract the continuum by fitting a new power law without the normalization: 
    alpha_full, A_full = power_law_fit(waves,fluxes)[0]

        # the resulting power law:
    plaw= A_full* (waves_full**alpha_full)

    fluxes_full = fluxes_full-plaw # subtracting the continuum with the plaw model

        # Now we fit the gaussian sigma for the individual spectral features:
    
    sigmas = fit_all_hiEW_line_sigma(waves_full,fluxes_full)


    return alpha, A, sigmas





def fit_spectra_gamma_saved(spec, tabrow):


    waves, fluxes = spec
    

    z = tabrow['Z']

    # make into restframe
    waves= waves / (1+z)

    # Keep a full version of the spectra with the spectral lines for later
    waves_full = np.copy(waves)
    fluxes_full = np.copy(fluxes)

    # Cut the high EW lines 
    waves, fluxes = cut_all_hiEW_lines(waves, fluxes)

    # Normalize at 3000 Angstroms (Or the closest we can get)
    norm_ind = np.where(np.abs(waves-3000) == np.min(np.abs(waves-3000)))[0]
    waves_n =  waves/waves[norm_ind]
    fluxes_n = fluxes/fluxes[norm_ind]

    # Fit a power law to the data: 
    alpha, A = power_law_fit(waves_n,fluxes_n)[0]


    # Now fit the spectral lines and get sigmas for the gaussians:
        # first we subtract the continuum by fitting a new power law without the normalization: 
    alpha_full, A_full = power_law_fit(waves,fluxes)[0]

        # the resulting power law:
    plaw= A_full* (waves_full**alpha_full)

    fluxes_full = fluxes_full-plaw # subtracting the continuum with the plaw model

        # Now we fit the gaussian sigma for the individual spectral features:
    
    sigmas = fit_all_hiEW_line_sigma(waves_full,fluxes_full)


    return alpha, A, sigmas



