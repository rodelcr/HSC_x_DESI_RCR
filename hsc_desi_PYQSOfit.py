from HSC_x_DESI_RCR.hsc_desi_spectra import *

import glob, os, sys, timeit

import numpy as np

#sys.path.append('../')
from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import warnings

from glob import glob 





path_ex = '.'
path_out = 'QSOFit_out/.'

#if not os.path.exists('qsopar.fits'):


hdr0 = fits.Header()

hdr0['Author'] = 'Rodrigo Cordova'
hdr0['Code'] = 'PyQSOFit'
primary_hdu = fits.PrimaryHDU(header=hdr0)

"""
Create parameter file
lambda    complexname  minwav maxwav linename ngauss inisca minsca maxsca inisig minsig maxsig voff vindex windex findex fvalue vary
"""

c = 2.998e5 #km/s

# broadlo = 900/np.sqrt(8*np.log(2))/c # FWHM in km/s turned into sigma/c 
# broadhi = 7000/np.sqrt(8*np.log(2))/c # FWHM in km/s turned into sigma/c 

# narrowlo = 50/np.sqrt(8*np.log(2))/c # FWHM in km/s turned into sigma/c 
# narrowhi = 800/np.sqrt(8*np.log(2))/c # FWHM in km/s turned into sigma/c 


broadlo = 400/np.sqrt(8*np.log(2))/c # FWHM in km/s turned into sigma/c 
broadhi = 7000/np.sqrt(8*np.log(2))/c # FWHM in km/s turned into sigma/c 

narrowlo = 10/np.sqrt(8*np.log(2))/c # FWHM in km/s turned into sigma/c 
narrowhi = 300/np.sqrt(8*np.log(2))/c # FWHM in km/s turned into sigma/c 



newdata = np.rec.array([
(6564.61, 'Ha', 6400, 6800, 'Ha_br',   1, 0.0, 0.0, 1e10, 5e-3, broadlo,  broadhi,   0.01, 0, 0, 0, 0.05 , 1),
(6564.61, 'Ha', 6400, 6800, 'Ha_na',   1, 0.0, 0.0, 1e10, 5.2e-4,narrowlo,narrowhi, 0.001, 1, 1, 0, 0.002, 1),
(6549.85, 'Ha', 6400, 6800, 'NII6549', 1, 0.0, 0.0, 1e10, 5.2e-4,narrowlo,narrowhi, 5e-3,  1, 1, 0, 0.001, 1),
(6585.28, 'Ha', 6400, 6800, 'NII6585', 1, 0.0, 0.0, 1e10, 5.2e-4,narrowlo, narrowhi, 5e-3,  1, 1, 0, 0.003, 1),
(6718.29, 'Ha', 6400, 6800, 'SII6718', 1, 0.0, 0.0, 1e10, 1e-3, narrowlo, narrowhi, 5e-3,  1, 1, 2, 0.001, 1),
(6732.67, 'Ha', 6400, 6800, 'SII6732', 1, 0.0, 0.0, 1e10, 1e-3, narrowlo, narrowhi, 5e-3,  1, 1, 2, 0.001, 1),
    
(4862.68, 'Hb', 4640, 5100, 'Hb_br',     1, 0.0, 0.0, 1e10, 5e-3, broadlo,  broadhi,   0.01, 0, 0, 0, 0.01 , 1),
(4862.68, 'Hb', 4640, 5100, 'Hb_na',     1, 0.0, 0.0, 1e10, 1e-3, narrowlo,narrowhi, 0.001, 1, 1, 0, 0.002, 1),
#(4960.30, 'Hb', 4640, 5100, 'OIII4959c', 1, 0.0, 0.0, 1e10, 5e-4, 2.3e-4, 0.00169, 0.001, 1, 1, 0, 0.002, 1),
#(5008.24, 'Hb', 4640, 5100, 'OIII5007c', 1, 0.0, 0.0, 1e10, 5e-4, 2.3e-4, 0.00169, 0.001, 1, 1, 0, 0.004, 1),
(4960.30, 'Hb', 4640, 5100, 'OIII4959w',   1, 0.0, 0.0, 1e10, 3e-3, narrowlo, narrowhi,  0.01,  2, 2, 0, 0.001, 1),
(5008.24, 'Hb', 4640, 5100, 'OIII5007w',   1, 0.0, 0.0, 1e10, 3e-3, narrowlo, narrowhi,  0.01,  2, 2, 0, 0.002, 1),
# (4687.02, 'Hb', 4640, 5100, 'HeII4687_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.005, 0, 0, 0, 0.001, 1),
# (4687.02, 'Hb', 4640, 5100, 'HeII4687_na', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.005, 1, 1, 0, 0.001, 1),

(4340, 'Hg', 4300, 4400, 'Hg_br',     1, 0.0, 0.0, 1e10, 5e-3, broadlo,  broadhi,   0.01, 0, 0, 0, 0.01 , 1),
(4340, 'Hg', 4300, 4400, 'Hg_na',     1, 0.0, 0.0, 1e10, 1e-3, narrowlo, narrowhi, 0.001, 1, 1, 0, 0.002, 1),

(4101.7, 'Hd', 4050, 4150, 'Hd_br',     1, 0.0, 0.0, 1e10, 5e-3, broadlo,  broadhi,   0.01, 0, 0, 0, 0.01 , 1),
(4101.7, 'Hd', 4050, 4150, 'Hd_na',     1, 0.0, 0.0, 1e10, 1e-3, narrowlo, narrowhi, 0.001, 1, 1, 0, 0.002, 1),


#(3934.78, 'CaII', 3900, 3960, 'CaII3934' , 2, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 99, 0, 0, -0.001, 1),

(3728.48, 'OII', 3650, 3800, 'OII3728', 1, 0.1, 0.0, 1e10, 1e-3, narrowlo, narrowhi, 0.01, 1, 1, 0, 0.001, 1),

(3426.84, 'NeV', 3380, 3480, 'NeV3426',    1, 0.1, 0.0, 1e10, 1e-3, narrowlo, narrowhi, 0.01, 0, 0, 0, 0.001, 1),

(2798.75, 'MgII', 2700, 2900, 'MgII_br', 1, 0.0, 0.0, 1e10, 5e-3, broadlo,  broadhi, 0.015, 0, 0, 0, 0.05, 1),
(2796.35, 'MgII', 2700, 2900, 'MgII_na1', 1, 0.0, 0.0, 1e10, 1e-3,narrowlo, narrowhi, 0.01, 1, 1, 0, 0.002, 1),
(2803.53, 'MgII', 2700, 2900, 'MgII_na2', 1, 0.0, 0.0, 1e10, 1e-3,narrowlo, narrowhi, 0.01, 1, 1, 0, 0.002, 1),

# (1908.73, 'CIII', 1700, 1970, 'CIII_br',   2, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 99, 0, 0, 0.01, 1),
#(1908.73, 'CIII', 1700, 1970, 'CIII_na',   1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01,  1, 1, 0, 0.002, 1),
#(1892.03, 'CIII', 1700, 1970, 'SiIII1892', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
#(1857.40, 'CIII', 1700, 1970, 'AlIII1857', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
#(1816.98, 'CIII', 1700, 1970, 'SiII1816',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
#(1786.7,  'CIII', 1700, 1970, 'FeII1787',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
#(1750.26, 'CIII', 1700, 1970, 'NIII1750',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),
#(1718.55, 'CIII', 1700, 1900, 'NIV1718',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),

(1549.06, 'CIV', 1500, 1700, 'CIV_br', 1, 0.1, 0.0, 1e10, 5e-3, broadlo,  broadhi,   0.015, 0, 0, 0, 0.05 , 1),
(1549.06, 'CIV', 1500, 1700, 'CIV_na', 1, 0.1, 0.0, 1e10, 1e-3,narrowlo,  narrowhi, 0.01,  1, 1, 0, 0.002, 1),
# (1640.42, 'CIV', 1500, 1700, 'HeII1640',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
# (1663.48, 'CIV', 1500, 1700, 'OIII1663',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
# (1640.42, 'CIV', 1500, 1700, 'HeII1640_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),
# (1663.48, 'CIV', 1500, 1700, 'OIII1663_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),

#(1402.06, 'SiIV', 1290, 1450, 'SiIV_OIV1', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
#(1396.76, 'SiIV', 1290, 1450, 'SiIV_OIV2', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
#(1335.30, 'SiIV', 1290, 1450, 'CII1335',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),
#(1304.35, 'SiIV', 1290, 1450, 'OI1304',    1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),

(1215.67, 'Lya', 1150, 1290, 'Lya_br', 1, 0.1, 0.0, 1e10, 5e-3, broadlo,  broadhi,   0.02, 0, 0, 0, 0.05 , 1),
(1215.67, 'Lya', 1150, 1290, 'Lya_na', 1, 0.1, 0.0, 1e10, 1e-3,narrowlo,  narrowhi, 0.01, 0, 0, 0, 0.002, 1)
],
formats = 'float32,      a20,  float32, float32,      a20,  int32, float32, float32, float32, float32, float32, float32, float32,   int32,  int32,  int32,   float32, int32',
names  =  ' lambda, compname,   minwav,  maxwav, linename, ngauss,  inisca,  minsca,  maxsca,  inisig,  minsig,  maxsig,  voff,     vindex, windex,  findex,  fvalue,  vary')

# Header
hdr1 = fits.Header()
hdr1['lambda'] = 'Vacuum Wavelength in Ang'
hdr1['minwav'] = 'Lower complex fitting wavelength range'
hdr1['maxwav'] = 'Upper complex fitting wavelength range'
hdr1['ngauss'] = 'Number of Gaussians for the line'

# Can be set to negative for absorption lines if you want
hdr1['inisca'] = 'Initial guess of line scale [flux]'
hdr1['minsca'] = 'Lower range of line scale [flux]'
hdr1['maxsca'] = 'Upper range of line scale [flux]'

hdr1['inisig'] = 'Initial guess of linesigma [lnlambda]'
hdr1['minsig'] = 'Lower range of line sigma [lnlambda]'
hdr1['maxsig'] = 'Upper range of line sigma [lnlambda]'

hdr1['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr1['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr1['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr1['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr1['fvalue'] = 'Relative scale factor for entries w/ same findex'

hdr1['vary'] = 'Whether or not to vary the line parameters (set to 0 to fix the line parameters to initial values)'

# Save line info
hdu1 = fits.BinTableHDU(data=newdata, header=hdr1, name='data')


"""
In this table, we specify the windows and priors / initial conditions and boundaries for the continuum fitting parameters.
"""

conti_windows = np.rec.array([
    (1150., 1170.), 
    (1275., 1290.),
    (1350., 1360.),
    (1445., 1465.),
    (1690., 1705.),
    (1770., 1810.),
    (1970., 2400.),
    (2480., 2675.),
    (2925., 3400.),
    (3500., 3650.),
    (3775., 3832.),
    (4000., 4050.),
    (4200., 4230.),
    (4435., 4640.),
    (5100., 5535.),
    (6005., 6035.),
    (6110., 6250.),
    (6800., 7000.),
    (7160., 7180.),
    (7500., 7800.),
    (8050., 8150.), # Continuum fitting windows (to avoid emission line, etc.)  [AA]
    ], 
    formats = 'float32,  float32',
    names =    'min,     max')


hdu2 = fits.BinTableHDU(data=conti_windows, name='conti_windows')

conti_priors = np.rec.array([
    ('Fe_uv_norm',  0.0,   0.0,   1e10,  1), # Normalization of the MgII Fe template [flux]
    ('Fe_uv_FWHM',  3000,  1200,  18000, 1), # FWHM of the MgII Fe template [AA]
    ('Fe_uv_shift', 0.0,   -0.01, 0.01,  1), # Wavelength shift of the MgII Fe template [lnlambda]
    ('Fe_op_norm',  0.0,   0.0,   1e10,  1), # Normalization of the Hbeta/Halpha Fe template [flux]
    ('Fe_op_FWHM',  3000,  1200,  18000, 1), # FWHM of the Hbeta/Halpha Fe template [AA]
    ('Fe_op_shift', 0.0,   -0.01, 0.01,  1), # Wavelength shift of the Hbeta/Halpha Fe template [lnlambda]
    ('PL_norm',     1.0,   0.0,   1e10,  1), # Normalization of the power-law (PL) continuum f_lambda = (lambda/3000)^-alpha
    ('PL_slope',    -1.5,  -5.0,  3.0,   1), # Slope of the power-law (PL) continuum
    ('Blamer_norm', 0.0,   0.0,   1e10,  1), # Normalization of the Balmer continuum at < 3646 AA [flux] (Dietrich et al. 2002)
    ('Balmer_Te',   15000, 10000, 50000, 1), # Te of the Balmer continuum at < 3646 AA [K?]
    ('Balmer_Tau',  0.5,   0.1,   2.0,   1), # Tau of the Balmer continuum at < 3646 AA
    ('conti_a_0',   0.0,   None,  None,  1), # 1st coefficient of the polynomial continuum
    ('conti_a_1',   0.0,   None,  None,  1), # 2nd coefficient of the polynomial continuum
    ('conti_a_2',   0.0,   None,  None,  1), # 3rd coefficient of the polynomial continuum
    # Note: The min/max bounds on the conti_a_0 coefficients are ignored by the code,
    # so they can be determined automatically for numerical stability.
    ],

    formats = 'a20,  float32, float32, float32, int32',
    names = 'parname, initial,   min,     max,     vary')

hdr3 = fits.Header()
hdr3['ini'] = 'Initial guess of line scale [flux]'
hdr3['min'] = 'FWHM of the MgII Fe template'
hdr3['max'] = 'Wavelength shift of the MgII Fe template'

hdr3['vary'] = 'Whether or not to vary the parameter (set to 0 to fix the continuum parameter to initial values)'


hdu3 = fits.BinTableHDU(data=conti_priors, header=hdr3, name='conti_priors')


"""
In this table, we allow user to customized some key parameters in our result measurements.
"""

measure_info = Table(
    [
        [[1350, 1450, 3000, 4200, 5100]],
        [[
            [2240, 2650], 
            [4435, 4685],
        ]]
    ],
    names=([
        'cont_loc',
        'Fe_flux_range'
    ]),
    dtype=([
        'float32',
        'float32'
    ])
)
hdr4 = fits.Header()
hdr4['cont_loc'] = 'The wavelength of continuum luminosity in results'
hdr4['Fe_flux_range'] = 'Fe emission wavelength range calculated in results'

hdu4 = fits.BinTableHDU(data=measure_info, header=hdr4, name='measure_info')

hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4])
hdu_list.writeto('qsopar.fits', overwrite=True)


# # Now that we made the priors, let's make the qso fit function 

# import importlib
# importlib.reload(sys.modules['PyQSOFit.pyqsofit.PyQSOFit'])
# #importlib.reload(sys.modules['HSC_x_DESI_RCR.hsc_desi_PYQSOfit'])
# from PyQSOFit.pyqsofit.PyQSOFit import *
# from glob import glob 


def essential_plotting_components_from_self(self):
    pp = list(self.conti_fit.params.valuesdict().values())
    mc_flag = 2
    broad_fwhm=1200

    ncomp_fit = len(self.fur_result) // (
            mc_flag * 6)  # TODO: Not 5 here. But better not use absolute value to fully fix this bug




    wave_eval = np.linspace(np.min(self.wave) - 200, np.max(self.wave) + 200, 5000)
    
    lines_total = np.zeros_like(wave_eval)

    f_conti_model_eval = self.PL(wave_eval, pp) + self.Fe_flux_mgii(wave_eval, pp[0:3]) + self.Fe_flux_balmer(
            wave_eval, pp[3:6]) + self.F_poly_conti(wave_eval, pp[11:]) + self.Balmer_conti(wave_eval, pp[8:11])


    for p in range(len(self.gauss_result) // (mc_flag * 3)):
        gauss_result_p = self.gauss_result[p * 3 * mc_flag:(p + 1) * 3 * mc_flag:mc_flag]

        # Broad or narrow line check
        if self.CalFWHM(self.gauss_result[(2 + p * 3) * mc_flag]) < broad_fwhm:
            # Narrow
            color = 'g'
            self.f_line_narrow_model += self.Onegauss(np.log(self.wave), gauss_result_p)
        else:
            # Broad
            color = 'r'
            self.f_line_br_model += self.Onegauss(np.log(self.wave), gauss_result_p)

        # Evaluate the line component
        line_single = self.Onegauss(np.log(wave_eval), gauss_result_p)
        # self.f_line_model += self.Onegauss(np.log(wave), gauss_result_p)

        # Plot the line component
        for c in range(ncomp_fit):
            #axn[1][c].plot(wave_eval, line_single, color=color, zorder=line_order[color]) 

        lines_total += line_single

    out = {'wave':self.wave,
           'f_conti_model_eval':f_conti_model_eval,
           'flux_prereduced':self.flux_prereduced,
           'f_line_model':self.f_line_model,
           'f_conti_model':self.f_conti_model,
           'qso':self.qso,
           'host':self.host,
           'flux':self.flux,
    }

    #ax.plot(wave_eval, line_single + f_conti_model_eval, color=color, zorder=5) ### 
    #ax.plot(wave_eval, lines_total + f_conti_model_eval, 'b', label='line', zorder=6) ### 
    #ax.plot(self.wave_prereduced[mask], self.flux_prereduced[mask], 'k', label='data', lw=1, zorder=2) ### 
    #ax.plot(self.wave, self.line_flux - self.f_line_model, 'gray',
    #                        label='resid', linestyle='dotted', lw=1, zorder=3) ### 

    #ax.plot(self.wave, self.qso + self.host, 'pink', label='host+qso temp', zorder=3) ### 
    #ax.plot(self.wave, self.flux, 'grey', label='data-host', zorder=1) ### 
    # Continuum results
    #ax.plot(wave_eval, f_conti_model_eval, 'c', lw=2, label='FeII', zorder=7) ### 
    #     ax.plot(wave_eval,
    #             self.PL(wave_eval, pp) + self.F_poly_conti(wave_eval, pp[11:]) + self.Balmer_conti(wave_eval,
    #                                                                                                pp[8:11]), 'y',
    #             lw=2, label='BC', zorder=8) ### 

    # ax.plot(wave_eval, self.PL(wave_eval, pp) + self.F_poly_conti(wave_eval, pp[11:]), color='orange', lw=2,
    #         label='conti', zorder=9) ### 
    
    return out



def HSC_DESI_QSOFIT(ID, table, save = False, download = False, plot = True, plot_show = False, save_fig = False, verbose = True, MC = False, save_path = None, BC= False):


    #fujidata = Table.read('Data/DESI_EDR_all_match_0pt5arcsec_S20A_20240401_AG_AGN_all_14h_FWHM1000.fits')


    row = table[table['TARGETID'] == ID]

    if not download:
        data = np.load(glob('Data/consolidated_spectra_and_ivar/*'+str(ID)+'_spectra.npy')[0])
        lam = data[0]  # OBS wavelength [A]
        flux = data[1]  # OBS flux [erg/s/cm^2/A]
        ivar = data[2]
        err = 1 / np.sqrt(ivar)  # 1 sigma error
        #err = np.median(np.abs(flux[:-1]- flux[1:]))*np.ones(len(flux))#1 / np.sqrt(data[1].data['ivar'])  # 1 sigma error

    if download: 
        coadd_spec = get_spec_data(ID, fujidata = table, survey= row['SURVEY'][0], program=row['PROGRAM'][0])
        lam, flux, ivar = combine_spec_arms_ivar(coadd_spec)
        err = 1 / np.sqrt(ivar)  # 1 sigma error


    plateid = ID
    z = row['Z'][0]  # Redshift

    lammin = np.min(lam)/(1+z)
    lammax = np.max(lam)/(1+z)


    # Optional
    ra = row['ra'][0]  # RA
    dec = row['dec'][0]  # DEC

    # get data prepared 
    q0 = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, path = path_ex)
    #np.savetxt(os.path.join(norm_path, 'mean.txt'),np.c_[lam,flux,err])
    wave_range=np.array([lammin,lammax])

    # one important note set FWHM of Fe II to 3000 km/s in the source code to improve the fitting!
    # do the fitting
    # TODO: BC03 will fail on this fitting!!!

    if save_path == None:
        save_path = path_out

    start = timeit.default_timer()
    q0.Fit(name=None, nsmooth=1, deredden=True, reject_badpix=False, wave_range=wave_range, \
            wave_mask=None, decompose_host=True, host_prior=False, decomp_na_mask=False,host_line_mask=False, 
            npca_gal=5, npca_qso=10, qso_type='CZBIN1',host_type='BC03',Fe_uv_op=True,\
            poly=False, BC=BC, rej_abs_conti=False, rej_abs_line=False, linefit=True, save_result=save, save_fig=save_fig, \
            kwargs_plot={'save_fig_path':save_path, 'broad_fwhm':1200}, save_fits_path=path_out, verbose=verbose, param_file_name='qsopar.fits', plot_fig = plot, MC = MC, nsamp = 50)
    #plt.xlim([100,7000])
    #plt.ylim([-1,100])
    end = timeit.default_timer()
    print(f'Fitting finished in {np.round(end - start, 1)}s')
    if plot_show == True:
        plt.show()

    np.save(f'{save_path}/{plateid}_fit.npy', q0)


    return q0


def HSC_DESI_QSOFIT_lambda_flux(lam, flux, z, err, name = None, save = False, plot = True, plot_show = False, save_fig = False, verbose = True, MC = False, save_path = None):


    #fujidata = Table.read('Data/DESI_EDR_all_match_0pt5arcsec_S20A_20240401_AG_AGN_all_14h_FWHM1000.fits')


    # row = table[table['TARGETID'] == ID]

    # if not download:
    #     data = np.load(glob('Data/consolidated_spectra_and_ivar/*'+str(ID)+'_spectra.npy')[0])
    #     lam = data[0]  # OBS wavelength [A]
    #     flux = data[1]  # OBS flux [erg/s/cm^2/A]
    #     ivar = data[2]
    #     err = 1 / np.sqrt(ivar)  # 1 sigma error
    #     #err = np.median(np.abs(flux[:-1]- flux[1:]))*np.ones(len(flux))#1 / np.sqrt(data[1].data['ivar'])  # 1 sigma error

    # if download: 
    #     coadd_spec = get_spec_data(ID, fujidata = table, survey= row['SURVEY'][0], program=row['PROGRAM'][0])
    #     lam, flux, ivar = combine_spec_arms_ivar(coadd_spec)
    #     err = 1 / np.sqrt(ivar)  # 1 sigma error


    plateid = name
    #z = row['Z'][0]  # Redshift

    lammin = np.min(lam)/(1+z)
    lammax = np.max(lam)/(1+z)


    # Optional
    #ra = row['ra'][0]  # RA
    #dec = row['dec'][0]  # DEC

    # get data prepared 
    q0 = QSOFit(lam, flux, err, z, plateid=plateid, path = path_ex)
    #np.savetxt(os.path.join(norm_path, 'mean.txt'),np.c_[lam,flux,err])
    wave_range=np.array([lammin,lammax])

    # one important note set FWHM of Fe II to 3000 km/s in the source code to improve the fitting!
    # do the fitting
    # TODO: BC03 will fail on this fitting!!!

    if save_path == None:
        save_path = path_out

    start = timeit.default_timer()
    q0.Fit(name=None, nsmooth=1, deredden=True, reject_badpix=False, wave_range=wave_range, \
            wave_mask=None, decompose_host=True, host_prior=False, decomp_na_mask=False,host_line_mask=False, 
            npca_gal=5, npca_qso=10, qso_type='CZBIN1',host_type='BC03',Fe_uv_op=True,\
            poly=False, BC=False, rej_abs_conti=False, rej_abs_line=False, linefit=True, save_result=save, save_fig=save_fig, \
            kwargs_plot={'save_fig_path':save_path, 'broad_fwhm':1200}, save_fits_path=path_out, verbose=verbose, param_file_name='qsopar.fits', plot_fig = plot, MC = MC, nsamp = 50)
    #plt.xlim([100,7000])
    #plt.ylim([-1,100])
    end = timeit.default_timer()
    print(f'Fitting finished in {np.round(end - start, 1)}s')
    if plot_show == True:
        plt.show()

    np.save(f'{save_path}/{plateid}_fit.npy', q0)

    return q0

