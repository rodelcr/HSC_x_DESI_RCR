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

if not os.path.exists('qsopar.fits'):
    # create a header

    hdr0 = fits.Header()
    hdr0['Author'] = 'Rodrigo Cordova'
    hdr0['Code'] = 'PyQSOFit'
    primary_hdu = fits.PrimaryHDU(header=hdr0)

    """
    Create parameter file
    lambda    complexname  minwav maxwav linename ngauss inisca minsca maxsca inisig minsig maxsig voff vindex windex findex fvalue vary
    """


    newdata = np.rec.array([
    (6564.61, 'Ha', 6400, 6800, 'Ha_br',   1, 0.0, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.01, 0, 0, 0, 0.05 , 1),
    (6564.61, 'Ha', 6400, 6800, 'Ha_na',   1, 0.0, 0.0, 1e10, 5.2e-4,2.3e-4,0.00169, 0.001, 1, 1, 0, 0.002, 1),
    (6549.85, 'Ha', 6400, 6800, 'NII6549', 1, 0.0, 0.0, 1e10, 5.2e-4,2.3e-4,0.00169, 5e-3,  1, 1, 0, 0.001, 1),
    (6585.28, 'Ha', 6400, 6800, 'NII6585', 1, 0.0, 0.0, 1e10, 5.2e-4,2.3e-4, 0.00169, 5e-3,  1, 1, 0, 0.003, 1),
    (6718.29, 'Ha', 6400, 6800, 'SII6718', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 2, 0.001, 1),
    (6732.67, 'Ha', 6400, 6800, 'SII6732', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 2, 0.001, 1),
        
    (4862.68, 'Hb', 4640, 5100, 'Hb_br',     1, 0.0, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.01, 0, 0, 0, 0.01 , 1),
    (4862.68, 'Hb', 4640, 5100, 'Hb_na',     1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.001, 1, 1, 0, 0.002, 1),
    (4960.30, 'Hb', 4640, 5100, 'OIII4959c', 1, 0.0, 0.0, 1e10, 5e-4, 2.3e-4, 0.00169, 0.001, 1, 1, 0, 0.002, 1),
    (5008.24, 'Hb', 4640, 5100, 'OIII5007c', 1, 0.0, 0.0, 1e10, 5e-4, 2.3e-4, 0.00169, 0.001, 1, 1, 0, 0.004, 1),
    (4960.30, 'Hb', 4640, 5100, 'OIII4959w',   1, 0.0, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.001, 1),
    (5008.24, 'Hb', 4640, 5100, 'OIII5007w',   1, 0.0, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.002, 1),
    # (4687.02, 'Hb', 4640, 5100, 'HeII4687_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.005, 0, 0, 0, 0.001, 1),
    # (4687.02, 'Hb', 4640, 5100, 'HeII4687_na', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.005, 1, 1, 0, 0.001, 1),

    #(3934.78, 'CaII', 3900, 3960, 'CaII3934' , 2, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 99, 0, 0, -0.001, 1),

    (3728.48, 'OII', 3650, 3800, 'OII3728', 1, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 1, 1, 0, 0.001, 1),

    (3426.84, 'NeV', 3380, 3480, 'NeV3426',    1, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 0, 0, 0, 0.001, 1),
    (3426.84, 'NeV', 3380, 3480, 'NeV3426_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025,   0.02,   0.01, 0, 0, 0, 0.001, 1),

    (2798.75, 'MgII', 2700, 2900, 'MgII_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05, 1),
    (2796.35, 'MgII', 2700, 2900, 'MgII_na1', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    (2803.53, 'MgII', 2700, 2900, 'MgII_na2', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),

    # (1908.73, 'CIII', 1700, 1970, 'CIII_br',   2, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 99, 0, 0, 0.01, 1),
    #(1908.73, 'CIII', 1700, 1970, 'CIII_na',   1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01,  1, 1, 0, 0.002, 1),
    #(1892.03, 'CIII', 1700, 1970, 'SiIII1892', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1857.40, 'CIII', 1700, 1970, 'AlIII1857', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1816.98, 'CIII', 1700, 1970, 'SiII1816',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1786.7,  'CIII', 1700, 1970, 'FeII1787',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1750.26, 'CIII', 1700, 1970, 'NIII1750',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),
    #(1718.55, 'CIII', 1700, 1900, 'NIV1718',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),

    (1549.06, 'CIV', 1500, 1700, 'CIV_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05,   0.015, 0, 0, 0, 0.05 , 1),
    (1549.06, 'CIV', 1500, 1700, 'CIV_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01,  1, 1, 0, 0.002, 1),
    # (1640.42, 'CIV', 1500, 1700, 'HeII1640',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
    # (1663.48, 'CIV', 1500, 1700, 'OIII1663',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
    # (1640.42, 'CIV', 1500, 1700, 'HeII1640_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),
    # (1663.48, 'CIV', 1500, 1700, 'OIII1663_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),

    #(1402.06, 'SiIV', 1290, 1450, 'SiIV_OIV1', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1396.76, 'SiIV', 1290, 1450, 'SiIV_OIV2', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1335.30, 'SiIV', 1290, 1450, 'CII1335',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),
    #(1304.35, 'SiIV', 1290, 1450, 'OI1304',    1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),

    (1215.67, 'Lya', 1150, 1290, 'Lya_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05,   0.02, 0, 0, 0, 0.05 , 1),
    (1215.67, 'Lya', 1150, 1290, 'Lya_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01, 0, 0, 0, 0.002, 1)
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


def HSC_DESI_QSOFIT(ID, table, save = False, download = False, plot = True, plot_show = False, save_fig = False, verbose = True, MC = False):


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

    start = timeit.default_timer()
    q0.Fit(name=None, nsmooth=1, deredden=True, reject_badpix=False, wave_range=wave_range, \
            wave_mask=None, decompose_host=True, host_prior=False, decomp_na_mask=False,host_line_mask=False, 
            npca_gal=5, npca_qso=10, qso_type='CZBIN1',host_type='BC03',Fe_uv_op=True,\
            poly=False, BC=False, rej_abs_conti=False, rej_abs_line=False, linefit=True, save_result=save, save_fig=save_fig, \
            kwargs_plot={'save_fig_path':path_out, 'broad_fwhm':1200}, save_fits_path=path_out, verbose=verbose, param_file_name='qsopar.fits', plot_fig = plot, MC = MC)
    #plt.xlim([100,7000])
    #plt.ylim([-1,100])
    end = timeit.default_timer()
    print(f'Fitting finished in {np.round(end - start, 1)}s')
    if plot_show == True:
        plt.show()

    return q0

