import numpy as np
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table, vstack, join
from astropy.convolution import convolve, Gaussian1DKernel
import sys
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.stats import binned_statistic
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression


import fitsio

import os
import urllib.request

from tqdm import tqdm
from glob import glob
from scipy import interpolate

from scipy.special import erf

import desispec.io

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.optimize import curve_fit


plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'figure.autolayout': True})


plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=15)
plt.rc('axes', linewidth=1.5) # change back to 1.5
plt.rc('axes', labelsize=20) # change back to 10
plt.rc('xtick', labelsize=22, direction='in')
plt.rc('ytick', labelsize=22, direction='in')
plt.rc('legend', fontsize=15) # change back to 7

# setting xtick parameters:

plt.rc('xtick.major',size=10,pad=4)
plt.rc('xtick.minor',size=5,pad=4)

plt.rc('ytick.major',size=10)
plt.rc('ytick.minor',size=5)



topdir = '/Volumes/RCR_Passport/HSC+DESI/Data'
if os.path.isfile(f'{topdir}/DESI_EDR_all_match_0pt5arcsec_S20A_092623_AG_AGN.fits'):
    fujidata = Table.read(f'{topdir}/DESI_EDR_all_match_0pt5arcsec_S20A_092623_AG_AGN.fits')


line_dictionary = {
    'Ha' : 6564,
    'Hb' : 4861.35,
    'Hg' : 4340,
    'Mg2': 2800,
    'C4': 1549,
    'C3':1909,
    'O31':4959,
    'O32':5007,
    'O2': 3728,
    'Lya': 1215.67 ,
    'Ne5': 3426,
}



line_names_dictionary = {
    'Ha' :r'$H \alpha$' ,
    'Hb' : r'$H \beta$',
    'Hg' :r'$H \gamma$',
    'Mg2': r'Mg II',
    'C4':  r'C IV',
    'C3': r'C III',
    'O31':r'O III',
    'O32':None,
    'O2': r'O II',
    'Lya': r'Ly$\alpha$' ,
    'Ne5': r'Ne V' ,
}


def load_dict(f):
    return np.load(f, allow_pickle=True).item()
    
def get_g_W3_from_table(fujidata):
     gW3 = fujidata['G_CMODEL_MAG'] -( fujidata['W3MAG_UNWISE'] + 5.174) # Jarret et al 2011 conversion from Vega to AB
     return gW3
     
     
def get_spec_data(tid, fujidata = None, survey=None, program=None, zs = False):
    #-- the index of the specific target can be uniquely determined with the combination of TARGETID, SURVEY, and PROGRAM
    idx = np.where( (fujidata["TARGETID"]==tid) & (fujidata["SURVEY"]==survey) & (fujidata["PROGRAM"]==program) )[0][0]

    #-- healpix values are integers but are converted here into a string for easier access to the file path
    hpx = fujidata["HEALPIX"].astype(str)

    if "sv" in survey:
        specprod = "fuji"

    specprod_dir = f"https://data.desi.lbl.gov/public/edr/spectro/redux/{specprod}"
    target_dir   = f"{specprod_dir}/healpix/{survey}/{program}/{hpx[idx][:-2]}/{hpx[idx]}"
    coadd_fname  = f"coadd-{survey}-{program}-{hpx[idx]}.fits"
    rr_fname  = f"redrock-{survey}-{program}-{hpx[idx]}.fits"
    
    dirname = f"{topdir}/{survey}/{program}/{hpx[idx]}/"
    flocal = f"{topdir}/{survey}/{program}/{hpx[idx]}/{coadd_fname}"
    flocal2 = f"{topdir}/{survey}/{program}/{hpx[idx]}/{rr_fname}"
    
    url = f"{target_dir}/{coadd_fname}"
    url2 = f"{target_dir}/{rr_fname}"

    if not os.path.isfile(flocal2):
        os.makedirs(dirname, exist_ok=True)
        
        if not os.path.isfile(flocal):
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, flocal)
        
        if not os.path.isfile(flocal2):
            print(f"downloading {url2}")
            urllib.request.urlretrieve(url2, flocal2)

    #-- read in the spectra with desispec
    coadd_obj  = desispec.io.read_spectra(f"{flocal}")
    coadd_tgts = coadd_obj.target_ids().data

    #-- select the spectrum of  targetid
    row = ( coadd_tgts==fujidata["TARGETID"][idx] )
    coadd_spec = coadd_obj[row]

    if zs == True:
            rr = fits.open(flocal2)

            z = rr[1].data['Z'][idx].astype(np.float32)
            zerr = rr[1].data['ZERR'][idx].astype(np.float32)

            return coadd_spec, [z, zerr]
    
    return coadd_spec


def combine_spec_arms(coadd_spec):
    cut_pixels = 100
    lens = np.zeros(3)
    j = 0
    for band in ("b","r","z"):
        lens[j] = len(coadd_spec.wave[band][cut_pixels:-cut_pixels]) # Clip the overlapping pixels
        j += 1

    tot_len = int(np.sum(lens))

    lens = lens.astype(int)

    waves_array = np.zeros(tot_len)
    flux_array = np.zeros(tot_len)
    
    j = 0
    #for band in ("b","r","z"):

    waves_array[:lens[0]] = coadd_spec.wave["b"][cut_pixels:-cut_pixels]
    flux_array[:lens[0]] = coadd_spec.flux["b"][0][cut_pixels:-cut_pixels]


    waves_array[lens[0]:lens[0]+lens[1]] = coadd_spec.wave["r"][cut_pixels:-cut_pixels]
    flux_array[lens[0]:lens[0]+lens[1]] = coadd_spec.flux["r"][0][cut_pixels:-cut_pixels]

    waves_array[lens[0]+lens[1]:tot_len] = coadd_spec.wave["z"][cut_pixels:-cut_pixels]
    flux_array[lens[0]+lens[1]:tot_len] = coadd_spec.flux["z"][0][cut_pixels:-cut_pixels]




    return waves_array,flux_array


def combine_spec_arms_ivar_deprecated(coadd_spec):
    cut_pixels = 100
    lens = np.zeros(3)
    j = 0
    for band in ("b","r","z"):
        lens[j] = len(coadd_spec.wave[band][cut_pixels:-cut_pixels]) # Clip the overlapping pixels
        j += 1

    tot_len = int(np.sum(lens))

    lens = lens.astype(int)

    waves_array = np.zeros(tot_len)
    flux_array = np.zeros(tot_len)
    ivar_array = np.zeros(tot_len)
    
    j = 0
    #for band in ("b","r","z"):

    waves_array[:lens[0]] = coadd_spec.wave["b"][cut_pixels:-cut_pixels]
    flux_array[:lens[0]] = coadd_spec.flux["b"][0][cut_pixels:-cut_pixels]
    ivar_array[:lens[0]] = coadd_spec.ivar["b"][0][cut_pixels:-cut_pixels]


    waves_array[lens[0]:lens[0]+lens[1]] = coadd_spec.wave["r"][cut_pixels:-cut_pixels]
    flux_array[lens[0]:lens[0]+lens[1]] = coadd_spec.flux["r"][0][cut_pixels:-cut_pixels]
    ivar_array[lens[0]:lens[0]+lens[1]] = coadd_spec.ivar["r"][0][cut_pixels:-cut_pixels]

    waves_array[lens[0]+lens[1]:tot_len] = coadd_spec.wave["z"][cut_pixels:-cut_pixels]
    flux_array[lens[0]+lens[1]:tot_len] = coadd_spec.flux["z"][0][cut_pixels:-cut_pixels]
    ivar_array[lens[0]+lens[1]:tot_len] = coadd_spec.ivar["z"][0][cut_pixels:-cut_pixels]




    return waves_array,flux_array, ivar_array 


    
def in_wavelength_range(wavelength, xlim):
    bool = False

    if (wavelength > xlim[0])*(wavelength < xlim[1]):
        bool = True
    return bool

def s_to_fwhm(sigma):
    return 2*np.sqrt(2*np.log(2))*sigma 

def read_in_relevant_fastspec_fuji(filename):
    hdu = fits.open(filename)[1]
    hdu2 = fits.open(filename)[2]

    tid = hdu.data['TARGETID']
    srv = hdu.data['SURVEY']
    prg = hdu.data['PROGRAM']
    hpx = hdu.data['HEALPIX']
    z = hdu2.data['Z']
    ha_s = hdu.data['HALPHA_SIGMA'] # gaussian sigma, km/s 
    hb_s = hdu.data['HBETA_SIGMA']
    mgii1_s = hdu.data['MGII_2796_SIGMA']
    mgii2_s = hdu.data['MGII_2803_SIGMA']
    civ_s = hdu.data['CIV_1549_SIGMA']

    tab = Table([tid,srv,prg,hpx,z, ha_s, hb_s, mgii1_s, mgii2_s, civ_s], 
                names = ('TARGETID','SURVEY','PROGRAM','HEALPIX', 'Z','HALPHA_SIGMA','HBETA_SIGMA','MGII_2796_SIGMA','MGII_2803_SIGMA', 'CIV_1549_SIGMA' ))

    return tab



def _redshift(data_in, z_in, z_out, data_type):
    """Redshift Correction for input data

    Parameters
    ----------
    data_in : numpy.ndarray
        Input data which is either flux values, wavelengths or ivars.
        Default DESI units are assumed.
    z_in : float or numpy.ndarray
        input redshifts
    z_out : float or numpy.ndarray
        output redshifts
    data_type : str
        "flux", "wave" or "ivar"

    Returns
    -------
    numpy.ndarray
        redshift corrected value corresponding to data type
    """
    exponent_dict = {"flux": -1, "wave": 1, "ivar": 2}
    assert data_type in exponent_dict.keys(), "Not a valid Data Type"
    data_in = np.atleast_2d(data_in)

    if z_in.ndim == 1:
        z_in = z_in[:, np.newaxis]
    exponent = exponent_dict[data_type]
    data_out = data_in * (((1 + z_out) / (1 + z_in)) ** exponent)

    return data_out


def _common_grid(flux, wave, ivar, z_in, z_out=0.0, wave_grid=None):
    """Bring spectra to a common grid

    Parameters
    ----------
    flux : np.ndarray
        numpy array containing the flux grid of shape [num_spectra, num_wavelengths]
    wave : np.ndarray
        numpy array containing the wavelength grid of shape [num_wavelengths] or [num_spectra, num_wavelengths]
    ivar : np.ndarray
        numpy array containing the inverse variance grid of shape [num_spectra, num_wavelengths]
    z_in : np.ndarray
        a 1D numpy array containing the redshifts of each spectra
    z_out : float, optional
        common redshift for the output data, by default 0.0
    wave_grid : np.ndarray, optional
        a 1D vector containing the wavelength grid for the output, by default None.
        If set to None, the wavelength grid is linearly spaced between the maximum and minimum
        possible wavelengths after redshift correction with a bin width of 0.8 Angstrom (DESI default)

    Returns
    -------
    flux_new: np.ndarray
        All the input fluxes brought to a common redshift and wavelength grid.
        Missing values and extrapolations are denoted with nan.
    ivar_new: np.ndarray
        All input inverse variances brought to a common redshift and wavelength grid.
        Missing values and extrapolations are denoted with nan.
    wave_grid: np.ndarray
        The common wavelength grid.
    """
    # Correct for redshift
    z_out = np.atleast_1d(z_out)
    flux_new = _redshift(flux, z_in, z_out, "flux")
    wave_new = _redshift(wave, z_in, z_out, "wave")
    ivar_new = _redshift(ivar, z_in, z_out, "ivar")
    # TODO Fix the resolution issue
    # resample to common grid
    if wave_grid is None:
        wave_grid = np.arange(np.min(wave_new), np.max(wave_new), 0.32)
    flux_new, ivar_new = resample_spectra(
        wave_grid, wave_new, flux_new, ivar_new, verbose=False, fill=np.nan
    )
    return flux_new, ivar_new, wave_grid


def _coadd_cameras(flux_cam, wave_cam, ivar_cam):
    """Adds spectra from the three cameras to give on combined spectra per object

    Parameters
    ----------
    flux_cam : dict
        Dictionary containing the flux values from the three cameras
    wave_cam : dict
        Dictionary containing the wavelength values from the three cameras
    ivar_cam : dict
        Dictionary containing the inverse variance values from the three cameras

    Returns
    -------
    Tuple
        returns the combined flux, wavelength and inverse variance grids.
    """
    # check_alignement_of_camera_wavelength(spectra)

    # ordering
    bands = list(wave_cam.keys())
    mwave = [np.mean(wave_cam[b]) for b in bands]
    sbands = np.array(bands)[np.argsort(mwave)]  # bands sorted by inc. wavelength

    # create wavelength array
    wave = None
    tolerance = 0.0001  # A , tolerance
    for b in sbands:
        if wave is None:
            wave = wave_cam[b]
        else:
            wave = np.append(wave, wave_cam[b][wave_cam[b] > wave[-1] + tolerance])
    nwave = wave.size

    # check alignment
    number_of_overlapping_cameras = np.zeros(nwave)
    for b in bands:
        windices = np.argmin(
            (
                np.tile(wave, (wave_cam[b].size, 1))
                - np.tile(wave_cam[b], (wave.size, 1)).T
            )
            ** 2,
            axis=1,
        )
        dist = np.sqrt(np.max(wave_cam[b] - wave[windices]))

        if dist > tolerance:
            print(
                "Cannot directly coadd the camera spectra because wavelength are not aligned,use --lin-step or --log10-step to resample to a common grid"
            )
            sys.exit(12)
        number_of_overlapping_cameras[windices] += 1
    # targets
    # TODO Add assertions to check all the input sizes are correct

    b = sbands[0]
    ntarget = len(flux_cam[b])
    flux = np.zeros((ntarget, nwave), dtype=flux_cam[b].dtype)
    ivar = np.zeros((ntarget, nwave), dtype=flux_cam[b].dtype)

    for b in bands:

        # indices
        windices = np.argmin(
            (
                np.tile(wave, (wave_cam[b].size, 1))
                - np.tile(wave_cam[b], (wave.size, 1)).T
            )
            ** 2,
            axis=1,
        )

        for i in range(ntarget):
            ivar[i, windices] += ivar_cam[b][i]
            flux[i, windices] += ivar_cam[b][i] * flux_cam[b][i]

    for i in range(ntarget):
        ok = ivar[i] > 0
        if np.sum(ok) > 0:
            flux[i][ok] /= ivar[i][ok]

    return flux, wave, ivar


def _normalize(flux, ivar):
    """
    A simple normalization to median=1 for flux
    Also adjusts inverse variance accordingly

    Parameters
    ----------
    flux: np.ndarray
        numpy array containing the flux grid of shape [num_spectra, num_wavelengths]

    ivar : np.ndarray
        numpy array containing the inverse variance grid of shape [num_spectra, num_wavelengths]

    Returns
    -------
    flux: np.ndarray
        flux that has been normalized to median one

    ivar: np.ndarray
        inverse variance that has been multipled by the normalization factor
        for the flux,squared

    """

    norm = np.nanmedian(flux, axis=1, keepdims=True)
    flux = flux / norm
    ivar = ivar * norm ** 2

    return flux, ivar


def _wavg(flux, ivar=None, weighted=False, weights=None):
    """
    Weighted average of the spectra.

    Parameters
    ----------
    flux: np.ndarray
        numpy array containing the flux grid of shape [num_spectra, num_wavelengths]

    ivar : np.ndarray
        numpy array containing the inverse variance grid of shape [num_spectra, num_wavelengths]

    weights : np.ndarray
        numpy array containing the weights grid of shape [num_spectra, num_wavelengths]

    weighted: True or False
        if false, use weight=1 for all the spectra
        else, perform a weighted average using the input for 'weights'

    Returns
    ----------
    avg: np.ndarray
        numpy array containing the averaged flux of shape [num_wavelengths]

    """

    if weighted:
        num = np.nansum(flux * weights, axis=0)
        denom = np.nansum(weights, axis=0)

        if 0.0 in denom:
            denom[denom == 0.0] = np.nan

        avg = np.nan_to_num(num / denom)
    else:
        avg = np.mean(flux, axis=0)

    ivar = np.nansum(ivar, axis=0)

    return avg, ivar


def _bootstrap(flux_spec, ndata, nbootstraps, len_spec):
    """
    Sample the spectra

    Parameters
    ----------
    flux_spec:  np.ndarray
        Numpy array containing the flux grid of shape [num_spectra, num_wavelengths]
        To avoid redundant calculations, this array can be the normalized spectra already
        brought to the common grid

    ndata: int
        The number of spectra to sample from

    nbootstraps: int
        The number of times to sample the data

    nsamples: int
        The number of bootstraps to do

    len_spec: int
        The number of wavelengths in the spectra

    Returns
    ----------
    stacks: np.ndarray
        numpy array containing the stacked spectra from the bootstraps of size [nsamples, len_spec]

    ivar: np.ndarray
        numpy array of size [len_spec] containing the inverse variance calculated from all the stacks

    """

    boot = np.zeros((nbootstraps, len_spec))
    for i in range(nbootstraps):
        idx = np.random.choice(ndata, 1, replace=True)
        boot[i] += flux_spec[idx][0]

    return boot


def stack_spectra(flux, wave, ivar, sky=None, bootstrap=False):
    """
    If flux/wave.ivar are dicts, coadd cameras.
    If sky is present model ivar and do modelled ivar weighted avg
    """

    stacks = np.zeros((nbootstraps, len_spec))
    for j in range(nbootstraps):
        boot = np.zeros((nsamples, len_spec))
        for i in range(nsamples):
            idx = np.random.choice(ndata, 1, replace=True)
            boot[i] += flux_spec[idx][0]
        stacks[j] += wavg(boot)

    ivar = 1.0 / (np.nanstd(stacks, axis=0)) ** 2

    return stacks, ivar


def model_ivar(ivar, sky_ivar, wave):
    n_obj = len(sky_ivar)
    sky_var = 1 / sky_ivar

    ivar_model = np.zeros_like(ivar)

    for i in range(n_obj):
        sky_mask = np.isfinite(sky_var[i])
        sky_var_interp = interp1d(
            wave[sky_mask], sky_var[i][sky_mask], fill_value="extrapolate", axis=-1
        )
        sky_var[i] = sky_var_interp(wave)
        sky_var[i] = sky_var[i] / median_filter(
            sky_var[i], 100
        )  # takes out the overall shape of sky var

        # Create polunomial function of wavelength
        poly_feat_m = PolynomialFeatures(3)
        poly_feat_c = PolynomialFeatures(3)
        coef_m = poly_feat_m.fit_transform(wave[:, np.newaxis])
        coef_c = poly_feat_c.fit_transform(wave[:, np.newaxis])

        obj_var = 1 / (ivar[i])
        obj_mask = np.isfinite(obj_var)  # TODO Check for Nan values here
        obj_back = median_filter(obj_var[obj_mask], 100, mode="nearest")
        X = (
            np.concatenate(
                [(coef_m * sky_var[i][:, np.newaxis])[obj_mask], coef_c[obj_mask]],
                axis=1,
            )
            + obj_back[:, np.newaxis]
        )
        Y = obj_var[obj_mask]
        model = LinearRegression(fit_intercept=False, n_jobs=-1)
        model.fit(X, Y)
        y_predict = model.predict(X)
        residual = (Y - y_predict) / Y
        # correct for the overall shape of the residuals
        wave_bins = np.arange(wave.min(), wave.max(), 500)
        binned_residual, _, _ = binned_statistic(
            wave[obj_mask], residual, statistic="median", bins=wave_bins
        )
        interp_binned_res = interp1d(
            (wave_bins[1:] + wave_bins[:-1]) / 2,
            binned_residual,
            kind="cubic",
            fill_value="extrapolate",
        )
        large_res = interp_binned_res(wave[obj_mask])
        y_pred_adjust = large_res * Y + y_predict
        ivar_model[i][obj_mask] = 1 / y_pred_adjust
        ivar_model[i][~obj_mask] = 0
    return ivar_model




def combine_spec_arms_ivar(coadd_spec):

    flux, wave, ivar = _coadd_cameras(coadd_spec.flux, coadd_spec.wave, coadd_spec.ivar)
    #print('Coadding cameras')
    return wave,flux[0], ivar[0]

