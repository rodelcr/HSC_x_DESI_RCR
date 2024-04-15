from HSC_x_DESI_RCR.hsc_desi_utils import *

from HSC_x_DESI_RCR.hsc_desi_spectra_fitting import *


def plot_coadd(coadd_spec, ax = None, labels = True, c = 'k', ls = '-', scale = True, step = False, smooth = False):
    if ax == None:
        fig = plt.figure()
        ax = plt.gca()

    ax.set_xlim(3600, 9800)
    ax.set_ylim(-3.9, 30)  #-- change depending on specific spectrum
    if labels == True:
        ax.set_ylabel(r"$F_{\lambda}\ \left[ 10^{-17}\ {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ \AA^{-1} \right]$")
        ax.set_xlabel(r"wavelength $\lambda\ \left[ \AA \right]$")

    max_ = np.zeros(3)
    min_ = np.zeros(3)

    j = 0
    for band in ("b","r","z"):
        flux = coadd_spec.flux[band][0]
        if smooth == True:
            flux = convolve(flux, Gaussian1DKernel(5))

        if step == False:
        
            ax.plot(coadd_spec.wave[band], flux,
                    color=c, label= "", ls = ls)
            
        if step == True:
            ax.step(coadd_spec.wave[band], flux,
                    color=c, label= "", ls = ls)
            
        max_[j] = np.max(flux)
        min_[j] = np.min(flux)

        j +=1
    
                
    if scale == True: 
        ax.set_ylim(np.min(min_)-1, np.max(max_)+2)


def plot_coadd_limited(coadd_spec, ax = None, labels = True, c = 'k', ls = '-', scale = True, lowerlim =3000, upperlim =10000, step = False, smooth = False, scale_amount = 0.5):
    if ax == None:
        fig = plt.figure()
        ax = plt.gca()

    ax.set_xlim(3600, 9800)
    ax.set_ylim(-3.9, 30)  #-- change depending on specific spectrum
    if labels == True:
        ax.set_ylabel(r"$F_{\lambda}\ \left[ 10^{-17}\ {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ \AA^{-1} \right]$")
        ax.set_xlabel(r"wavelength $\lambda\ \left[ \AA \right]$")

    max_ = np.zeros(3)
    min_ = np.zeros(3)

    j = 0
    for band in ("b","r","z"):
        wavelens = coadd_spec.wave[band][20:-20] # Clip the overlapping pixels
        flux = coadd_spec.flux[band][0]
        if smooth == False:
            flux = coadd_spec.flux[band][0][20:-20]
        if smooth == True:
            flux = convolve(flux, Gaussian1DKernel(5))[20:-20]

        wavelim = np.where((wavelens>lowerlim)&(wavelens < upperlim))

        wavelens = wavelens[wavelim]
        flux  =flux[wavelim]

        if step == False:
            ax.plot(wavelens, flux,
                    color=c, label= "", ls = ls)

        if step == True:
            ax.step(wavelens, flux,
                    color=c, label= "", ls = ls)
            

        flux2 = np.copy(flux)
        if len(flux)<1:
            flux = np.zeros(2)
            

        max_[j] = np.max(flux)
        min_[j] = np.min(flux)
        
        if len(flux2)<1:
            min_[j] = 1000
            
        # Clip even further to ensure the middle of the plot has data
            
        waves_all, fluxes_all =  combine_spec_arms(coadd_spec)
        middle = np.mean([lowerlim, upperlim])
        wavelim_all = np.where((waves_all>(middle-10))&(waves_all < (middle+10)))

        waves_all = waves_all[wavelim_all]
        fluxes_all  =fluxes_all[wavelim_all]

        if len(fluxes_all)<1:
            min_[j] = 1000
            max_[j] = 0
            
        j +=1
                        
    if scale == True: 
        ax.set_ylim(np.min(min_)-scale_amount, np.max(max_)+scale_amount)


    return max_, min_




def plot_coadd_restframe(coadd_spec, z, ax = None, labels = True, c = 'k', ls = '-', scale = True):
    if ax == None:
        fig = plt.figure()
        ax = plt.gca()

    ax.set_ylim(-3.9, 30)  #-- change depending on specific spectrum
    if labels == True:
        ax.set_ylabel(r"$F_{\lambda}\ \left[ 10^{-17}\ {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ \AA^{-1} \right]$")
        ax.set_xlabel(r"wavelength $\lambda\ \left[ \AA \right]$")

    max_ = np.zeros(3)
    j = 0
    for band in ("b","r","z"):
        
        ax.plot(coadd_spec.wave[band]/(1+z), convolve(coadd_spec.flux[band][0], Gaussian1DKernel(5)),
                color=c, label= "", ls = ls)
        
        max_[j] = np.max(convolve(coadd_spec.flux[band][0], Gaussian1DKernel(5)))
        j +=1
    
    ax.set_xlim(2000, 9900)
    if scale == True: 
        ax.set_ylim(-3, np.max(max_)+2)


def add_subplot_axes(ax,rect,facecolor='w'): # matplotlib 2.0+
#def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    #subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax
    
def plot_coadd_zoom_in(coadd_spec, tabrow, fastspec = None, ax = None, labels = True, c = 'k', ls = '-', 
                       scale = True, ylim = None, ylim1 = None, ylim2 = None, addtl_text= None):
    fig = plt.figure(figsize=(18,7))
    axes = fig.add_subplot(111)
    i = 0
    plot_coadd(coadd_spec, ax = axes, labels = False, c = c, scale = True)
    axes.set_ylim(ylim)

    if len(tabrow) > 1:
        print('Incorrect Table input. Must provide a single row of information for metadata.')
        

    z = tabrow['Z']
    zphot = tabrow['best_z']
    id = tabrow['TARGETID']
    class_ = tabrow['FINALCLASS']
    utype = tabrow['UMAP_TYPE']
    
    axes.text(0.01,0.05, 'DESI Target ID = {}'.format(id), transform=axes.transAxes)
    axes.text(0.8,0.9, 'Z(DESI) = {:0.2f}'.format(z), transform=axes.transAxes)
    axes.text(0.8,0.8, 'Z(HSC) = {:0.2f}'.format(zphot), transform=axes.transAxes)
    axes.text(0.02,0.9, 'FINALCLASS = {}'.format(class_), transform=axes.transAxes)
    axes.text(0.02,0.8, 'UMAP TYPE = {}'.format(utype), transform=axes.transAxes)




    rect = [0.2,0.8,0.2,0.25]
    ax1 = add_subplot_axes(axes,rect)
    plot_coadd_limited(coadd_spec, ax = ax1, labels = False, c = c, lowerlim=(4900-200)*(1+z), upperlim=(4900+200)*(1+z))
    ax1.set_xlim([(4900-200)*(1+z), (4900+200)*(1+z)]) 
    ax1.set_ylabel(r'$H \beta$ + O III')
    ax1.axvline(4861.35*(1+z), ls = ':', label = r'$H \beta$')
    ax1.axvline(5007*(1+z), ls = '--', c = 'k',label = r'O III')
    ax1.axvline(4959*(1+z), ls = '--', c = 'k',label = None)
    ax1.set_ylim(ylim1)
    
    rect = [0.5,0.8,0.2,0.25]
    ax2 = add_subplot_axes(axes,rect)
    plot_coadd_limited(coadd_spec, ax = ax2, labels = False, c = c, lowerlim=(6564-100)*(1+z), upperlim=(6564+100)*(1+z))
    ax2.set_xlim([(6564-100)*(1+z), (6564+100)*(1+z)]) 
    ax2.axvline(6564*(1+z), ls = ':',label = None)

    ax2.set_ylabel(r'$H \alpha$')
    ax2.set_ylim(ylim2)


    axes.set_ylabel(r"$F_{\lambda}\ \left[ 10^{-17}\ {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ \AA^{-1} \right]$", fontsize = 30, x = 0.0)
    axes.set_xlabel(r"Observed Wavelength $\lambda\ \left[ \AA \right]$", fontsize = 30)
    #plt.savefig('20240307_example_10_FINALCLASS1_matched.png', bbox_inches = 'tight')


def plot_single_emission_line(axes, z, key, ls = ':', c = 'k', line_labels = True):
    xlims = axes.get_xlim()
    trans = axes.get_xaxis_transform()
    wavelength = line_dictionary[key]*(1+z)
    name = line_names_dictionary[key] 
    if line_labels== False:
        name = None 
    if in_wavelength_range(wavelength, xlims):
        axes.axvline(line_dictionary[key]*(1+z), ls = ls, alpha=0.5, zorder = 0, c = c)
        axes.text(line_dictionary[key]*(1+z) + 5, 0.6,name, alpha=0.5, zorder = 0, transform=trans, c = c,rotation=90) #offset the name a little bit


def plot_all_emission_lines(axes, z, c = 'k', ls = ':', line_labels = True):

    lines = [i for i in line_dictionary.keys()]

    [plot_single_emission_line(axes, z, i,  ls = ls, c = c, line_labels=line_labels) for i in lines]


def plot_coadd_triptych(coadd_spec, tabrow, fastspec= None, ax = None, labels = True, c = 'k', ls = '-', 
                       scale = True, ylim = None, ylim1 = None, ylim2 = None, addtl_text= None, step = True, smooth = False,
                       hsc_linelabels = False):
    fig = plt.figure(figsize=(34,8))
    gs = fig.add_gridspec( 1,3,  width_ratios=(2.5, 1,1),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.1, hspace=0.05)

    axes = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    plot_coadd_limited(coadd_spec, ax = axes, labels = False, c = c, scale = True, step = False, smooth = True, scale_amount=2.)

    z = tabrow['Z']
    zphot = tabrow['best_z']
    zphot_type = tabrow['ZTYPE']
    id = tabrow['TARGETID']
    class_ = tabrow['FINALCLASS']
    utype = tabrow['UMAP_TYPE']
    gw3 = get_g_W3_from_table(tabrow)

    
    alpha, A, sigmas = fit_spectra_gamma(coadd_spec, tabrow)

    
    axes.text(0.02,0.03, 'DESI Target ID = {}'.format(id), transform=axes.transAxes)
    axes.text(0.2,0.9, 'Z(DESI) = {:0.2f}'.format(z), transform=axes.transAxes)
    axes.text(0.2,0.8, 'Z(HSC) = {:0.2f}'.format(zphot), transform=axes.transAxes, color= 'r')
    axes.text(0.2,0.85, 'ZTYPE(HSC) = {}'.format(zphot_type), transform=axes.transAxes)
    axes.text(0.02,0.9, 'FINALCLASS = {}'.format(class_), transform=axes.transAxes)
    axes.text(0.02,0.85, r'$g-W3$ = {:0.2f}'.format(gw3), transform=axes.transAxes)
    axes.text(0.02,0.8, 'UMAP TYPE = {}'.format(utype), transform=axes.transAxes)
    axes.text(0.02,0.75, r'$\gamma \,[ \lambda ^{{- \gamma}}]$ = {{{:0.2f}}}'.format(alpha*-1.), transform=axes.transAxes)


    axes.text(0.65,0.95, r'Manual Fits (Restframe)', transform=axes.transAxes)
    axes.text(0.65,0.9, r'FWHM($H\alpha$) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['Ha']), s_to_fwhm(sigmas['Ha'])*line_dictionary['Ha']/2.99e5 ), transform=axes.transAxes)
    axes.text(0.65,0.85, r'FWHM($H\beta$) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['Hb']), s_to_fwhm(sigmas['Hb'])*line_dictionary['Hb']/2.99e5 ), transform=axes.transAxes)
    axes.text(0.65,0.8, r'FWHM(Mg II) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['Mg2']), s_to_fwhm(sigmas['Mg2'])*line_dictionary['Mg2']/2.99e5 ), transform=axes.transAxes)
    axes.text(0.65,0.75, r'FWHM(C IV) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['C4']), s_to_fwhm(sigmas['C4'])*line_dictionary['C4']/2.99e5 ), transform=axes.transAxes)


    if fastspec != None:
        fastspec_info = fastspec[id == fastspec['TARGETID']]
        if len(fastspec_info['Z'])>1:
            print('WARNING: More than one row of fastspec data matched with TARGETID')
        fast_spec_z = fastspec_info['Z'][0]
        ha_s = fastspec_info['HALPHA_SIGMA'][0] # gaussian sigma, km/s 
        hb_s = fastspec_info['HBETA_SIGMA'][0]
        mgii1_s = fastspec_info['MGII_2796_SIGMA'][0]
        civ_s = fastspec_info['CIV_1549_SIGMA'][0]

        
        axes.text(0.4,0.95, r'FASTSPEC Fits', transform=axes.transAxes)
        axes.text(0.4,0.9, r'FWHM($H\alpha$) = {:0.2f} km/s'.format(s_to_fwhm(ha_s)), transform=axes.transAxes)
        axes.text(0.4,0.85, r'FWHM($H\beta$) = {:0.2f} km/s'.format(s_to_fwhm(hb_s)), transform=axes.transAxes)
        axes.text(0.4,0.8, r'FWHM(Mg II) = {:0.2f} km/s'.format(s_to_fwhm(mgii1_s)), transform=axes.transAxes)
        axes.text(0.4,0.75, r'FWHM(C IV) = {:0.2f} km/s'.format(s_to_fwhm(civ_s)), transform=axes.transAxes)
        
        axes.text(0.2,0.75, 'Z(FASTSPEC) = {:0.2f}'.format(fast_spec_z), transform=axes.transAxes)


    plot_all_emission_lines(axes, z)
    plot_all_emission_lines(axes, zphot, c ='r', line_labels= hsc_linelabels)
    axes.set_ylim(ylim)

    ma1, mi1 = plot_coadd_limited(coadd_spec, ax = ax1, labels = False, c = c, lowerlim=(4900-200)*(1+z), upperlim=(4900+200)*(1+z), step = step, smooth = smooth)
    if np.max(ma1) > 0.:
        ax1.set_xlim([(4900-200)*(1+z), (4900+200)*(1+z)]) 

        ax1.axvline(line_dictionary['Hb']*(1+z), ls = ':', label = r'$H \beta$', alpha=0.5, zorder = 0)
        ax1.axvline(line_dictionary['O31']*(1+z), ls = '--', c = 'k',label = r'O III', alpha=0.5, zorder = 0)
        ax1.axvline(line_dictionary['O32']*(1+z), ls = '--', c = 'k',label = None, alpha=0.5, zorder = 0)

    
    if np.max(ma1) == 0.:
        plot_coadd_limited(coadd_spec, ax = ax1, labels = False, c = c, lowerlim=(1909-100)*(1+z), upperlim=(1909+100)*(1+z), step = step, smooth = smooth)
        ax1.set_xlim([(1909-100)*(1+z), (1909+100)*(1+z)]) 
        ax1.axvline(line_dictionary['C3']*(1+z), ls = ':', label = r'C III', alpha=0.5, zorder = 0)



    ax1.set_ylim(ylim1)
    ax1.legend(fontsize = 20, loc =2 )


    ma2, mi2 = plot_coadd_limited(coadd_spec, ax = ax2, labels = False, c = c, lowerlim=(6564-100)*(1+z), upperlim=(6564+100)*(1+z), step = step, smooth = smooth)
    if np.max(ma2) > 0.:
        ax2.set_xlim([(6564-100)*(1+z), (6564+100)*(1+z)]) 
        ax2.axvline(line_dictionary['Ha']*(1+z), ls = ':', alpha=0.5, zorder = 0, label = r'$H \alpha$')

    if np.max(ma2) == 0.:
        plot_coadd_limited(coadd_spec, ax = ax2, labels = False, c = c, lowerlim=(2800-100)*(1+z), upperlim=(2800+100)*(1+z), step = step, smooth = smooth)
        ax2.set_xlim([(2800-100)*(1+z), (2800+100)*(1+z)]) 

        ax2.axvline(line_dictionary['Mg2']*(1+z), ls = ':', alpha=0.5, zorder = 0, label = r'Mg II 2800')

    ax2.legend(fontsize = 20, loc =2 )
    ax2.set_ylim(ylim2)

    #print(ma1, mi1,ma2, mi2)

    axes.set_ylabel(r"$F_{\lambda}\ \left[ 10^{-17}\ {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ \AA^{-1} \right]$", fontsize = 30, x = 0.0)
    fig.supxlabel(r"Observed Wavelength $\lambda\ \left[ \AA \right]$", fontsize = 30, y= -0.05)
    #plt.savefig('20240307_example_10_FINALCLASS1_matched.png', bbox_inches = 'tight')





def plot_coadd_limited_combined(spec, ax = None, labels = True, c = 'k', ls = '-', scale = True, lowerlim =3000, upperlim =10000, step = False, smooth = False, scale_amount = 0.5):
    if ax == None:
        fig = plt.figure()
        ax = plt.gca()

    ax.set_xlim(3600, 9800)
    ax.set_ylim(-3.9, 30)  #-- change depending on specific spectrum
    if labels == True:
        ax.set_ylabel(r"$F_{\lambda}\ \left[ 10^{-17}\ {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ \AA^{-1} \right]$")
        ax.set_xlabel(r"wavelength $\lambda\ \left[ \AA \right]$")

    j = 0
    wavelens = spec[0]
    flux = spec[1]

    max_ = np.array([1000,1000])
    min_ = np.array([0,0])
    
    if smooth == True:
        flux = convolve(flux, Gaussian1DKernel(5))

    wavelim = np.where((wavelens>lowerlim)&(wavelens < upperlim))

    wavelens = wavelens[wavelim]
    flux  =flux[wavelim]

    if step == False:
        ax.plot(wavelens, flux,
                color=c, label= "", ls = ls)

    if step == True:
        ax.step(wavelens, flux,
                color=c, label= "", ls = ls)
        
    # Clip even further to ensure the middle of the plot has data
        
    waves_all, fluxes_all =  spec

    min_[:] = np.min(fluxes_all)
    max_[:] = np.max(fluxes_all)


    middle = np.mean([lowerlim, upperlim])
    wavelim_all = np.where((waves_all>(middle-10))&(waves_all < (middle+10)))

    waves_all = waves_all[wavelim_all]
    fluxes_all  =fluxes_all[wavelim_all]
    
    if len(fluxes_all)<1:
        min_[:] = 1000
        max_[:] = 0
    
 
    if scale == True: 
        ax.set_ylim(np.min(min_)-scale_amount, np.max(max_)+scale_amount)


    return max_, min_





def plot_coadd_triptych_from_savedrow(tabrow, fastspec= None, ax = None, labels = True, c = 'k', ls = '-', 
                       scale = True, ylim = None, ylim1 = None, ylim2 = None, addtl_text= None, step = True, smooth = False,
                       hsc_linelabels = False):
    
    data = np.load(glob('Data/consolidated_spectra/*'+str(tabrow['TARGETID'])+'_spectra.npy')[0])

    lam, spec = data

    fig = plt.figure(figsize=(34,8))
    gs = fig.add_gridspec( 1,3,  width_ratios=(2.5, 1,1),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.1, hspace=0.05)

    axes = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    plot_coadd_limited_combined(data, ax = axes, labels = False, c = c, scale = True, step = False, smooth = True, scale_amount=2.)


    z = tabrow['Z']
    zphot = tabrow['best_z']
    zphot_type = tabrow['ZTYPE']
    id = tabrow['TARGETID']
    class_ = tabrow['FINALCLASS']
    utype = tabrow['UMAP_TYPE']
    gw3 = get_g_W3_from_table(tabrow)

    
    alpha, A, sigmas = fit_spectra_gamma_saved([lam, spec], tabrow)

    
    axes.text(0.02,0.03, 'DESI Target ID = {}'.format(id), transform=axes.transAxes)
    axes.text(0.2,0.9, 'Z(DESI) = {:0.2f}'.format(z), transform=axes.transAxes)
    axes.text(0.2,0.8, 'Z(HSC) = {:0.2f}'.format(zphot), transform=axes.transAxes, color= 'r')
    axes.text(0.2,0.85, 'ZTYPE(HSC) = {}'.format(zphot_type), transform=axes.transAxes)
    axes.text(0.02,0.9, 'FINALCLASS = {}'.format(class_), transform=axes.transAxes)
    axes.text(0.02,0.85, r'$g-W3$ = {:0.2f}'.format(gw3), transform=axes.transAxes)
    axes.text(0.02,0.8, 'UMAP TYPE = {}'.format(utype), transform=axes.transAxes)
    axes.text(0.02,0.75, r'$\gamma \,[ \lambda ^{{- \gamma}}]$ = {{{:0.2f}}}'.format(alpha*-1.), transform=axes.transAxes)


    axes.text(0.65,0.95, r'Manual Fits (Restframe)', transform=axes.transAxes)
    axes.text(0.65,0.9, r'FWHM($H\alpha$) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['Ha']), s_to_fwhm(sigmas['Ha'])*line_dictionary['Ha']/2.99e5 ), transform=axes.transAxes)
    axes.text(0.65,0.85, r'FWHM($H\beta$) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['Hb']), s_to_fwhm(sigmas['Hb'])*line_dictionary['Hb']/2.99e5 ), transform=axes.transAxes)
    axes.text(0.65,0.8, r'FWHM(Mg II) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['Mg2']), s_to_fwhm(sigmas['Mg2'])*line_dictionary['Mg2']/2.99e5 ), transform=axes.transAxes)
    axes.text(0.65,0.75, r'FWHM(C IV) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['C4']), s_to_fwhm(sigmas['C4'])*line_dictionary['C4']/2.99e5 ), transform=axes.transAxes)


    if fastspec != None:
        fastspec_info = fastspec[id == fastspec['TARGETID']]
        if len(fastspec_info['Z'])>1:
            print('WARNING: More than one row of fastspec data matched with TARGETID')
        fast_spec_z = fastspec_info['Z'][0]
        ha_s = fastspec_info['HALPHA_SIGMA'][0] # gaussian sigma, km/s 
        hb_s = fastspec_info['HBETA_SIGMA'][0]
        mgii1_s = fastspec_info['MGII_2796_SIGMA'][0]
        civ_s = fastspec_info['CIV_1549_SIGMA'][0]

        
        axes.text(0.4,0.95, r'FASTSPEC Fits', transform=axes.transAxes)
        axes.text(0.4,0.9, r'FWHM($H\alpha$) = {:0.2f} km/s'.format(s_to_fwhm(ha_s)), transform=axes.transAxes)
        axes.text(0.4,0.85, r'FWHM($H\beta$) = {:0.2f} km/s'.format(s_to_fwhm(hb_s)), transform=axes.transAxes)
        axes.text(0.4,0.8, r'FWHM(Mg II) = {:0.2f} km/s'.format(s_to_fwhm(mgii1_s)), transform=axes.transAxes)
        axes.text(0.4,0.75, r'FWHM(C IV) = {:0.2f} km/s'.format(s_to_fwhm(civ_s)), transform=axes.transAxes)
        
        axes.text(0.2,0.75, 'Z(FASTSPEC) = {:0.2f}'.format(fast_spec_z), transform=axes.transAxes)


    plot_all_emission_lines(axes, z)
    plot_all_emission_lines(axes, zphot, c ='r', line_labels= hsc_linelabels)
    axes.set_ylim(ylim)

    ma1, mi1 = plot_coadd_limited_combined(data, ax = ax1, labels = False, c = c, lowerlim=(4900-200)*(1+z), upperlim=(4900+200)*(1+z), step = step, smooth = smooth)
    if np.max(ma1) > 0.:
        ax1.set_xlim([(4900-200)*(1+z), (4900+200)*(1+z)]) 

        ax1.axvline(line_dictionary['Hb']*(1+z), ls = ':', label = r'$H \beta$', alpha=0.5, zorder = 0)
        ax1.axvline(line_dictionary['O31']*(1+z), ls = '--', c = 'k',label = r'O III', alpha=0.5, zorder = 0)
        ax1.axvline(line_dictionary['O32']*(1+z), ls = '--', c = 'k',label = None, alpha=0.5, zorder = 0)

    
    if np.max(ma1) == 0.:
        plot_coadd_limited_combined(data, ax = ax1, labels = False, c = c, lowerlim=(1909-100)*(1+z), upperlim=(1909+100)*(1+z), step = step, smooth = smooth)
        ax1.set_xlim([(1909-100)*(1+z), (1909+100)*(1+z)]) 
        ax1.axvline(line_dictionary['C3']*(1+z), ls = ':', label = r'C III', alpha=0.5, zorder = 0)



    ax1.set_ylim(ylim1)
    ax1.legend(fontsize = 20, loc =2 )


    ma2, mi2 = plot_coadd_limited_combined(data, ax = ax2, labels = False, c = c, lowerlim=(6564-100)*(1+z), upperlim=(6564+100)*(1+z), step = step, smooth = smooth)
    if np.max(ma2) > 0.:
        ax2.set_xlim([(6564-100)*(1+z), (6564+100)*(1+z)]) 
        ax2.axvline(line_dictionary['Ha']*(1+z), ls = ':', alpha=0.5, zorder = 0, label = r'$H \alpha$')

    if np.max(ma2) == 0.:
        plot_coadd_limited_combined(data, ax = ax2, labels = False, c = c, lowerlim=(2800-100)*(1+z), upperlim=(2800+100)*(1+z), step = step, smooth = smooth)
        ax2.set_xlim([(2800-100)*(1+z), (2800+100)*(1+z)]) 

        ax2.axvline(line_dictionary['Mg2']*(1+z), ls = ':', alpha=0.5, zorder = 0, label = r'Mg II 2800')

    ax2.legend(fontsize = 20, loc =2 )
    ax2.set_ylim(ylim2)

    #print(ma1, mi1,ma2, mi2)

    axes.set_ylabel(r"$F_{\lambda}\ \left[ 10^{-17}\ {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ \AA^{-1} \right]$", fontsize = 30, x = 0.0)
    fig.supxlabel(r"Observed Wavelength $\lambda\ \left[ \AA \right]$", fontsize = 30, y= -0.05)
    #plt.savefig('20240307_example_10_FINALCLASS1_matched.png', bbox_inches = 'tight')







def plot_coadd_triptych_from_number(targetid, table, fastspec= None, ax = None, labels = True, c = 'k', ls = '-', 
                       scale = True, ylim = None, ylim1 = None, ylim2 = None, addtl_text= None, step = True, smooth = False,
                       hsc_linelabels = False):
    
    data = np.load(glob('Data/consolidated_spectra/*'+str(targetid)+'_spectra.npy')[0])
    tabrow = table[table['TARGETID'] == targetid]

    if len(tabrow) > 1:
        print('Multiple matches in table')

    if len(tabrow) == 0:
        print('Match not found in table')

    lam, spec = data

    fig = plt.figure(figsize=(34,8))
    gs = fig.add_gridspec( 1,3,  width_ratios=(2.5, 1,1),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.1, hspace=0.05)

    axes = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    plot_coadd_limited_combined(data, ax = axes, labels = False, c = c, scale = True, step = False, smooth = True, scale_amount=2.)


    z = np.array(tabrow['Z'])[0] #tabrow['Z']
    zphot = np.array(tabrow['best_z'])[0] #tabrow['best_z']
    zphot_type = np.array(tabrow['ZTYPE'])[0] #tabrow['ZTYPE']
    id = np.array(tabrow['TARGETID'])[0] #tabrow['TARGETID']
    class_ = np.array(tabrow['FINALCLASS'])[0] #tabrow['FINALCLASS']
    utype = np.array(tabrow['UMAP_TYPE'])[0] #tabrow['UMAP_TYPE']
    gw3 = np.array(get_g_W3_from_table(tabrow))[0]

    
    alpha, A, sigmas = fit_spectra_gamma_saved([lam, spec], tabrow)

    
    axes.text(0.02,0.03, 'DESI Target ID = {}'.format(id), transform=axes.transAxes)
    axes.text(0.2,0.9, 'Z(DESI) = {:0.2f}'.format(z), transform=axes.transAxes)
    axes.text(0.2,0.8, 'Z(HSC) = {:0.2f}'.format(zphot), transform=axes.transAxes, color= 'r')
    axes.text(0.2,0.85, 'ZTYPE(HSC) = {}'.format(zphot_type), transform=axes.transAxes)
    axes.text(0.02,0.9, 'FINALCLASS = {}'.format(class_), transform=axes.transAxes)
    axes.text(0.02,0.85, r'$g-W3$ = {:0.2f}'.format(gw3), transform=axes.transAxes)
    axes.text(0.02,0.8, 'UMAP TYPE = {}'.format(utype), transform=axes.transAxes)
    axes.text(0.02,0.75, r'$\gamma \,[ \lambda ^{{- \gamma}}]$ = {{{:0.2f}}}'.format(alpha*-1.), transform=axes.transAxes)


    axes.text(0.65,0.95, r'Manual Fits (Restframe)', transform=axes.transAxes)
    axes.text(0.65,0.9, r'FWHM($H\alpha$) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['Ha']), s_to_fwhm(sigmas['Ha'])*line_dictionary['Ha']/2.99e5 ), transform=axes.transAxes)
    axes.text(0.65,0.85, r'FWHM($H\beta$) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['Hb']), s_to_fwhm(sigmas['Hb'])*line_dictionary['Hb']/2.99e5 ), transform=axes.transAxes)
    axes.text(0.65,0.8, r'FWHM(Mg II) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['Mg2']), s_to_fwhm(sigmas['Mg2'])*line_dictionary['Mg2']/2.99e5 ), transform=axes.transAxes)
    axes.text(0.65,0.75, r'FWHM(C IV) = {:0.2f} km/s = {:0.2f} $\AA$'.format(s_to_fwhm(sigmas['C4']), s_to_fwhm(sigmas['C4'])*line_dictionary['C4']/2.99e5 ), transform=axes.transAxes)


    if fastspec != None:
        fastspec_info = fastspec[id == fastspec['TARGETID']]
        if len(fastspec_info['Z'])>1:
            print('WARNING: More than one row of fastspec data matched with TARGETID')
        fast_spec_z = fastspec_info['Z'][0]
        ha_s = fastspec_info['HALPHA_SIGMA'][0] # gaussian sigma, km/s 
        hb_s = fastspec_info['HBETA_SIGMA'][0]
        mgii1_s = fastspec_info['MGII_2796_SIGMA'][0]
        civ_s = fastspec_info['CIV_1549_SIGMA'][0]

        
        axes.text(0.4,0.95, r'FASTSPEC Fits', transform=axes.transAxes)
        axes.text(0.4,0.9, r'FWHM($H\alpha$) = {:0.2f} km/s'.format(s_to_fwhm(ha_s)), transform=axes.transAxes)
        axes.text(0.4,0.85, r'FWHM($H\beta$) = {:0.2f} km/s'.format(s_to_fwhm(hb_s)), transform=axes.transAxes)
        axes.text(0.4,0.8, r'FWHM(Mg II) = {:0.2f} km/s'.format(s_to_fwhm(mgii1_s)), transform=axes.transAxes)
        axes.text(0.4,0.75, r'FWHM(C IV) = {:0.2f} km/s'.format(s_to_fwhm(civ_s)), transform=axes.transAxes)
        
        axes.text(0.2,0.75, 'Z(FASTSPEC) = {:0.2f}'.format(fast_spec_z), transform=axes.transAxes)


    plot_all_emission_lines(axes, z)
    plot_all_emission_lines(axes, zphot, c ='r', line_labels= hsc_linelabels)
    axes.set_ylim(ylim)

    ma1, mi1 = plot_coadd_limited_combined(data, ax = ax1, labels = False, c = c, lowerlim=(4900-200)*(1+z), upperlim=(4900+200)*(1+z), step = step, smooth = smooth)
    if np.max(ma1) > 0.:
        ax1.set_xlim([(4900-200)*(1+z), (4900+200)*(1+z)]) 

        ax1.axvline(line_dictionary['Hb']*(1+z), ls = ':', label = r'$H \beta$', alpha=0.5, zorder = 0)
        ax1.axvline(line_dictionary['O31']*(1+z), ls = '--', c = 'k',label = r'O III', alpha=0.5, zorder = 0)
        ax1.axvline(line_dictionary['O32']*(1+z), ls = '--', c = 'k',label = None, alpha=0.5, zorder = 0)

    
    if np.max(ma1) == 0.:
        plot_coadd_limited_combined(data, ax = ax1, labels = False, c = c, lowerlim=(1909-100)*(1+z), upperlim=(1909+100)*(1+z), step = step, smooth = smooth)
        ax1.set_xlim([(1909-100)*(1+z), (1909+100)*(1+z)]) 
        ax1.axvline(line_dictionary['C3']*(1+z), ls = ':', label = r'C III', alpha=0.5, zorder = 0)



    ax1.set_ylim(ylim1)
    ax1.legend(fontsize = 20, loc =2 )


    ma2, mi2 = plot_coadd_limited_combined(data, ax = ax2, labels = False, c = c, lowerlim=(6564-100)*(1+z), upperlim=(6564+100)*(1+z), step = step, smooth = smooth)
    if np.max(ma2) > 0.:
        ax2.set_xlim([(6564-100)*(1+z), (6564+100)*(1+z)]) 
        ax2.axvline(line_dictionary['Ha']*(1+z), ls = ':', alpha=0.5, zorder = 0, label = r'$H \alpha$')

    if np.max(ma2) == 0.:
        plot_coadd_limited_combined(data, ax = ax2, labels = False, c = c, lowerlim=(2800-100)*(1+z), upperlim=(2800+100)*(1+z), step = step, smooth = smooth)
        ax2.set_xlim([(2800-100)*(1+z), (2800+100)*(1+z)]) 

        ax2.axvline(line_dictionary['Mg2']*(1+z), ls = ':', alpha=0.5, zorder = 0, label = r'Mg II 2800')

    ax2.legend(fontsize = 20, loc =2 )
    ax2.set_ylim(ylim2)

    #print(ma1, mi1,ma2, mi2)

    axes.set_ylabel(r"$F_{\lambda}\ \left[ 10^{-17}\ {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ \AA^{-1} \right]$", fontsize = 30, x = 0.0)
    fig.supxlabel(r"Observed Wavelength $\lambda\ \left[ \AA \right]$", fontsize = 30, y= -0.05)
    #plt.savefig('20240307_example_10_FINALCLASS1_matched.png', bbox_inches = 'tight')








def add_new_columns_to_table(tab):
    line_names = np.array(list(line_dictionary.keys()))
    fwhm_names= np.array([i+'_FWHM' for i in line_names])

    [tab.add_column(0.0, name = i) for i in fwhm_names]
    tab.add_column(0.0, name = 'PLAW_GAMMA')

    return tab


def add_fit_info_to_tabrow(coadd_spec, tabrow):

    alpha, A, sigmas = fit_spectra_gamma(coadd_spec, tabrow)

    line_names = np.array(list(line_dictionary.keys()))
    fwhm_names= np.array([i+'_FWHM' for i in line_names])

    for i in np.arange(len(fwhm_names)):
        fwhm = s_to_fwhm(sigmas[line_names[i]])
        tabrow[fwhm_names[i]] = fwhm
        tabrow['PLAW_GAMMA'] = -1*alpha

    return tabrow


def add_fit_info_to_tabrow_saved(coadd_spec, tabrow):

    alpha, A, sigmas = fit_spectra_gamma_saved(coadd_spec, tabrow)

    line_names = np.array(list(line_dictionary.keys()))
    fwhm_names= np.array([i+'_FWHM' for i in line_names])

    for i in np.arange(len(fwhm_names)):
        fwhm = s_to_fwhm(sigmas[line_names[i]])
        tabrow[fwhm_names[i]] = fwhm
        tabrow['PLAW_GAMMA'] = -1*alpha

    return tabrow


def save_spec_file(spec, tabrow, savedir = './'):

    waves, fluxes = combine_spec_arms(spec)

    np.save(savedir+'/'+str(tabrow['TARGETID'])+'_spectra.npy', np.array([waves,fluxes]))

def save_spec_file_ivar(spec, tabrow, savedir = './'):

    waves, fluxes, ivar = combine_spec_arms_ivar(spec)

    np.save(savedir+'/'+str(tabrow['TARGETID'])+'_spectra.npy', np.array([waves,fluxes, ivar]))



