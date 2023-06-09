import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean

def create_animation(fun, idx, filename='my-animation.gif', dpi=200, FPS=18, loop=0):
    '''
    See https://pythonprogramming.altervista.org/png-to-gif/
    fun(i) - a function creating one snapshot, has only one input:
        - number of frame i
    idx - range of frames, i in idx
    FPS - frames per second
    filename - animation name
    dpi - set 300 or so to increase quality
    loop - number of repeats of the gif
    '''
    frames = []
    for i in idx:
        fun(i)
        plt.savefig('.frame.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        frames.append(Image.open('.frame.png').convert('RGB'))
        print(f'Frame {i} is created', end='\r')
    os.system('rm .frame.png')
    # How long to persist one frame in milliseconds to have a desired FPS
    duration = 1000 / FPS
    print(f'Animation {filename} at FPS={FPS} will last for {len(idx)/FPS} seconds')
    frames[0].save(
        filename, format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop)
    
def default_rcParams(kw={}):
    '''
    Also matplotlib.rcParamsDefault contains the default values,
    but:
    - backend is changed
    - without plotting something as initialization,
    inline does not work
    '''
    plt.plot()
    plt.close()
    rcParams = matplotlib.rcParamsDefault.copy()
    try:
        rcParams.pop('backend') # can break inlining
    except:
        pass
    matplotlib.rcParams.update(rcParams)
    
    matplotlib.rcParams.update({
        'font.family': 'MathJax_Main',
        'mathtext.fontset': 'cm',

        'figure.figsize': (4, 4),

        'figure.subplot.wspace': 0.3,
        
        'font.size': 14,
        #'axes.labelsize': 10,
        #'axes.titlesize': 12,
        #'xtick.labelsize': 10,
        #'ytick.labelsize': 10,
        #'legend.fontsize': 10,

        'axes.formatter.limits': (-3,2),
        'axes.formatter.use_mathtext': True,
        'axes.labelpad': 0,
        'axes.titlelocation' : 'center',
        
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    matplotlib.rcParams.update(**kw)

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str
    
def imshow(_q, cbar=True, location='right', cbar_label=None, ax=None, cmap=None, 
    vmax = None, vmin = None, pct=99, axes=False, interpolation='none', normalize='False', normalize_postfix='', **kwargs):

    def rms(x):
        return float(np.sqrt(np.mean(x.astype('float64')**2)))
    def mean(x):
        return float(np.mean(x.astype('float64')))

    if normalize != 'False':
        if normalize == 'mean':
            q_norm = mean(_q)
            q_str = f'$\\mu_x={latex_float(q_norm)}$'
        else:
            q_norm = rms(_q)
            q_str = f'${latex_float(q_norm)}$'    
        q = _q / q_norm
        if len(normalize_postfix) > 0:
            q_str += f' {normalize_postfix}'
    else:
        q = _q

    if q.min() < 0:
        vmax = np.percentile(np.abs(q), pct) if vmax is None else vmax
        vmin = -vmax if vmin is None else vmin
    else:
        vmax = np.percentile(q, pct) if vmax is None else vmax
        vmin = 0 if vmin is None else vmin

    cmap=cmocean.cm.balance if cmap is None else cmap
    
    kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
    
    if ax is None:
        ax = plt.gca()

    # flipud because imshow inverts vertical axis
    im = ax.imshow(np.flipud(q), **kw, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    if axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    if normalize != 'False':
        ax.text(0.05,0.85,q_str,transform = ax.transAxes, fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    if cbar:
        divider = make_axes_locatable(ax)
        if location == 'right':
            cax = divider.append_axes('right', size="5%", pad=0.1)
            cbar_kw = dict()
        elif location == 'bottom':
            cax = divider.append_axes('bottom', size="5%", pad=0.1)
            cbar_kw = dict(orientation='horizontal')
        cb = plt.colorbar(im, cax = cax, label=cbar_label, **cbar_kw)
    
    # Return axis to initial image
    plt.sca(ax)
    return im