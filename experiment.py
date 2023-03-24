import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cmocean
from functools import cached_property
from scipy import interpolate
import xrft

class Experiment():
    def __init__(self, folder):
        self.folder = folder
        for j, file in enumerate(['GYRE_360d_00010101_00301230_ocean.nc', 
            'GYRE_7200d_00010101_01210105_UVTgrids.nc', 
            'GYRE_7200d_00010101_00210102_UVTgrids.nc', 
            'GYRE_7200d_00010101_00210105_UVTgrids.nc']):
            path = os.path.join(folder, file)
            if os.path.exists(path):
                self.ds = xr.open_dataset(path)
                if j==0:
                    self.ds = self.ds.isel(time_counter=slice(-20,None)).mean(dim='time_counter')
                else:
                    self.ds = self.ds.isel(time_counter=slice(-1,None)).mean(dim='time_counter')
        
        for file in ['GYRE_5d_00010101_00301230_surf.nc',
                     'ssh.nc']:
            path = os.path.join(folder, file)
            if os.path.exists(path):
                self.surf = xr.open_dataset(path)
                if file == 'ssh.nc':
                    self.surf = self.surf.coarsen(time_counter=5, boundary='trim').mean().rename(
                    {'nav_lat_grid_T': 'latT', 'nav_lon_grid_T': 'lonT',
                    'x_grid_T': 'xT', 'y_grid_T': 'yT'}
                    )
                else:
                    self.surf = self.surf.rename(
                    {'nav_lat': 'latT', 'nav_lon': 'lonT',
                    'x': 'xT', 'y': 'yT'}
                )
                if len(self.surf.time_counter) == 75:
                    self.surf = self.surf.isel(time_counter=slice(0,72))
                elif len(self.surf.time_counter) > 1000:
                    self.surf = self.surf.isel(time_counter=slice(-72,None))

        if len(self.ds.x_grid_T) == 122:
            self.mask = xr.open_dataset('mesh_mask_R4.nc')
        elif len(self.ds.x_grid_T) == 272:
            self.mask = xr.open_dataset('mesh_mask_R9.nc')

        self.ds = self.ds.rename(
                        {'x_grid_T': 'xT', 'y_grid_T': 'yT',
                         'x_grid_U': 'xU', 'y_grid_U': 'yU',
                         'x_grid_V': 'xV', 'y_grid_V': 'yV',
                         'nav_lat_grid_T': 'latT', 'nav_lon_grid_T': 'lonT',
                         'nav_lat_grid_U': 'latU', 'nav_lon_grid_U': 'lonU',
                         'nav_lat_grid_V': 'latV', 'nav_lon_grid_V': 'lonV'})
        dx = float(self.mask.e1t[0,0,0])
        self.ds['xT'] = self.ds['xT'] * dx - dx/2
        self.ds['xU'] = self.ds['xU'] * dx
        self.ds['xV'] = self.ds['xV'] * dx - dx/2
        self.ds['yT'] = self.ds['yT'] * dx - dx/2
        self.ds['yU'] = self.ds['yU'] * dx - dx/2
        self.ds['yV'] = self.ds['yV'] * dx

        self.surf['xT'] = self.ds['xT']
        self.surf['yT'] = self.ds['yT']
        return
    
    @property
    def u(self):
        return self.ds.vozocrtx
    
    @property
    def v(self):
        return self.ds.vomecrty
    
    @property
    def u2(self):
        return self.ds.squaredu

    @property
    def v2(self):
        return self.ds.squaredv.data + self.v*0 # bug with coordinates

    @property
    def T(self):
        return self.ds.votemper
    
    @property
    def SST(self):
        return self.T.isel(deptht=0)
    
    @property
    def SSH(self):
        return self.ds.sossheig
    
    @property
    def Tu(self):
        return self.ds.veltempx
    
    @property
    def Tv(self):
        return self.ds.veltempy

    @property
    def u_surf(self):
        # see https://os.copernicus.org/articles/15/477/2019/
        dx = grid_step = float(self.mask.e1t[0,0,0])
        ssh = self.surf.sossheig.data
        g = 9.8
        f = self.mask.ff.squeeze().data
        u = 0*ssh
        u[:,:-1,:] = - (ssh[:,1:,:] - ssh[:,:-1,:]) / dx
        u = u * g / f
        return u + self.surf.sossheig*0
    
    @property
    def v_surf(self):
        # see https://os.copernicus.org/articles/15/477/2019/
        dx = grid_step = float(self.mask.e1t[0,0,0])
        ssh = self.surf.sossheig.data
        g = 9.8
        f = self.mask.ff.squeeze().data
        v = 0*ssh
        v[:,:,:-1] = (ssh[:,:,1:] - ssh[:,:,:-1]) / dx
        v = v * g / f
        return v + self.surf.sossheig*0
    
    @property
    def uzonal_barotropic(self):
        U = (remesh(self.u,self.T) - remesh(self.v,self.T)) / np.sqrt(2)
        return self.mean_z(U)
    
    def zonal_section(self, field, Lon=-75, Lat=np.linspace(20,40,100)):
        x = field[lon(field)].data
        y = field[lat(field)].data
        z = field.data

        u = np.zeros((z.shape[0],Lat.shape[0]))
        for k in range(z.shape[0]):
            f = interpolate.LinearNDInterpolator(list(zip(x.ravel(),y.ravel())),z[k].ravel())
            u[k,:] = f(Lon,Lat)
        return xr.DataArray(u, dims=('depth', 'lat'), coords={'lat': Lat, 'depth': field.deptht.data})
    
    @cached_property
    def uzonal_section(self):
        U = (remesh(self.u,self.T) - remesh(self.v,self.T)) / np.sqrt(2)
        return self.zonal_section(U, Lon=-75, Lat=np.linspace(20,40,100))
    
    @cached_property
    def Tzonal_section(self):
        return self.zonal_section(self.T, Lon=-72, Lat=np.linspace(20,40,100))
    
    @cached_property
    def vabs(self):
        return np.sqrt(remesh(self.u2,self.T) + remesh(self.v2,self.T))
    
    @cached_property
    def EKE(self):
        field = self.u2.data + self.v2.data - self.u.data**2 - self.v.data**2
        field = 0.5 * field + self.T*0
        return field
    
    @cached_property
    def EKEz(self):
        return self.EKE.isel(xT=slice(1,-1), yT=slice(1,-1)).mean(['xT', 'yT'])
    
    @property
    def EKEs(self):
        return self.EKE.isel(deptht=0)
    
    @property
    def EKE_level(self):
        return float(self.mean_z(self.EKEz).values)
    
    @property
    def EKE_spectrum(self):
        #x=(500e+3,1500e+3); y=(1000e+3,2000e+3)
        x = (0,3180e+3)
        y = (0,2120e+3)
        u = self.u_surf.sel(xT=slice(*x), yT=slice(*y))
        v = self.v_surf.sel(xT=slice(*x), yT=slice(*y))
        print(u.shape, v.shape)
        u = u - u.mean('time_counter')
        v = v - v.mean('time_counter')
        return compute_isotropic_KE(u.drop(['latT', 'lonT']), v.drop(['latT', 'lonT']), window=None, 
        nfactor=2, truncate=True, detrend='linear', window_correction=False).mean('time_counter')
    
    @cached_property
    def MOC(self):
        Vmerid = self.meridional_flux(self.u, self.v)
        Psi = 0 * Vmerid

        nz = len(self.T.deptht)
        e3t = self.mask.e3t_1d.squeeze().data

        for k in range(nz-2, -1, -1):
            Psi[:,k] = Psi[:,k+1] + e3t[k] * Vmerid[:,k]
        
        return Psi / 1.e+6 # to Sverdrups 
    
    @cached_property
    def heat_flux2d(self):
        Qx = self.Tu.data - self.u.data * T_to_u(self.T.data)
        Qy = self.Tv.data - self.v.data * T_to_v(self.T.data)
        
        umask = self.mask.umask.data[0]
        vmask = self.mask.vmask.data[0]
        e3t = self.mask.e3t_1d.squeeze().data
        rau0 = 1026.
        rcp = 3991.86795711963

        Qx = Qx * umask * rau0 * rcp
        Qy = Qy * vmask * rau0 * rcp

        return self.meridional_flux(Qx, Qy)
    
    @cached_property
    def heat_flux(self):
        return self.integrate_z(self.heat_flux2d)
    
    def meridional_flux(self, Qx, Qy, Nlat=100):
        (nz, ny, nx) = Qx.shape
        if isinstance(Qx, xr.DataArray):
            Qx = Qx.data
            Qy = Qy.data

        grid_step = float(self.mask.e1t[0,0,0])
        lat = self.T.latT.data

        divQ = np.zeros((nz, ny, nx))
        for i in range(1,nx):
            for j in range(1,ny):
                divQ[:,j,i] = (Qx[:,j,i] - Qx[:,j,i-1]) / grid_step \
                            + (Qy[:,j,i] - Qy[:,j-1,i]) / grid_step

        llat = np.linspace(15.35,49.38, Nlat)

        Q = np.zeros((Nlat,nz))
        for ilat in range(Nlat):
            for k in range(nz):
                Q[ilat,k] = np.sum(np.multiply(divQ[k,:,:],lat<llat[ilat])) * grid_step ** 2
        
        return xr.DataArray(Q, dims=('lat', 'depth'), coords={'lat': llat, 'depth': self.T.deptht.data})
    
    def integrate_z(self, field):
        d = depth(field)
        e3t = self.mask.e3t_1d.squeeze().rename({'z': d})
        return (e3t*field).sum(dim=d)
    
    def mean_z(self, field):
        d = depth(field)
        e3t = self.mask.e3t_1d.squeeze().rename({'z': d})
        return (e3t*field).isel({d: slice(0,-1)}).sum(dim=d) / e3t.isel({d: slice(0,-1)}).sum(dim=d)
    
    def test_flux(self):
        nz,ny,nx = self.T.shape
        psi = np.zeros((nz,ny,nx))
        psi[:nz-1,1:-1,1:-1] = np.random.randn(nz-1,ny-2,nx-2)
        u = 0*psi
        v = 0*psi
        for i in range(1,nx):
            for j in range(1,ny):
                u[:,j,i] = psi[:,j,i] - psi[:,j-1,i]
                v[:,j,i] = -psi[:,j,i] + psi[:,j,i-1]
        
        Q = self.integrate_z(self.meridional_flux(u,v))
        print('Typical value should be 1e-7, ', np.max(np.abs(Q)).data)

    def lk_error(self, field, target, k=2):
        dd = depth(field)
        xx = x(field)
        yy = y(field)

        if xx is not None:
            error= field - remesh(target,field)
        else:
            error = field - target
        
        if dd is not None:
            if k > 0:
                e3t = self.mask.e3t_1d.squeeze().rename({'z': dd})
                error = error * e3t
                e3t = e3t.isel({dd: slice(0,-1)})
            error = error.isel({dd: slice(0,-1)})
        if xx is not None:
            error = error.isel({xx: slice(1,-1)})
        if yy is not None:
            error = error.isel({yy: slice(1,-1)})

        if k > 0:
            out = ((np.abs(error)**k).mean())**(1./k)
            if dd is not None:
                out = out /  ((np.abs(e3t)**k).mean())**(1./k)
        else:
            out = np.abs(error).max()
        return float(out.values)

    def error(self, target):
        '''
        target is the reference experiment with probably 
        different resolution
        '''
        d = {}
        for key in ['MOC', 'heat_flux2d', 'heat_flux', 'SST', 'SSH', 'uzonal_barotropic', 'uzonal_section', 'EKEs', 'EKEz']:
              d[key] = self.lk_error(self.__getattribute__(key), target.__getattribute__(key))

        d['EKE_ratio'] = self.EKE_level / target.EKE_level
        return d
        
    def plot_2D(self, var, **kw):
        if isinstance(var,str):
            field = self.__getattribute__(var)
        elif isinstance(var, xr.DataArray):
            field = var
        else:
            raise ValueError('var must be a string or a DataArray')

        lon, lat, d = coords(field)

        try:
            field = field.isel({d: 0})
        except:
            pass

        field = field.isel({x(field): slice(1,-1), y(field): slice(1,-1)})
        
        fun = field.plot.contourf if 'levels' in kw else field.plot

        fun(x=lon, y=lat, 
            cmap=cmocean.cm.balance, **kw)
        plt.gca().set_aspect(aspect=1)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

    def plot_EKE(self, **kw):
        self.EKE.isel(deptht=0,xT=slice(1,-1), yT=slice(1,-1)).plot.contourf(x='lonT', y='latT', 
            levels=np.linspace(0,0.3,11), vmin=0, cmap=cmocean.cm.balance, 
            vmax=0.4,
            cbar_kwargs = {'label': 'EKE, $m^2/s^2$'}, **kw)
        plt.gca().set_aspect(aspect=1)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Surface EKE')
    
    def plot_EKEmean(self, **kw):
        self.plot_2D(self.mean_z(self.EKE), levels=15, cbar_kwargs = {'label': 'EKE, $m^2/s^2$'}, **kw)
        plt.title('Depth-averaged EKE')

    def plot_EKEz(self, **kw):
        plt.plot(self.EKEz, self.EKEz.deptht, **kw)
        plt.xlabel('EKE, $m^2/s^2$')
        plt.ylabel('depth, $m$')
        plt.grid()
        plt.xlim(0 , 0.03)
        plt.ylim(4000, -100)
        plt.title('Lateral mean EKE')
        plt.xticks([0,0.005,0.01,0.015,0.02,0.025,0.03],['0','0.005','0.01','0.015', '0.02', '0.025', '0.03']);
        plt.yticks([0, 1000, 2000, 3000, 4000], ['0', '1000', '2000', '3000', '4000'])
    
    def plot_SST(self, target=None, **kw):
        '''
        If target is not none, Error,
        self.T - target.T
        is computed
        '''
        if target is None:
            T = self.T
        else:
            T = self.T - remesh(target.T,self.T)
        T = T.isel(deptht=0).isel(xT=slice(1,-1), yT=slice(1,-1))

        levels = np.arange(8,30,1) if target is None else np.linspace(-3,3,11)
        
        T.plot.contourf(x='lonT', y='latT', 
            levels=levels, vmin=11, vmax=24, 
            cmap=cmocean.cm.balance,
            cbar_kwargs = {'label': 'SST, $^oC$'}, **kw)
        Cplot = T.plot.contour(x='lonT', y='latT',
            levels=levels, colors='k', linewidths=0.5)
        plt.gca().set_aspect(aspect=1)
        plt.gca().clabel(Cplot, Cplot.levels)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')    
        plt.title('SST')

    def plot_SSH(self, target=None, **kw):
        if target is None:
            T = self.SSH
        else:
            T = self.SSH - remesh(target.SSH,self.SSH)

        T = T.isel(xT=slice(1,-1), yT=slice(1,-1))
        T.plot.contourf(x='lonT', y='latT', 
            levels=np.linspace(-0.6,0.6,9),
            cmap=cmocean.cm.balance,
            cbar_kwargs = {'label': 'SSH, $m$'}, **kw)
        Cplot = T.plot.contour(x='lonT', y='latT',
            levels=np.linspace(-0.6,0.6,9), colors='k', linewidths=0.5)
        plt.gca().set_aspect(aspect=1)
        plt.gca().clabel(Cplot, Cplot.levels)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')    
        plt.title('SSH')

    def plot_uzonal(self, target = None):
        if target is None:
            uzonal = self.uzonal_barotropic
        else:
            uzonal = self.uzonal_barotropic - remesh(target.uzonal_barotropic,self.uzonal_barotropic)
        uzonal = uzonal.isel(xT=slice(1,-1), yT=slice(1,-1))

        uzonal.plot.contourf(x='lonT', y='latT', 
            levels=np.linspace(-0.1,0.1,9),
            cmap=cmocean.cm.balance,
            cbar_kwargs = {'label': 'Barotropic zonal velocity, $m/s$'})
        #Cplot = uzonal.plot.contour(x='lonU', y='latU',
        #    levels=np.linspace(-0.1,0.1,9), colors='k', linewidths=0.5)
        plt.gca().set_aspect(aspect=1)
        #plt.gca().clabel(Cplot, Cplot.levels)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')    
        

    def plot_MOC_heat(self, target=None):
        if target is None:
            heat_flux2d = self.heat_flux2d
            MOC = self.MOC
        else:
            heat_flux2d = self.heat_flux2d - target.heat_flux2d
            MOC = self.MOC - target.MOC

        heat_flux2d.plot.contourf(
                x='lat', y='depth', levels=np.linspace(-2.5e+11,2.5e+11,11),
                vmin=-2.5e+11, vmax=2.5e+11, cmap=cmocean.cm.balance,
                cbar_kwargs = {'label': 'Eddy Meridional Heat Flux, $W/m$'})
        Cplot = MOC.plot.contour(x='lat', y='depth', levels=np.linspace(-5,5,21), colors='k', linewidths=0.5)
        plt.gca().clabel(Cplot, 
            [-3.0,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,2.0],
            fmt={-3.0: '-3', -2.0: '-2', -1.5: '-1.5', -1.0: '-1', -0.5: '-0.5', 0.0: '0', 0.5: '0.5', 1.0: '1', 2.0: '2'})
        plt.ylim(4000,5)
        plt.xticks([20,25,30,35,40,45])
        plt.yscale('log')
        plt.xlabel('Latitude')
        plt.ylabel('Depth, m')

    def plot_uzonal_section(self, target=None):
        if target is None:
            uzonal = self.uzonal_section
        else:
            uzonal = self.uzonal_section - target.uzonal_section
        
        uzonal.plot.contourf(x='lat', y='depth',
            levels=np.linspace(-0.4,0.4,17),
            cmap=cmocean.cm.balance,
            cbar_kwargs = {'label': 'Zonal velocity, $m/s$'})
        Cplot = uzonal.plot.contour(x='lat', y='depth',
            levels=np.linspace(-0.4,0.4,17), colors='k', linewidths=0.5)
        plt.gca().clabel(Cplot, Cplot.levels[0:None:2])
        plt.ylim(1000,5)
        plt.xticks([20,25,30,35,40])
        plt.xlim([20,38])
        #plt.yscale('log')
        plt.yticks([0, 200, 400, 600, 800, 1000], ['0', '200', '400', '600', '800', '1000'])
        plt.xlabel('Latitude at 75W')
        plt.ylabel('Depth, m')

    def plot_Tzonal_section(self, target):
        levels=np.arange(4,25,2)
        self.Tzonal_section.plot.contourf(x='lat', y='depth',
            levels=levels,
            cmap=cmocean.cm.balance,
            cbar_kwargs = {'label': 'Temperature, $^oC$'})
        Cplot = target.Tzonal_section.plot.contour(x='lat', y='depth',
            levels=levels, colors='k', linewidths=0.5, linestyles='-')
        plt.gca().clabel(Cplot, Cplot.levels[0:None:2])
        plt.ylim(700,0)
        plt.xticks([20,25,30,35,40])
        plt.yticks([0, 200, 400, 600, 800], ['0', '200', '400', '600', '800'])
        plt.xlabel('Latitude at 72W')
        plt.ylabel('Depth, m')
        plt.plot(np.nan,np.nan,ls='-',color='k', lw=1, label='$1/9^o$')
        plt.legend(loc='lower right')
        
def lon(field):
    for lon_out in ['lonT', 'lonU', 'lonV']:
        if lon_out in field.coords:
            break
    return lon_out

def lat(field):
    for lat_out in ['latT', 'latU', 'latV']:
        if lat_out in field.coords:
            break
    return lat_out

def x(field):
    for x_out in ['xT', 'xU', 'xV']:
        if x_out in field.dims:
            break
    return x_out if x_out in field.dims else None

def y(field):
    for y_out in ['yT', 'yU', 'yV']:
        if y_out in field.dims:
            break
    return y_out if y_out in field.dims else None

def depth(field):
    for depth_out in ['deptht', 'depthu', 'depthv', 'depth']:
        if depth_out in field.coords:
            break
    return depth_out if depth_out in field.dims else None

def coords(field):
    return lon(field), lat(field), depth(field)

def T_to_u(T):
    (nz, ny, nx) = T.shape
    u = np.zeros((nz, ny, nx))
    for i in range(0,nx-1):
        u[:,:,i] = 0.5 * (T[:,:,i] + T[:,:,i+1])
    return u

def T_to_v(T):
    (nz, ny, nx) = T.shape
    v = np.zeros((nz, ny, nx))
    for j in range(0,ny-1):
        v[:,j,:] = 0.5 * (T[:,j,:] + T[:,j+1,:])
    return v

def remesh(field, target):
    x_out, y_out = x(target), y(target)
    x_in, y_in = x(field), y(field)

    if x_in is not None:
        nfactor = int(np.floor(field[x_in].shape[0]/target[x_out].shape[0]))
    else:
        nfactor = -1

    if nfactor > 1:
        field = field.isel({x_in:slice(1,-1), y_in:slice(1,-1)}).coarsen({x_in: nfactor, y_in: nfactor}).mean()
        x_in, y_in = x(field), y(field)

    out = field # just initialization
    if x_in is not None:
        out = field.interp({x_in: target[x_out], y_in: target[y_out]}, method='linear').fillna(0)
        out[lat(target)] = target[lat(target)]
        out[lon(target)] = target[lon(target)]
    if depth(field) in field.coords:
        out = out.rename({depth(out): depth(target)})
        out[depth(target)] = target[depth(target)]
    
    return out

def compute_isotropic_KE(u, v, window='hann', 
        nfactor=2, truncate=True, detrend='linear', window_correction=True):
    
    Eu = xrft.isotropic_power_spectrum(u, dim=('xT','yT'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)
    Ev = xrft.isotropic_power_spectrum(v, dim=('xT','yT'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)

    E = (Eu+Ev) / 2 # because power spectrum is twice the energy
    E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers
    return E