import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cmocean

class Experiment():
    def __init__(self, folder):
        self.folder = folder
        for j, file in enumerate(['GYRE_360d_00010101_00301230_ocean.nc', 
            'GYRE_7200d_00010101_01210105_UVTgrids.nc', 
            'GYRE_7200d_00010101_00210102_UVTgrids.nc']):
            path = os.path.join(folder, file)
            if os.path.exists(path):
                self.ds = xr.open_dataset(path)
                if j==0:
                    self.ds = self.ds.isel(time_counter=slice(-20,None)).mean(dim='time_counter')
                else:
                    self.ds = self.ds.isel(time_counter=slice(-1,None)).mean(dim='time_counter')

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
    def Tu(self):
        return self.ds.veltempx
    
    @property
    def Tv(self):
        return self.ds.veltempy
    
    @property
    def vabs(self):
        return np.sqrt(self.u2.data + self.v2.data) + self.T*0
    
    @property
    def EKE(self):
        field = self.u2.data + self.v2.data - self.u.data**2 - self.v.data**2
        field = 0.5 * field + self.T*0
        return field
    
    @property
    def EKEz(self):
        return self.EKE.isel(xT=slice(1,-1), yT=slice(1,-1)).mean(['xT', 'yT'])
    
    @property
    def MOC(self):
        Vmerid = self.meridional_flux(self.u, self.v)
        Psi = 0 * Vmerid

        nz = len(self.T.deptht)
        e3t = self.mask.e3t_1d.squeeze().data

        for k in range(nz-2, -1, -1):
            Psi[:,k] = Psi[:,k+1] + e3t[k] * Vmerid[:,k]
        
        return Psi / 1.e+6 # to Sverdrups 
    
    @property
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
    
    @property
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

        llat = np.linspace(np.min(lat),np.max(lat), Nlat)

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

        print(field.shape)
        
        fun = field.plot.contourf if 'levels' in kw else field.plot

        fun(x=lon, y=lat, 
            aspect=1, size=4,
            cmap=cmocean.cm.balance, **kw)
        plt.gca().set_aspect(aspect=1)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

    def plot_EKE(self, **kw):
        self.EKE.isel(deptht=0,xT=slice(1,-1), yT=slice(1,-1)).plot.contourf(x='lonT', y='latT', 
            levels=15, vmin=0, cmap=cmocean.cm.balance, 
            vmax=0.4,  aspect=1, size=4, 
            cbar_kwargs = {'label': 'EKE, $m^2/s^2$'}, **kw)
        plt.gca().set_aspect(aspect=1)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Surface EKE')
    
    def plot_EKEmean(self, **kw):
        self.plot_2D(self.mean_z(self.EKE), levels=15, cbar_kwargs = {'label': 'EKE, $m^2/s^2$'}, **kw)
        plt.title('Depth-averaged EKE')

    def plot_EKEz(self, **kw):
        plt.semilogy(self.EKEz, self.EKEz.deptht, **kw)
        plt.xlabel('EKE, $m^2/s^2$')
        plt.ylabel('depth, $m$')
        plt.grid()
        plt.xlim(0 , 0.03)
        plt.ylim(4000, 5)
        plt.title('Lateral mean EKE')
    
    def plot_SST(self, **kw):
        T = self.T.isel(deptht=0).isel(xT=slice(1,-1), yT=slice(1,-1))
        T.plot.contourf(x='lonT', y='latT', 
            levels=np.arange(8,30,2), vmin=11, vmax=24, 
            cmap=cmocean.cm.balance, aspect=1, size=4, 
            cbar_kwargs = {'label': 'SST, $^oC$'}, **kw)
        Cplot = T.plot.contour(x='lonT', y='latT',
            levels=np.arange(8,30,2), colors='k', linewidths=0.5)
        plt.gca().set_aspect(aspect=1)
        plt.gca().clabel(Cplot, Cplot.levels)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')    
        plt.title('SST')

    def plot_MOC_heat(self):
        self.heat_flux2d.plot.contourf(
                x='lat', y='depth', levels=np.linspace(-2.5e+11,2.5e+11,11),
                vmin=-2.5e+11, vmax=2.5e+11, cmap=cmocean.cm.balance)
        Cplot = self.MOC.plot.contour(x='lat', y='depth', levels=np.linspace(-5,5,21), colors='k', linewidths=0.5)
        plt.gca().clabel(Cplot, 
            [-3.0,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,2.0],
            fmt={-3.0: '-3', -2.0: '-2', -1.5: '-1.5', -1.0: '-1', -0.5: '-0.5', 0.0: '0', 0.5: '0.5', 1.0: '1', 2.0: '2'})
        plt.ylim(4000,5)
        plt.yscale('log')
        plt.xlabel('Latitude')
        plt.ylabel('Depth, m')
    

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
    return x_out

def y(field):
    for y_out in ['yT', 'yU', 'yV']:
        if y_out in field.dims:
            break
    return y_out

def depth(field):
    for depth_out in ['deptht', 'depthu', 'depthv', 'depth']:
        if depth_out in field.coords:
            break
    return depth_out

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

# def remesh(field, target):
#     x_out, y_out, depth_out = x(target), y(target), depth(target)
#     x_in, y_in, depth_in = x(field), y(field), depth(field)
#     print(x_out, y_out, depth_out)
#     return field.interp({x_in: target[x_out], y_in: target[y_out], depth_in: target[depth_out]}, method='linear')
