import os
#os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import pyfesom2 as pf
import xarray as xr
import numpy as np
import shutil
import sys
import random
import string

from scipy.io import loadmat
from scipy.interpolate import griddata

sys.path.append("/p/home/jusers/mueller29/juwels/jupyter_notebooks/implicit_filtering/")
from implicit_filter import CuPyFilter
import math
import dask
from dask import delayed
import dask.array as da
from dask.distributed import Client
from dask.distributed import Lock


##### define all the functions

def generate_random_string(length):
    # Define the characters to choose from
    characters = string.ascii_letters + string.digits  # You can add more characters if needed
    # Use random.choices to generate a list of random characters
    random_characters = random.choices(characters, k=length)
    # Join the characters into a string
    random_string = ''.join(random_characters)
    return random_string

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Distance in kilometers
    distance = R * c
    
    return distance

def get_dxdy(lon1, lon2, lat1, lat2):
    # Calculate distance and angle
    dist = haversine(lon1, lat1, lon2, lat2)
    ang = np.arctan2(lon2 - lon1, lat2 - lat1)
    
    # Calculate dx and dy
    dx = dist * np.cos(ang)
    dy = dist * np.sin(ang)
    
    return dx, dy

def get_points_within_radius(lon_in, lat_in, lon_target, lat_target, radius, fac):
    # Compute distances from all points to the target point
    distances = haversine(lon_in, lat_in, lon_target, lat_target)
    max_radius = radius*fac
    # Filter points within the maximum radius
    within_radius_indices = np.where(distances <= max_radius)[0]
    within_radius_lon = lon_in[within_radius_indices]
    within_radius_lat = lat_in[within_radius_indices]
    
    dx =  within_radius_lon - lon_target
    dy =  within_radius_lat - lat_target

    # Calculate differences between target point and filtered points
    dx_km, dy_km = np.vectorize(get_dxdy)(lon_target, within_radius_lon, lat_target, within_radius_lat)
    dx_rel = dx_km/radius
    dy_rel = dy_km/radius
    return within_radius_indices, dx, dy, dx_km, dy_km, dx_rel, dy_rel

def process_eddy(lon_target, lat_target, radius_target, lon_in, lat_in, data_in, fac, X, Y):
    within_radius_indices, _, _, _, _, dx_rel, dy_rel = get_points_within_radius(lon_in, lat_in, lon_target, lat_target, radius_target, fac)
    data_tmp = data_in[within_radius_indices]
    gridded_data = griddata((dx_rel.ravel(), dy_rel.ravel()), data_tmp.ravel(), (X, Y), method='linear', fill_value=np.nan)

    del within_radius_indices
    del data_in
    del data_tmp
    del dx_rel
    del dy_rel
    del X
    del Y
    return gridded_data

def create_empty_netcdf(outfile,date,var,ne,nx,ny):
    if (os.path.exists(outfile)):
        os.remove(outfile)    
    date_string = '-'.join([str(date.dt.year.values), '01', f'01'])
    
    tref = np.datetime64(date_string,'ns')    
    tnow = date.values
    time_values_seconds = (tnow-tref) / np.timedelta64(1, 's')
    
    # Create the dataset
    ds = xr.Dataset()
    # Assuming 'x' and 'y' are the dimensions of the slices
    ds[var] = xr.DataArray(np.empty((1,ne, nx, ny)), dims=['time','ee', 'x', 'y']) #the gridded field around eddy
    ds['cyc'] = xr.DataArray(np.empty((1,ne)), dims=['time','ee']) #sense of rotation
    ds['lon'] = xr.DataArray(np.empty((1,ne)), dims=['time','ee']) #center lon
    ds['lat'] = xr.DataArray(np.empty((1,ne)), dims=['time','ee']) #center lat
    ds['rad'] = xr.DataArray(np.empty((1,ne)), dims=['time','ee']) #radius
    ds['time'] = xr.DataArray([time_values_seconds], dims='time', attrs={'units': 'seconds since 2015-01-01 0:0:0'}) #the same for every eddy on that day, so can be set here
    ds['num'] =  xr.DataArray(np.empty((1,ne)), dims=['time','ee']) #running number per day -> 'ee'
    # Write the dataset to a NetCDF file with specified permissions
    ds.to_netcdf(outfile, mode='w', encoding={var: {'dtype': 'float32'}})        
        
    
@delayed
def process_date(lon_target, lat_target, type_target, radius_target, lon_in, lat_in, data_in, fac, X, Y, ind,var,outfile,lock):
    gridded_data = process_eddy(lon_target, lat_target, radius_target, lon_in, lat_in, data_in, fac, X, Y)
    with lock:    
        with xr.open_dataset(outfile, mode='a') as ds:
            ds[var][0, ind, : , :] = gridded_data
            ds['cyc'][0, ind] = type_target
            ds['lon'][0, ind] = lon_target
            ds['lat'][0, ind] = lat_target
            ds['rad'][0, ind] = radius_target
            ds['num'][0,ind] = ind
        ds.to_netcdf(outfile, mode='a', encoding={var: {'dtype': 'float32'}})

def main():
    #inputs
    year=int(sys.argv[1]) #which year from input
    var=str(sys.argv[2]) #which variable from input
    #depth=int(sys.argv[3]) #which year from input
    depth=100 #which year from input

    #spawn a parallel cluster
    n_cores = 12
    mem_lim = str(int(100*np.floor(960/n_cores)))+'MB' #128GB total memory when running 4 jobs on booster node, set to MB, divide by number of cores and round to next 100
    dask_dir = '/p/scratch/chhb19/mueller29/dask_dir/'+generate_random_string(10)
    if os.path.exists(dask_dir):
        shutil.rmtree(dask_dir)
    if 'client' in locals() or 'client' in globals():
        client.close()
    client = Client(local_directory=dask_dir,n_workers=n_cores, threads_per_worker=1,memory_limit=mem_lim)
    client.amm.start()
    
    time_chunk = 1; #chunk data saily -> each day is treated separately anyway
    nod2_chunk = 11538465; #make sure no chunking in space!
    fac=5 # factor of 5*eddy_radius around each eddy center point

    # paths
    data_path = '/p/scratch/chhb19/mueller29/AO_40_nopc/'
    mesh_path = '/p/project/chhb19/meshes/AO_40/'
    
    # load fesom mesh stuff
    mesh=pf.load_mesh(mesh_path)
    model_lons=mesh.x2
    model_lats=mesh.y2
    
    # rotate to equator (eddy detection stuff is saved in rotated coordinates)
    lons_rot,lats_rot = pf.ut.scalar_g2r(-90,90,90,model_lons,model_lats)

    # load eddy shapes from Nencioli detection
    global all_days, all_lats, all_lons, all_rads, all_types, all_speeds  # Declare variables as global

    mat_path = '/p/project/chhb19/mueller29/Nencioli_tracking/' + str(depth) + 'm/python_readable/'
    vars = ['lons', 'lats', 'rads', 'speeds', 'types']

    all_days = None  # Initialize all_days
    all_lats = None  # Initialize all_lats
    all_lons = None  # Initialize all_lons
    all_rads = None  # Initialize all_rads
    all_types = None  # Initialize all_types
    all_speeds = None  # Initialize all_speeds

    for mat_var in vars:
        try:
            mat = loadmat(mat_path + 'eddy_' + mat_var + '.mat')
            all_data = mat.get(mat_var)
            if all_data is not None:
                globals()['all_' + mat_var] = all_data.flatten()
                print(f"Loaded {mat_var} successfully.")
            else:
                print(f"Warning: Variable {mat_var} is not found in the loaded .mat file.")
        except Exception as e:
            print(f"Error loading {mat_var}:", e)

    # Load 'eddy_days.mat'
    try:
        mat = loadmat(mat_path + 'eddy_days.mat')
        all_days = mat['days']
        print("Loaded days successfully.")
    except Exception as e:
        print("Error loading 'eddy_days.mat':", e)

    # Now keep only eddies from the year to analyze (eddy data contains all years)
    if all_lats is not None:  # Check if all_lats has been assigned a value
        ind_year = all_days[:, 0] == year
        all_days = all_days[ind_year]
        all_lats = all_lats[ind_year]
        all_lons = all_lons[ind_year]
        all_rads = all_rads[ind_year]
        all_types = all_types[ind_year]
        all_speeds = all_speeds[ind_year]
    else:
        print("Error: all_lats has not been initialized.")
    
    #load filter
    jf = CuPyFilter.load_from_file('/p/home/jusers/mueller29/juwels/jupyter_notebooks/implicit_filtering/filter_cash_AO_1km.npz')
    dxm = 1 # Approximate resolution of the mesh in km
    Kc = np.array([100]) * dxm #!!!this might have to be adjusted!!! right now 100km filter (high-pass)
    Kc = 2 * math.pi / Kc  # Wavenumbers
    cyclic_length = 360  # in degrees; if not cyclic, take it larger than  zonal size
    cyclic_length = cyclic_length * math.pi / 180 

    # load FESOM data
    data_all=xr.open_mfdataset((data_path+var+'.fesom.'+str(year)+'.nc'),chunks={'time':time_chunk,'nod2':nod2_chunk})[var]

    
    #define regular mesh in multiples of R
    reg_x = np.arange(-3.1,3.1,.1)
    reg_y = np.arange(-3,3.1,.1)
    X,Y = np.meshgrid(reg_x,reg_y)
    
    nx = X.shape[0] #size of gridded data in x direction
    ny = X.shape[1] #size of gridded data in y direction
    ndays = len(all_days) #number of days 
    out_path='/p/scratch/chhb19/mueller29/composites/100m/'+var+'/'
    os.makedirs(out_path, exist_ok=True)
    
    for dd in range(330,ndays): #####remember to reset!! this is just to rerun end of November and December
        lons=all_lons[dd].flatten()
        lats=all_lats[dd].flatten()
        rads=all_rads[dd].flatten()
        types=all_types[dd].flatten()
        speeds=all_speeds[dd].flatten()
        date=all_days[dd]

        ok_size = rads > 1.5
        ok_speed = speeds > 0

        lons=lons[ok_size&ok_speed]
        lats=lats[ok_size&ok_speed]
        rads=rads[ok_size&ok_speed]
        types=types[ok_size&ok_speed]
        speeds=speeds[ok_size&ok_speed]   
        
        ne = len(lons)
        
        # # Zero-pad the month and day components
        date_string = ''.join([str(date[0]), f'{date[1]:02}', f'{date[2]:02}'])
        # # Convert to datetime object
        # datetime_object = np.datetime64(date_string,'ns')
        # date_string = np.datetime_as_string(datetime_object,unit='D').replace('-', '')

        #load and high-pass filter data
        data = data_all[dd,:].values
        data_coarse = jf.compute(1, Kc[-1], data)
        data_in = data-data_coarse
        date_xr = data_all.time[dd]
        
        filename = 'composite_'+var+'_'+date_string+'.nc'

        #Create the empty Dataset
        outfile = (out_path+filename)    
        create_empty_netcdf(outfile, date_xr, var, ne, nx, ny)

        #persist the input data
        data_in_dask_array = da.from_array(data_in, chunks='auto')
        client.persist(data_in_dask_array)

        lon_in_dask_array = da.from_array(lons_rot, chunks='auto')
        client.persist(lon_in_dask_array)

        lat_in_dask_array = da.from_array(lats_rot, chunks='auto')
        client.persist(lat_in_dask_array)

        # datetime_dask_array = da.from_array(datetime_object)
        # client.persist(datetime_dask_array)

        #make composites first lazy, then compute in parallel and save
        results = []
        lock = Lock()
        
        for ii in range(ne):
            lon_eddy = lons[ii]
            lat_eddy = lats[ii]
            type_eddy = types[ii]
            rad_eddy = rads[ii]
            results.append(process_date(lon_eddy, lat_eddy, type_eddy, rad_eddy, lon_in_dask_array, lat_in_dask_array, data_in_dask_array, fac, X, Y, ii,var,outfile,lock))  
                    
        result=dask.compute(results)

    client.close()
        
if __name__ == '__main__':
    main()
    
    
    
