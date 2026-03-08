
import numpy as np
import pandas as pd
import xarray as xr
import os
from melodies_monet import driver

# Setup paths
control_file = "docs/examples/control_idealized.yaml"
output_dir = "docs/examples/output/idealized"
os.makedirs(output_dir, exist_ok=True)

# 1. Read control
an = driver.analysis()
an.control = control_file
an.read_control()

# 2. Generate dummy model data
control = an.control_dict
nlat, nlon = 10, 20
lon = np.linspace(-161, -60, nlon)
lat = np.linspace(18, 60, nlat)
time = pd.date_range(control['analysis']['start_time'], control['analysis']['end_time'], freq="3h")
ntime = time.size

lat_da = xr.DataArray(lat, dims="lat", name="lat")
lon_da = xr.DataArray(lon, dims="lon", name="lon")
time_da = xr.DataArray(time, dims="time", name="time")

ds_dict = {}
for field_name in control['model']['test_model']['variables'].keys():
    data = np.random.rand(ntime, nlat, nlon)
    da = xr.DataArray(data, coords=[time_da, lat_da, lon_da], dims=['time', 'lat', 'lon'])
    ds_dict[field_name] = da
ds = xr.Dataset(ds_dict).expand_dims("z", axis=1)
ds["z"] = [1]
# Add crs for monet
ds.attrs['projection'] = '{"proj": "latlong", "ellps": "WGS84", "datum": "WGS84", "no_defs": true}'

model_file = os.path.join("docs/examples", control['model']['test_model']['files'])
ds.to_netcdf(model_file)

# 3. Generate dummy obs data
n = 50
lats = np.random.uniform(lat[0], lat[-1], n)
lons = np.random.uniform(lon[0], lon[-1], n)
ds_dict = {}
for field_name0 in control['model']['test_model']['variables'].keys():
    field_name = control['model']['test_model']['mapping']['test_obs'][field_name0]
    values = np.random.rand(ntime, n)[:, np.newaxis]
    da = xr.DataArray(
        values,
        coords={
            "x": ("x", np.arange(n)),
            "time": ("time", time),
            "latitude": (("y", "x"), lats[np.newaxis, :]),
            "longitude": (("y", "x"), lons[np.newaxis, :]),
            "siteid": (("y", "x"), np.array([f"{x:03}" for x in range(n)])[np.newaxis, :]),
        },
        dims=("time", "y", "x")
    )
    ds_dict[field_name] = da
obs_file = os.path.join("docs/examples", control['obs']['test_obs']['filename'])
xr.Dataset(ds_dict).to_netcdf(obs_file)

# 4. Run MM analysis
os.chdir("docs/examples")
# Monkey patch open_obs to avoid monetio readers for this test
def open_obs_mock(self, time_interval=None, control_dict=None):
    import xarray as xr
    self.obj = xr.open_dataset(self.file)
driver.observation.open_obs = open_obs_mock

an.control = "control_idealized.yaml"
an.read_control()
an.open_models()
an.open_obs()
print("Models opened:", list(an.models.keys()))
print("Obs opened:", list(an.obs.keys()))
an.pair_data()
print("Paired keys before save/read:", list(an.paired.keys()))

if 'test_obs_test_model' in an.paired:
    print("SUCCESS: Found test_obs_test_model before save/read")
else:
    print("FAILURE: test_obs_test_model NOT FOUND before save/read")
    exit(1)

# 5. Test save/read
an.save_analysis()
an.paired = {} # Clear it
an.read_analysis()
print("Paired keys after save/read:", list(an.paired.keys()))

if 'test_obs_test_model' in an.paired:
    print("SUCCESS: Found test_obs_test_model after save/read")
else:
    print("FAILURE: test_obs_test_model NOT FOUND after save/read")
    exit(1)
