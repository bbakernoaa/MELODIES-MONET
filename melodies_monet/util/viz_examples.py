import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_wind_components(u: xr.DataArray, v: xr.DataArray, interactive: bool = False):
    """
    Demonstrate the Aero Protocol Two-Track visualization rule.

    Parameters
    ----------
    u : xr.DataArray
        U component of wind.
    v : xr.DataArray
        V component of wind.
    interactive : bool, optional
        Whether to produce an interactive hvPlot instead of static Matplotlib, by default False.
    """
    if interactive:
        # Track B: Interactive (hvPlot / GeoViews)
        # rasterize=True is mandatory for large grids
        plot = u.hvplot.quadmesh(
            x="longitude", y="latitude", geo=True, rasterize=True, cmap="RdBu_r", title="U Wind Component (Interactive)"
        )
        return plot
    else:
        # Track A: Publication (Matplotlib + Cartopy)
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})

        # transform= is mandatory in plot calls
        u.plot(ax=ax, x="longitude", y="latitude", transform=ccrs.PlateCarree(), cmap="RdBu_r", cbar_kwargs={"label": "m/s"})

        ax.coastlines()
        ax.set_title("U Wind Component (Static)")
        plt.show()


if __name__ == "__main__":
    # Mock data for demonstration
    lon = np.linspace(-180, 180, 360)
    lat = np.linspace(-90, 90, 180)
    u_data = np.random.randn(180, 360)

    u = xr.DataArray(u_data, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"], name="u")
    v = u.copy(name="v")  # Simplified for demo

    print("Generating static plot (Track A)...")
    # plot_wind_components(u, v, interactive=False)
    # Commented out actual plt.show() for non-interactive environment execution

    print("Aero Protocol Visualization Examples Ready.")
