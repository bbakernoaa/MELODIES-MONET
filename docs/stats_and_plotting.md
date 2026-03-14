# Statistics and Plotting Capabilities in MELODIES-MONET

MELODIES-MONET provides a dynamic integration with the `monet-stats` and `monet-plots` libraries, optimized for high-performance and lazy execution via Xarray and Dask.

## Statistics Configuration

Statistics are defined in the `stats` section of the control file. You can specify any statistical metric available in `monet-stats` by its abbreviation.

### YAML Structure
```yaml
stats:
  my_stats_group:
    type: table           # Generate a summary table
    data: ['eval_label']  # List of evaluation labels to include
    stat_list: ['MB', 'RMSE', 'R2', 'KGE', 'HSS'] # Any monet-stats abbreviation
    threshold: 70.0       # Optional: Threshold for contingency scores (e.g., HSS, POD)
    create_table: True    # Optional: Whether to generate a plot of the stats table
    out_table_kwargs:     # Optional: Customization for the stats table plot
      fontsize: 12
      figsize: [8, 5]
```

### Dynamic Lookup and Lazy Evaluation
Statistics are looked up dynamically in `monet-stats`. The system preserves laziness by passing Xarray objects directly to the underlying metrics. Laziness is only broken when scalar values are required for CSV output or table rendering.

Common statistics include:
- **Error Metrics**: `MB`, `MAE`, `RMSE`, `NMB`, `NME`
- **Correlation/Efficiency**: `R`, `R2`, `IOA`, `KGE`, `NSE`, `CCC`
- **Contingency Scores**: `HSS`, `ETS`, `POD`, `FAR`, `CSI`, `TSS`

## Plotting Configuration

Plotting is defined in the `plotting` section. MELODIES-MONET maps requests to the appropriate plot class in `monet-plots` while maintaining a lazy-first workflow.

### YAML Structure
```yaml
plotting:
  my_plot_group:
    type: timeseries      # e.g., 'timeseries', 'scatter', 'taylor', 'kde', 'spatial'
    data: ['eval_label']
    title: "My custom title"
    ylabel: "Concentration (ppb)"
    plot_kwargs:          # Group-level plot customizations
      linestyle: "--"
      marker: "o"
    figsize: [10, 6]      # Subplot parameters passed to plt.subplots
```

### Supported Plot Types
- `timeseries`: Standard time series plots (native Xarray support).
- `scatter`: Scatter plots with regression lines.
- `taylor`: Taylor diagrams for skill evaluation.
- `kde`: Kernel Density Estimation plots.
- `spatial`: Spatial maps of concentrations or bias.

### Keyword Argument (Kwargs) Propagation
MELODIES-MONET handles propagation of keyword arguments using a hierarchical approach:
1.  **Data-Specific `plot_kwargs`**: Defined in the `data` section for each model/obs. These are used for specific elements (e.g., line `color`, `label`).
2.  **Group-Specific `plot_kwargs`**: Defined in the `plotting` section. These apply globally to all items in the plot group.
3.  **Top-Level Plot Args**: Keys like `title`, `ylabel`, and `xlabel` are passed to plot initialization.
4.  **Subplot Args**: Keys like `figsize` and `gridspec_kw` are passed to `plt.subplots`.

### Example: Combining Kwargs
```yaml
data:
  my_model:
    type: model
    ...
    plot_kwargs: {'color': 'blue', 'label': 'Model A'}

plotting:
  combined_ts:
    type: timeseries
    data: ['eval_my_model']
    plot_kwargs: {'linewidth': 2.0} # linewidth 2.0 will be used for Model A
```

## Internal Architecture

The orchestrator uses a bridge module (`melodies_monet.util.bridge`) to manage the interface between scientific data and visualization/analysis libraries. Key features:
- **Strict Lazy Preservation**: Avoids `to_dataframe()` or `.values` calls on large datasets.
- **Dynamic Adaption**: Automatically identifies time coordinates and handles data grouping/averaging before plotting.
- **Unified Interface**: Statistics and plots from `monet-stats` and `monet-plots` are seamlessly integrated, including support for custom plugins if registered in the libraries.
