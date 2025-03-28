import pandas as pd
import numpy as np
import math
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
import plotly.express as px
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CheckboxGroup, Button, CustomJS
from bokeh.models.widgets import RadioButtonGroup, RadioGroup
from bokeh.layouts import column, row
from bokeh.io import curdoc
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from bokeh.io import output_file
from bokeh.models import HoverTool

# Get filenames from input args
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results', required=False, help='Provide runtimes.csv filename')
parser.add_argument('-e', '--emp', required=False, help='Provide empirical roofline filename')
parser.add_argument('-g', '--gpu', required=True, help='Name of GPU')
parser.add_argument('-d', '--datatypes', required=True, help='Data types you want to plot (ex. F32, F64). Provide in space separated format')
parser.add_argument('-hw', '--hardware', required=False, help='Provide directory to hardware metrics you want to plot (ex. Power, Frequency).')
parser.add_argument('-m', '--memory', required=True, help='Memory types you want to plot (ex. L2, HBM). Provide in space separated format')
parser.add_argument('-o', '--operations', required=False, help='Operations you want to plot (ex. ADD, MUL). Provide in space separated format')
parser.add_argument('-f', '--filename', required=True, help='Filename which you want the plot to be')

args = parser.parse_args()

print('Parsing arguments...')

if args.results:
    df = pd.read_csv(args.results)
    df['PERF'] = df['PERF'] / 1000
else:
    print('No runtimes file provided.')

if args.emp:
    roof_df = pd.read_csv(args.emp)
gpu_name = args.gpu
data_types = args.datatypes.split()
op_types = args.operations.split()
mem_types = args.memory.split()
hw_metrics_dir = args.hardware
n_threads=7471104
filename = args.filename
print('Parsing complete.')

# Reorganize memory types
order=['LDS', 'L1', 'L2', 'HBM']
mem_types = [mem for mem in order if mem in mem_types]

if args.hardware:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))
else:
    fig, ax1 = plt.subplots(1, 1, figsize=(24, 12))

# Extract roofline data into dictionary for easier access
print('Gathering empirical roofline data...')
mem_ops = {}
if args.emp:
    # for data in data_types:
    #     for mem in mem_types:
    slope = roof_df['HBMBw'].mean() / 1000
    peak = roof_df['FP64Flops'].mean() / 1000
    if 'pubbench' in args.results:
        if df['PERF'].max() > peak:
            peak = df['PERF'].max() / 1000
    mem_ops['HBM' + '_' + 'VALU' + '_' + 'FP64'] = (slope, peak)
    print('Data gathered.')

gpus = df['GPU'].unique()
emp_roofs = {}
for gpu in gpus:
    slope = (df[(df['GPU'] == gpu) & (df['AI_HBM_MULADD_FP32'] != 0)]['PERF'] / df[(df['GPU'] == gpu) & (df['AI_HBM_MULADD_FP32'] != 0)]['AI_HBM_MULADD_FP32']).max()
    peak = df[df['GPU'] == gpu]['PERF'].max()
    emp_roofs[gpu] = (slope, peak)

# elif args.create:
#     print('Creating new rooflines...')
kernels = {}
if args.results:
    ai_cols = [c for c in df.columns if 'AI_' in c]
    for data in data_types:
        ai_data_cols = [c for c in ai_cols if data == c.split('_')[-1]]
        for op in op_types:
            ai_data_op_cols = [c for c in ai_data_cols if op == c.split('_')[-2]]
            for mem in mem_types:
                ai_data_op_mem_col = [c for c in ai_data_op_cols if mem == c.split('_')[-3]][0]
                # print(op, data, ai_data_op_mem_cols)
                # print(ai_data_op_mem_cols)
                df_subset = df[df[ai_data_op_mem_col] != 0]

                perf_values = df_subset['PERF']
                ai_values = df_subset[ai_data_op_mem_col]
                gpus = df_subset['GPU']

                # slope = (perf_values / ai_values).max() / 1000

                # max_index = (perf_values / ai_values).idxmax()
                # max_perf = perf_values.loc[max_index]
                # max_ai = ai_values.loc[max_index]
                # print(f'Max PERF value: {max_perf}')
                # print(f'Max AI value: {max_ai}')
                # peak = roof_df[roof_df[ai_data_op_mem_col] != 0]['PERF'].max() / 1000
                # print(slope, peak)
                kernels[mem + '_' + op + '_' + data] = (gpus, ai_values, perf_values)
    # print('New rooflines created.')

print('Plotting rooflines...')
# plt.figure(figsize=(12, 6))
# Generate AI values for x-axis
AI = np.logspace(-1, 6, 10000)

# Define colors and markers
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
markers = ['circle', 'square', 'triangle', 'diamond', 'inverted_triangle', 'hex', 'cross', 'asterisk']
gpus = df['GPU'].unique()
data_type_colors = {data: color for data, color in zip(data_types, colors)}
gpu_type_colors = {data: color for data, color in zip(gpus, colors)}
op_type_markers = {op: marker for op, marker in zip(op_types, markers)}

# Create Bokeh figure
tooltips = [("AI", "@x"), ("Performance", "@y"), ("Operation", "@op"), ("Data Type", "@data"), ("Memory", "@mem")]
p = figure(x_axis_type='log', y_axis_type='log', title='Empirical Rooflines',
           x_axis_label='Arithmetic Intensity (FLOPs/Byte)',
           y_axis_label='Performance (TFLOPs/sec)', tools='wheel_zoom,box_zoom,reset,save', width=900, height=600)

# Dictionaries to hold references to the plotted lines
op_sources = {}
data_sources = {}
gpu_sources = {}
# marker_plots = []
for key, (gpus, ais, perfs) in kernels.items():
    mem, op, data = key.split('_')
    # color = gpu_type_colors[data]
    marker = op_type_markers[op]

    for gpu, ai, perf in zip(gpus, ais, perfs):
        color = gpu_type_colors[gpu]
        source_marker = ColumnDataSource(data=dict(x=[ai], y=[perf], op=[op], data=[data], mem=[mem], gpu=[gpu]))

        marker_plot = p.scatter('x', 'y', source=source_marker, size=6, color=color, marker=marker, visible=False)
        # marker_plots.append(marker_plot)
        # Add hover tool

        # Store references for filtering
        if op not in op_sources:
            op_sources[op] = []
        if data not in data_sources:
            data_sources[data] = []
        if gpu not in gpu_sources:
            gpu_sources[gpu] = []

        op_sources[op].append(marker_plot)
        data_sources[data].append(marker_plot)
        gpu_sources[gpu].append(marker_plot)

# hover = HoverTool(renderers=[marker_plots], tooltips=[
#             ("AI", "@x"),
#             ("Performance", "@y"),
#             ("Operation", "@op"),
#             ("Data Type", "@data"),
#             ("Memory", "@mem")
#         ])

for gpu, (slope, peak) in emp_roofs.items():
    # mem = key.split('_')[0]
    # op = key.split('_')[1]
    # data = key.split('_')[2]
    x_intersect = peak / slope
    x_slope = AI[AI <= x_intersect] # Generate x values for sloped line
    x_slope = np.array([float(x_slope[0]), float(x_slope[-1])])
    x_horizontal = AI[AI >= x_intersect] # Generate x values for horizontal line
    x_horizontal = [x_horizontal.min(), x_horizontal.max()]
    y_slope = slope * x_slope # Generate y values for sloped line (bandwidth * AI)
    y_slope = [y_slope.min(), y_slope.max()]
    y_horizontal = np.full_like(x_horizontal, peak) # Generate y values for horizontal line (peak performance)
    y_horizontal = [y_horizontal.min(), y_horizontal.max()]
    # print(f'Plotting {mem} {op} {data}...')
    # print(f'Slope (Bw): {np.round(slope, 2)}')

    # Create data sources for the lines
    source_slope = ColumnDataSource(data=dict(x=x_slope, y=y_slope, gpu=[gpu]*2))
    source_horizontal = ColumnDataSource(data=dict(x=x_horizontal, y=y_horizontal, gpu=[gpu]*2))

    # Plot the lines without labels
    slope = p.line('x', 'y', source=source_slope, line_width=2, color=gpu_type_colors[gpu], visible=False)
    peak = p.line('x', 'y', source=source_horizontal, line_width=2, color=gpu_type_colors[gpu], line_dash='dashed', visible=False)
    slope.name = 'roofline'
    peak.name = 'roofline'
    gpu_sources[gpu].append(slope)
    gpu_sources[gpu].append(peak)

# Create CheckboxGroups for Operation Types and Data Types with no active selections
op_checkboxes = CheckboxGroup(labels=[f"{op} ({marker})" for op, marker in zip(op_types, markers)], active=[])
data_checkboxes = RadioGroup(
    labels=[f"{data}" for data in data_types],
    active=None
)
gpus = df['GPU'].unique()
# min_length = min(len(gpus), len(colors))
gpu_checkboxes = CheckboxGroup(labels=[f"{gpu} ({color})" for gpu, color in zip(gpus, colors)], active=[])

# Define the JavaScript Callback for Checkbox Interactions
callback_code = """
    // Function to update visibility based on active checkboxes
    function update_visibility() {
        const selected_ops = new Set(op_checkboxes.active.map(i => op_checkboxes.labels[i].split(' ')[0]));
        const selected_data = data_checkboxes.active !== null ? data_checkboxes.labels[data_checkboxes.active] : null;
        const selected_gpus = new Set(gpu_checkboxes.active.map(i => gpu_checkboxes.labels[i].split(' ')[0]));

        for (const [op, renderers] of Object.entries(op_sources)) {
            const op_visible = selected_ops.has(op);
            for (const renderer of renderers) {
                const data = renderer.data_source.data.data[0];
                const data_visible = selected_data === null || selected_data === data;
                const gpu = renderer.data_source.data.gpu[0];
                const gpu_visible = selected_gpus.size === 0 || selected_gpus.has(gpu);
                renderer.visible = op_visible && data_visible && gpu_visible;
            }
        }
        for (const [gpu, renderers] of Object.entries(gpu_sources)) {
            const gpu_visible = selected_gpus.has(gpu);
            for (const renderer of renderers) {
                if (renderer.name === 'roofline') {
                    renderer.visible = gpu_visible;
                }
            }
        }
    }

    // Update the visibility of the plots
    update_visibility();
"""

# Create the CustomJS callback
callback = CustomJS(args=dict(op_checkboxes=op_checkboxes, data_checkboxes=data_checkboxes, gpu_checkboxes=gpu_checkboxes,
                              op_sources=op_sources, data_sources=data_sources, gpu_sources=gpu_sources), code=callback_code)

# Attach the callback to the 'active' property change
op_checkboxes.js_on_change('active', callback)
data_checkboxes.js_on_change('active', callback)
gpu_checkboxes.js_on_change('active', callback)

# Layout and show
widgets = column(op_checkboxes, data_checkboxes, gpu_checkboxes)

# Create a horizontal layout with the plot and widgets
layout = row(p, widgets)

# Add the layout to the current document
curdoc().add_root(layout)
curdoc().template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Roofline Plot</title>
    <style>
        .custom-checkbox .bk-input-group label {
            color: inherit;
        }
    </style>
</head>
<body>
    {% block contents %}
    {{ super() }}
    {% endblock %}
</body>
</html>
"""
output_file(filename=f'{filename}.html')
show(layout)
# print(f'roofline_plots/{gpu_name.lower()}_emp_rooflines_bokeh.html')

# fig.write_html(f'roofline_plots/{gpu_name.lower()}_emp_rooflines.html')

# Plot kernel runtime data if provided
# if args.results:
#     # subplot 1
#     new_df = pd.DataFrame
#     df['Kernel_Length'] = df['Kernel'] + '_' + df['Length'].astype(str)
#     unique_combinations = df['Kernel_Length'].unique()
#     unique_lengths = df["Length"].unique()
#     unique_iterations = df["Iterations"].unique()
#     combo_color_map = {combo: color for combo, color in zip(unique_combinations, plt.get_cmap("tab10").colors)}

#     def get_shade(color, iteration, max_iterations):
#         try:
#             iteration = math.log10(iteration)
#             max_iterations = math.log10(max_iterations)
#         except ValueError:
#             print(f"Error: iteration: {iteration}")
#         alpha = (iteration / max_iterations) * 0.8 + 0.2
#         return color[0], color[1], color[2], alpha
        

#     print('Plotting kernels...')
#     max_iterations = max(unique_iterations)

#     # Filter performance and AI measurements to only consist of the operations and memory types we want to plot
#     # perf_cols = [c for c in df.columns if 'PERF_' in c and any(op in c for op in data_types)]
#     ai_cols = [c for c in df.columns if 'AI_' in c and any(op in c for op in data_types) and any(mem in c for mem in mem_types)]

#     # For each kernel, plot its performance and AI measurements
#     handles = []
#     labels = []

#     # subplot 2
#     for _, row in df.iterrows():
#         length = row['Length']
#         iteration = row['Iterations']
#         color = combo_color_map[row['Kernel_Length']]
#         shade = get_shade(color, iteration, max_iterations)
#         for ai_col in ai_cols:
#             # for perf_col in perf_cols:
#             if float(row[ai_col]) > 0: # If performance is 0, that specific data was not gathered for that kernel
#                 scatter = ax1.scatter(row[ai_col], row['PERF'], color=shade, edgecolor="black")
#                 # label = f"{row['Kernel']}, 2^{int(math.log2(row['Length']))} Length, {row['Iterations']} Iters, {ai_col}, {perf_col}"
#                 # handles.append(scatter)
#                 # labels.append(label)
#     # custom_legend = plt.legend(handles, labels, loc='lower right', title='Kernel Data', prop={'size': 8})
#     # for every csv file in hw_metrics dir, get the last value in df[Power[W]] and save it in dictionary
#     if args.hardware:
#         print('Gathering power data from hw_metrics directory...')
#         hw_data = {}
#         for file in os.listdir(hw_metrics_dir):
#             if file.endswith('.csv'):
#                 # strip all the column names of whitespace
#                 # print(os.path.join(hw_metrics_dir, file))
#                 hw_df = pd.read_csv(os.path.join(hw_metrics_dir, file))
#                 hw_df.columns = hw_df.columns.str.strip()
#                 # print(hw_df['Power[W]'].iloc)
#                 if not hw_df.empty:
#                     # print(f'{file} is empty')
#                     if 'Power[W]' in hw_df.columns:
#                         # length = file.split('_')[-1]
#                         ai = file.split('_')[-1][:-4]
#                         # print(ai)
#                         hw_data[float(ai)] = float(hw_df['Power[W]'].iloc[-1])
#         # Create a new subplot for power data
#         # fig, ax1 = plt.subplots(figsize=(12, 6))

#         # Plot power data
#         for ai, power in hw_data.items():
#             # print(f'AI: {ai}, Power: {power}')
#             ax2.scatter(ai, power, edgecolor="black")

#         ax2.set_xscale('log')
#         ax2.set_yscale('log')
#         ax2.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
#         ax2.set_ylabel("Power (W)")
#         ax2.set_title(f"Power Consumption vs Arithmetic Intensity ({gpu_name})")
#         # ax2.legend()
#         ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
#         ax2.set_xlim(0.01, 100000)
#         # ax2.set_ylim(100, 100000)
#         ax2.xaxis.set_major_locator(mtick.LogLocator(base=10.0, subs=[], numticks=10))
#         ax2.xaxis.set_minor_locator(mtick.LogLocator(base=10.0, subs='auto', numticks=10))
#         ax2.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{float(x)}"))
#         ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{float(y)}"))
        

#         # Save power plot
#         # power_filename = f'roofline_plots/{gpu_name.lower()}_power_vs_ai'
#         # plt.savefig(power_filename, bbox_inches='tight')
#         # print(f'Power plot saved as {power_filename}.png')
#         # for ai, power in hw_data.items():
            
#         print('Power data gathered')

#     print('Plotting complete.')

# Set log-log scale and add ticks


# Save plot
# os.makedirs('roofline_plots', exist_ok=True)
# if 'pubbench' in args.results:
#     filename = f'roofline_plots/{gpu_name.lower()}_pubbench_rooflines'
# else:
#     filename = f'roofline_plots/{gpu_name.lower()}_emp_rooflines'
plt.savefig(filename, bbox_inches='tight')
print(f'Plot saved as {filename}.png')