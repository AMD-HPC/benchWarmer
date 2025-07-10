import pandas as pd
import numpy as np
import math
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker, Title, Label
from bokeh.models import ColumnDataSource, CheckboxGroup, Button, FixedTicker, LogScale, LinearScale, CustomJS, Range1d, DataRange1d, Spacer
from bokeh.models.widgets import RadioButtonGroup, RadioGroup
from bokeh.models.axes import LinearAxis
from bokeh.layouts import column, row
from bokeh.io import curdoc
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_hex, LinearSegmentedColormap, to_rgba
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
    df['Power'] = df['Power'].fillna(0)
    df['PERF'] = df['PERF'] / 1000
    # Consolidate rows for GPU 'MI250X_two_gcd'
    if 'MI250X_two_gcd' in df['GPU'].values:
        ai_cols = [col for col in df.columns if 'AI_' in col]
        df_grouped = df[df['GPU'] == 'MI250X_two_gcd'].groupby(['Iterations'] + ai_cols, as_index=False).agg(
            {col: 'first' if col != 'PERF' else 'sum' for col in df.columns if col not in ['Iterations'] + ai_cols}
        )
        df = pd.concat([df[df['GPU'] != 'MI250X_two_gcd'], df_grouped], ignore_index=True)
    df['GPU'] = df['GPU'].replace({'MI250X_two_gcd': 'MI250X (2 GCDs)'})
else:
    print('No runtimes file provided.')

peak_bw = {}
gpus = ['MI250X', 'MI250X (2 GCDs)', 'MI300A', 'MI300X', 'A100', 'H100']
for gpu in gpus:
    if gpu not in ['A100', 'H100', 'MI250X (2 GCDs)']:
        emp_df = pd.read_csv(f'./roofline_csvs/roofline_{gpu}.csv')
        emp_df['HBMBw'] = emp_df['HBMBw'].astype(float)
        peak_bw[gpu] = emp_df['HBMBw'].mean() / 1000
    else:
        peak_bw['H100'] = 3350 / 1000
        peak_bw['A100'] = 1592 / 1000
        peak_bw['MI250X (2 GCDs)'] = 2500 / 1000




# if args.emp:
#     roof_df = pd.read_csv(args.emp)
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
# print('Gathering empirical roofline data...')
# gpus = df['GPU'].unique()
# emp_roofs = {}
# # if args.emp:
# # for data in data_types:
# #     for mem in mem_types:
# slope = roof_df['HBMBw'].mean() / 1000
# peak = roof_df['MFMAF32Flops'].mean() / 1000
# # if 'pubbench' in args.results:
# #     if df['PERF'].max() > peak:
# #         peak = df['PERF'].max() / 1000
# for gpu in gpus:
#     emp_roofs[gpu] = (slope, peak)
# print('Data gathered.')

# for gpu in gpus:
#     slope = (df[(df['GPU'] == gpu) & (df['AI_HBM_MULADD_FP32'] != 0)]['PERF'] / df[(df['GPU'] == gpu) & (df['AI_HBM_MULADD_FP32'] != 0)]['AI_HBM_MULADD_FP32']).max()
#     peak = df[df['GPU'] == gpu]['PERF'].max()
#     emp_roofs[gpu] = (slope, peak)
#     emp_roofs[gpu]

# elif args.create:
#     print('Creating new rooflines...')
kernels = defaultdict(lambda: defaultdict(dict))
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
                for _, row_data in df_subset.iterrows():
                    gpu = row_data['GPU']
                    ai_value = row_data[ai_data_op_mem_col]
                    perf_value = row_data['PERF']
                    # if 'RSQ' in ai_data_op_mem_col:
                    #     perf_value *= 1.5
                    #     ai_value *= 0.75
                    power_value = row_data['Power']
                    if ai_value in kernels[mem + '_' + op + '_' + data][gpu]:
                        existing_perf, existing_power = kernels[mem + '_' + op + '_' + data][gpu][ai_value]
                        kernels[mem + '_' + op + '_' + data][gpu][ai_value] = (existing_perf + perf_value, existing_power)
                    else:
                        # if op == 'GEMM':
                            # print(ai_value, ai_data_op_mem_col)
                        kernels[mem + '_' + op + '_' + data][gpu][ai_value] = (perf_value, power_value)

emp_roofs = defaultdict(lambda: defaultdict(dict))
for gpu in gpus:
    gpu_df = df[df['GPU'] == gpu]
    for mem in mem_types:
        for op in op_types:
            for data in data_types:
                ai_df = gpu_df[gpu_df['AI_' + mem + '_' + op + '_' + data] != 0]
                print(op, data, gpu, ai_df['PERF'].max())
                emp_roofs[gpu][mem + '_' + op + '_' + data] = (peak_bw[gpu], ai_df['PERF'].max())

print('Plotting rooflines...')

AI = np.logspace(-2, 6, 10000)

# Define colors and markers
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
markers = ['circle', 'square', 'triangle', 'diamond', 'inverted_triangle', 'hex', 'cross', 'asterisk']
gpus = df['GPU'].unique()
data_type_colors = {data: color for data, color in zip(data_types, colors)}
gpu_type_colors = {data: color for data, color in zip(gpus, colors)}
op_type_markers = {op: marker for op, marker in zip(op_types, markers)}

# Create Bokeh figure
tooltips = [("AI", "@x"), ("Performance", "@y TFLOPS/sec"), ("Operation", "@op"), ("Data Type", "@data_type"), ("Power", "@power W")]
p = figure(x_axis_type='log', y_range=(0.05, 1e3), x_range=(0.05, 1e5), y_axis_type='log', title='Empirical Rooflines with Power',
           x_axis_label='Arithmetic Intensity (FLOPs/Byte)', toolbar_location="right",
           y_axis_label='Performance (TFLOPs/sec)', tools='wheel_zoom,box_zoom,reset,save', width=900, height=600)
p.title.text_font_size = '14pt'
p.xaxis.axis_label_text_font_size = '16pt'
p.yaxis.axis_label_text_font_size = '16pt'
p.xaxis.major_label_text_font_size = '14pt'
p.yaxis.major_label_text_font_size = '14pt'

p.toolbar_location = None

# Create a toolbar box manually (you can also use p.toolbar as the toolbar itself)
# toolbar = Toolbar(toolbar=p.toolbar, toolbar_location="right")
# Dictionaries to hold references to the plotted lines
roofline_sources = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
gpu_sources = {}

# min_power = 200
# max_power = df['Power'].max()
# norm = plt.Normalize(
#     min_power,
#     max_power,
# )
# color_map = LinearSegmentedColormap.from_list(
#     'green_to_red', plt.cm.get_cmap('hsv')(np.linspace(0.33, 0, 256))
# )

# # Convert to a hex palette
# rgba_colors = [to_rgba(color_map(i / 255), alpha=0.7) for i in range(256)]  # 0.3 = 30% opacity
# hex_colors_with_alpha = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}{int(a*255):02x}' 
#                          for r, g, b, a in rgba_colors]

# # Create Bokeh color mapper
# color_mapper = LinearColorMapper(palette=hex_colors_with_alpha, low=min_power, high=max_power)

# # Add the color bar
# color_bar = ColorBar(
#     color_mapper=color_mapper,
#     ticker=BasicTicker(),
#     label_standoff=12,
#     border_line_color=None,
#     location=(0, 0)
# )

# p.add_layout(color_bar, 'right')

# y_axis_style = {
#     "text_font": p.yaxis[0].axis_label_text_font,
#     "text_font_size": p.yaxis[0].axis_label_text_font_size,
#     "text_font_style": p.yaxis[0].axis_label_text_font_style,
#     "text_color": p.yaxis[0].axis_label_text_color,
# }

# power_label = Title(
#     text='Power (W)',
#     text_font=y_axis_style["text_font"],
#     text_font_size=y_axis_style["text_font_size"],
#     text_font_style=y_axis_style["text_font_style"],
#     text_color=y_axis_style["text_color"],
#     offset=45
# )
# p.add_layout(power_label, 'right')

log_width = 0.303

scatter_data = {
    "x": [],
    "y": [],
    "op": [],
    "data_type": [],
    "gpu": [],
    "power": [],
    "xs": [],
    "ys": [],
    "color": []
}

# roofline_data = {
#     "op": [],
#     "data_type": [],
#     "slope": [],
#     "peak": []
# }

for key, gpu_data in kernels.items():
    # print(max([values[-1].max() for values in kernels.values()]))
    mem, op, data_type = key.split('_')
    marker = op_type_markers[op]

    for gpu, ai_data in gpu_data.items():
        # color = gpu_type_colors[gpu]
        y_range = p.y_range
        # color_map = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
        roofline = False
        slope, peak = emp_roofs[gpu]['HBM_' + op + '_' + data_type]
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
        slope_line = p.line('x', 'y', source=source_slope, line_width=2, color='black', visible=False)
        peak_line = p.line('x', 'y', source=source_horizontal, line_width=2, color='black', line_dash='dashed', visible=False)
        slope_line.name = 'roofline'
        peak_line.name = 'roofline'

        # roofline_data = {}

        # roofline_data["slope"].append(slope)
        # roofline_data["peak"].append(peak)

        roofline_sources[gpu][op][data_type].append(slope_line)
        roofline_sources[gpu][op][data_type].append(peak_line)

        # roofline_sources[gpu][op][data_type] = {
        #     "figure": p,
        #     "renderers": [slope_line, peak_line]
        # }
        

        # for data_type, entry in roofline_sources[gpu][op].items():
        #     # print(entry)
        #     p = entry["figure"]
        #     for line in entry["renderers"]:
        #         hover = HoverTool(
        #             renderers=[line],
        #             tooltips=[
        #                 ("AI", "@x"),
        #                 ("Performance", "@y TFLOPS/sec"),
        #             ]
        #         )
        #         p.add_tools(hover)
        
        
        # roofline_data["peak"].append(peak)
        # roofline_data["op"].append(op)
        # roofline_data["data_type"].append(data_type)
        # roofline_data["gpu"].append(gpu)
        
        for ai, (perf, power) in ai_data.items():
            # Create source with all data including power
            x_left = ai / (10 ** (log_width / 2))
            x_right = ai * (10 ** (log_width / 2))
            bar_width = x_right - x_left
            # color = to_hex(color_map(norm(power)))
            # print(gpu)
            slope, peak = emp_roofs[gpu][mem + '_' + op + '_' + data_type]

            if ai < peak / slope:
                y_patch = [0.05, slope * x_left, slope * x_right, 0.05]
            else:
                y_patch = [0.05, peak, peak, 0.05]
            
            knee = peak / slope
            # print(knee)

            if ai > knee and roofline == False:
                x_left = knee
                y_patch[1] = peak
                prev_xs = scatter_data['xs'][-1]
                prev_xs[2] = knee
                prev_xs[3] = knee

                prev_ys = scatter_data['ys'][-1]
                prev_ys[2] = peak

                scatter_data['xs'][-1] = prev_xs
                scatter_data['ys'][-1] = prev_ys

                roofline = True
            
            x_patch = [x_left, x_left, x_right, x_right]

        #     peak, slope = emp_roofs[gpu]['HBM_' + op + '_' + data_type]
        #     x_intersect = peak / slope
        #     x_slope = AI[AI <= x_intersect] # Generate x values for sloped line
        #     x_slope = np.array([float(x_slope[0]), float(x_slope[-1])])
        #     x_horizontal = AI[AI >= x_intersect] # Generate x values for horizontal line
        #     x_horizontal = [x_horizontal.min(), x_horizontal.max()]
        #     y_slope = slope * x_slope # Generate y values for sloped line (bandwidth * AI)
        #     y_slope = [y_slope.min(), y_slope.max()]
        #     y_horizontal = np.full_like(x_horizontal, peak) # Generate y values for horizontal line (peak performance)
        #     y_horizontal = [y_horizontal.min(), y_horizontal.max()]
        # # print(f'Plotting {mem} {op} {data}...')
        # # print(f'Slope (Bw): {np.round(slope, 2)}')

        # # Create data sources for the lines
        #     source_slope = ColumnDataSource(data=dict(x=x_slope, y=y_slope, gpu=[gpu]*2))
        #     source_horizontal = ColumnDataSource(data=dict(x=x_horizontal, y=y_horizontal, gpu=[gpu]*2))

        #     # Plot the lines without labels
        #     slope = p.line('x', 'y', source=source_slope, line_width=2, color='black', visible=False)
        #     peak = p.line('x', 'y', source=source_horizontal, line_width=2, color='black', line_dash='dashed', visible=False)
        #     slope.name = 'roofline'
        #     peak.name = 'roofline'
            

            scatter_data["x"].append(float(f"{ai:.10f}"))
            scatter_data["y"].append(float(f"{perf:.10f}"))
            scatter_data["op"].append(op)
            scatter_data["data_type"].append(data_type)
            scatter_data["gpu"].append(gpu)
            scatter_data["power"].append(power)
            scatter_data["xs"].append(x_patch)
            scatter_data["ys"].append(y_patch)
            scatter_data["color"].append(None)
            # scatter_data["slope"].append(slope)
            # scatter_data["peak"].append(peak)
            
            if gpu not in gpu_sources:
                gpu_sources[gpu] = []
# print(scatter_data)

for gpu, gpu_df in emp_roofs.items():

    min_power = df[df['GPU'] == gpu]['Power'].min()
    max_power = df[df['GPU'] == gpu]['Power'].max()
    min_power = min_power
    max_power = max_power
    norm = plt.Normalize(
        min_power,
        max_power,
    )
    color_map = LinearSegmentedColormap.from_list(
        'green_to_red', plt.cm.get_cmap('hsv')(np.linspace(0.33, 0, 256))
    )

    # Convert to a hex palette
    rgba_colors = [to_rgba(color_map(i / 255), alpha=0.7) for i in range(256)]  # 0.3 = 30% opacity
    hex_colors_with_alpha = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}{int(a*255):02x}' 
                            for r, g, b, a in rgba_colors]

    # Create Bokeh color mapper
    color_mapper = LinearColorMapper(palette=hex_colors_with_alpha, low=min_power, high=max_power)

    # Add the color bar
    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        visible=False
    )

    y_axis_style = {
        "text_font": p.yaxis[0].axis_label_text_font,
        "text_font_size": p.yaxis[0].axis_label_text_font_size,
        "text_font_style": p.yaxis[0].axis_label_text_font_style,
        "text_color": p.yaxis[0].axis_label_text_color,
    }
    p.add_layout(color_bar, 'right')

    power_label = Title(
        text='Power (W)',
        text_font=y_axis_style["text_font"],
        text_font_size=y_axis_style["text_font_size"],
        text_font_style=y_axis_style["text_font_style"],
        text_color=y_axis_style["text_color"],
        offset=220,
        visible=False,
    )

    p.add_layout(power_label, 'right')

    for i, gpu_entry in enumerate(scatter_data['gpu']):
        if gpu_entry == gpu:
            power = scatter_data['power'][i]
            # if gpu == 'MI300A':
            #     print(power)
            #     print(df[df['GPU'] == 'GPU']['Power'].min())
            #     print(df[df['GPU'] == 'GPU']['Power'].max())
            #     print(to_hex(color_map(norm(power))))
            # print(to_hex(color_map(norm(power))))
            scatter_data['color'][i] = to_hex(color_map(norm(power)))
    
    # mem = key.split('_')[0]
    # op = key.split('_')[1]
    # data = key.split('_')[2]
    # for key, (peak, slope) in gpu_df:

    #     x_intersect = peak / slope
    #     x_slope = AI[AI <= x_intersect] # Generate x values for sloped line
    #     x_slope = np.array([float(x_slope[0]), float(x_slope[-1])])
    #     x_horizontal = AI[AI >= x_intersect] # Generate x values for horizontal line
    #     x_horizontal = [x_horizontal.min(), x_horizontal.max()]
    #     y_slope = slope * x_slope # Generate y values for sloped line (bandwidth * AI)
    #     y_slope = [y_slope.min(), y_slope.max()]
    #     y_horizontal = np.full_like(x_horizontal, peak) # Generate y values for horizontal line (peak performance)
    #     y_horizontal = [y_horizontal.min(), y_horizontal.max()]
    # # print(f'Plotting {mem} {op} {data}...')
    # # print(f'Slope (Bw): {np.round(slope, 2)}')

    # # Create data sources for the lines
    #     source_slope = ColumnDataSource(data=dict(x=x_slope, y=y_slope, gpu=[gpu]*2))
    #     source_horizontal = ColumnDataSource(data=dict(x=x_horizontal, y=y_horizontal, gpu=[gpu]*2))

    #     # Plot the lines without labels
    #     slope = p.line('x', 'y', source=source_slope, line_width=2, color='black', visible=False)
    #     peak = p.line('x', 'y', source=source_horizontal, line_width=2, color='black', line_dash='dashed', visible=False)
    #     slope.name = 'roofline'
    #     peak.name = 'roofline'
    color_bar.name = 'power'
    power_label.name = 'power'
    
    # gpu_sources[gpu].append(slope)
    # gpu_sources[gpu].append(peak)
    gpu_sources[gpu].append(color_bar)
    gpu_sources[gpu].append(power_label)

source_all = ColumnDataSource(data=scatter_data)
source_full = ColumnDataSource(data=scatter_data)

# roofline_all = ColumnDataSource(data=roofline_data)
# roofline_full = ColumnDataSource(data=roofline_data)

scatter_renderer = p.scatter(
    x="x", y="y", source=source_all,
    size=12, color='black', marker='circle', visible=False
)

power_renderer = p.patches(
    xs='xs',
    ys='ys',
    source=source_all,
    fill_color='color',
    fill_alpha=0.7,
    line_color=None,
    level='underlay',
    visible=False
)

hover_tool = HoverTool(
    tooltips=tooltips,
    renderers=[scatter_renderer],
)
p.add_tools(hover_tool)

# Create CheckboxGroups for Operation Types and Data Types with no active selections
op_checkboxes = RadioGroup(labels=[op for op in op_types], active=0)
data_checkboxes = RadioGroup(
    labels=[f"{data}" for data in data_types],
    active=0
)
gpus = df['GPU'].unique()
gpu_checkboxes = RadioGroup(labels=[gpu for gpu in gpus], active=0)

bigger_font_css = """
:host {
    /* everything inside this RadioGroup inherits this size */
    font-size: 16px;
    font-family: Arial, sans-serif;
}
"""

for rg in (op_checkboxes, data_checkboxes, gpu_checkboxes):
    rg.stylesheets.append(bigger_font_css)

# Define the JavaScript Callback for RadioGroup Interactions
callback_code = """
    // Function to update visibility based on active radiogroups
    function update_visibility() {
        const selected_op = op_checkboxes.active !== null ? op_checkboxes.labels[op_checkboxes.active].split(' ')[0] : null;
        const selected_data = data_checkboxes.active !== null ? data_checkboxes.labels[data_checkboxes.active] : null;
        const selected_gpu = gpu_checkboxes.active !== null ? gpu_checkboxes.labels[gpu_checkboxes.active] : null;

        const full_data = source_full.data;
        const filtered = {
            x: [], y: [], op: [], data_type: [], gpu: [], power: [],
            xs: [], ys: [], color: []
        };

        for (let i = 0; i < full_data.x.length; i++) {
            const op_match = selected_op === null || full_data.op[i] === selected_op;
            const data_match = selected_data === null || full_data.data_type[i] === selected_data;
            const gpu_match = selected_gpu === null || full_data.gpu[i] === selected_gpu;

            if (op_match && data_match && gpu_match) {
                filtered.x.push(full_data.x[i]);
                filtered.y.push(full_data.y[i]);
                filtered.op.push(full_data.op[i]);
                filtered.data_type.push(full_data.data_type[i]);
                filtered.gpu.push(full_data.gpu[i]);
                filtered.power.push(full_data.power[i]);
                filtered.xs.push(full_data.xs[i]);
                filtered.ys.push(full_data.ys[i]);
                filtered.color.push(full_data.color[i]);
            }
        }

        // Replace the source data with filtered view
        source_all.data = filtered;
        source_all.change.emit();
        p.change.emit();
        scatter_renderer.visible = true;
        power_renderer.visible = true;
        console.log(source_all.data);

        for (const [gpu, renderers] of Object.entries(gpu_sources)) {
            const gpu_visible = selected_gpu === null || selected_gpu === gpu;
            for (const renderer of renderers) {
                if (renderer.name === 'power') {
                    renderer.visible = gpu_visible;
                }
            }
        }

        for (const [gpu, ops] of Object.entries(roofline_sources)) {
            for (const [op, data_types] of Object.entries(ops)) {
                for (const [data_type, renderers] of Object.entries(data_types)) {
                    const should_show =
                        gpu === selected_gpu &&
                        op === selected_op &&
                        data_type === selected_data;

                    for (const renderer of renderers) {
                        renderer.visible = should_show;
                    }
                }
            }
        }
    }

    update_visibility();
"""
# Create the CustomJS callback
callback = CustomJS(args=dict(op_checkboxes=op_checkboxes, data_checkboxes=data_checkboxes, gpu_checkboxes=gpu_checkboxes,
                              gpu_sources=gpu_sources, roofline_sources=roofline_sources, source_all=source_all, source_full=source_full, scatter_renderer=scatter_renderer, power_renderer=power_renderer, p=p), code=callback_code)

# Attach the callback to the 'active' property change
op_checkboxes.js_on_change('active', callback)
data_checkboxes.js_on_change('active', callback)
gpu_checkboxes.js_on_change('active', callback)

# Layout and show
widgets = column(op_checkboxes, data_checkboxes, gpu_checkboxes)

centered_plot = row(
    Spacer(width=0, sizing_mode="stretch_width"),
    p,
    widgets,
    p.toolbar,
    Spacer(width=0, sizing_mode="stretch_width"),
    sizing_mode="stretch_width"
)

# Final layout
layout = column(
    Spacer(height=70),
    centered_plot,
    Spacer(height=70),
    sizing_mode="stretch_width"
)

# 3 – Write a self-contained interactive HTML file
output_file(f"{filename}.html")   # sets the output file name
show(layout)  

# output_file(filename=f'{filename}.html')
# show(layout)

plt.savefig(filename, bbox_inches='tight')
print(f'Plot saved as {filename}.png')