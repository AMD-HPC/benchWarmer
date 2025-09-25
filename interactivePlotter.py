import pandas as pd
import numpy as np
import math
import argparse
import os
import matplotlib
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
import bokeh.layouts as bl
from bokeh.io import curdoc
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_hex, LinearSegmentedColormap, to_rgba
from bokeh.io import output_file
from bokeh.models import HoverTool
import glob

# Get filenames from input args
parser = argparse.ArgumentParser()
parser.add_argument('--results', required=True, help='Provide runtimes.csv filename')
parser.add_argument('--rocstar', required=False, help='Provide rocstar directory')
parser.add_argument('-f', '--filename', required=True, help='Filename which you want the plot to be')

args = parser.parse_args()
#AI, DATA, OP
print('Parsing arguments...')


df = pd.read_csv(args.results) 


df['PERF'] = df['PERF'] / 1000
df['DATA'] = df['DATA'].str.upper()
df['OP'] = df['OP'].str.upper()
data_types = df['DATA'].unique()
op_types = df['OP'].unique()
gpus = df['GPU'].unique()
rocstar_dir = args.rocstar
print(f'Data Types: {data_types}')
print(f'Operations: {op_types}')
if rocstar_dir:
    print(f'rocstar directory: {rocstar_dir}')
else:
    print('No rocstar directory provided.')
filename = args.filename
print('Parsing complete.')



if rocstar_dir:
    print('Parsing power metrics')
    if 'Power' not in df.columns:
        df['Power'] = None
    for _, row in df.iterrows():
        gpu = row['GPU']
        iterations = row['Iterations']
        data_type = row['DATA']
        op_type = row['OP']
        file = f'{rocstar_dir}/{gpu}/hw_metrics_{iterations}_{op_type.lower()}_{data_type.lower()}.csv'
        power_df = pd.read_csv(file)
        power_df.columns = power_df.columns.str.strip()
        if power_df.empty:
            print(f'Power metrics file is empty: {file}')
            exit(1)
        max_power = power_df['Power[W]'].max()
        df.loc[row.name, 'Power'] = max_power
    print('Parsed power metrics')

if 'Power' not in df.columns:
    print("Power values are missing. Pass in directory to power values")
    exit(1)
df['Power'] = df['Power'].astype(float)
df['Power'] = df['Power'].fillna(0)
# df = df[~(df.filter(like='GEMM').gt(0).any(axis=1))]
# Consolidate rows for GPU 'MI250X_two_gcd'
if 'MI250X_two_gcd' in df['GPU'].values:
    ai_cols = [col for col in df.columns if 'AI_' in col]
    df_grouped = df[df['GPU'] == 'MI250X_two_gcd'].groupby(['Iterations'] + ['OP'] + ['DATA'] + ['AI'], as_index=False).agg(
        {col: 'sum' if col in ['PERF', 'POWER'] else 'first' for col in df.columns if col}
    )
    df = pd.concat([df[df['GPU'] != 'MI250X_two_gcd'], df_grouped], ignore_index=True)
df['GPU'] = df['GPU'].replace({'MI250X_two_gcd': 'MI250X (2 GCDs)'})

df = df.sort_values(by='AI')

peak_bw = {}

theo_bw = {}
theo_perf = {}
theo_peak_bw = defaultdict(lambda: defaultdict(dict))
theo_peak_perf = defaultdict(lambda: defaultdict(dict))
gpus = ['MI250X', 'MI250X (2 GCDs)', 'MI300A', 'MI300X', 'A100', 'H100']
for gpu in gpus:
    # for 
    if gpu not in ['A100', 'H100', 'MI250X (2 GCDs)']:
        emp_df = pd.read_csv(f'./roofline_csvs/roofline_{gpu}.csv')
        emp_df['HBMBw'] = emp_df['HBMBw'].astype(float)
        peak_bw[gpu] = emp_df['HBMBw'].mean() / 1000
    else:
        peak_bw['H100'] = 2982 / 1000
        peak_bw['A100'] = 1379 / 1000
        peak_bw['MI250X (2 GCDs)'] = 2500 / 1000

theo_bw['H100'] = 3.35
theo_bw['MI300X'] = 5.3
theo_bw['MI250X (2 GCDs)'] = 3.2
theo_bw['MI250X'] = 1.6
theo_bw['MI300A'] = 5.3
theo_bw['A100'] = 1.555

theo_perf['H100'] = 66.9
theo_perf['MI300X'] = 163.4
theo_perf['MI250X (2 GCDs)'] = 90.5
theo_perf['MI250X'] = 45.25
theo_perf['MI300A'] = 122.6
theo_perf['A100'] = 19.5

for gpu in gpus:
    for op in op_types:
        for data in data_types:
            # print(gpu, op, data)
            theo_peak_bw[gpu][op][data] = theo_bw[gpu]

for gpu in gpus:
    for op in op_types:
        for data in data_types:
            theo_peak_perf[gpu][op][data] = theo_perf[gpu]

theo_peak_perf['H100']['GEMM']['FP32'] = 66.9
theo_peak_perf['H100']['GEMM']['BF16'] = 989.4
theo_peak_perf['H100']['GEMM']['FP8'] = 1978.9

theo_peak_perf['MI300X']['GEMM']['FP32'] = 163.4
theo_peak_perf['MI300X']['GEMM']['BF16'] = 1307.4
theo_peak_perf['MI300X']['GEMM']['FP8'] = 2614.9

kernels = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
emp_roofs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
theo_emp_roofs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

print('Gathering runtime data')

for _, row in df.iterrows():
    subset_df = df[(df['GPU'] == row['GPU']) & (df['OP'] == row['OP']) & (df['DATA'] == row['DATA'])]
    gpu = row['GPU']
    op = row['OP']
    data = row['DATA']
    ai = row['AI']
    perf = row['PERF']
    power = row['Power']
    if gpu == 'MI250X_two_gcd':
        kernels[gpu][op][data][ai][0] += ((perf, power))
    else:
        kernels[gpu][op][data][ai].append((perf, power))
    emp_roofs[gpu][op][data] = (peak_bw[gpu], subset_df['PERF'].max())
    theo_emp_roofs[gpu][op][data] = (theo_peak_bw[gpu][op][data], theo_peak_perf[gpu][op][data])

print('Gathered runtime data')

print('Plotting rooflines...')

AI = np.logspace(-2, 6, 10000)

# Create Bokeh figure

# round to nearest power of 10
lowest_ai = df['AI'].min()
highest_ai = df['AI'].max()
x_min = 10 ** np.floor(np.log10(lowest_ai))
x_max = 10 ** np.ceil(np.log10(highest_ai))

p = figure(x_axis_type='log', y_range=(0.1, 1e4), x_range=(x_min, x_max), y_axis_type='log', title='Empirical Rooflines with Power',
           x_axis_label='Arithmetic Intensity (FLOPs/Byte)', toolbar_location="right",
           y_axis_label='Performance (TFLOPs/sec)', tools='wheel_zoom,box_zoom,reset,save', width=900, height=600)
p.title.text_font_size = '14pt'
p.xaxis.axis_label_text_font_size = '16pt'
p.yaxis.axis_label_text_font_size = '16pt'
p.xaxis.major_label_text_font_size = '14pt'
p.yaxis.major_label_text_font_size = '14pt'

p.toolbar_location = None

roofline_sources = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
peak_roofline_sources = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
gpu_sources = {}

log_width = 0.303

scatter_data_gemm = {
    "x": [],
    "y": [],
    "op": [],
    "data_type": [],
    "gpu": [],
    "power": [],
    "color": [],
    "m": [],
    "n": [],
    "k": []
}

scatter_data = {
    "x": [],
    "y": [],
    "op": [],
    "data_type": [],
    "gpu": [],
    "power": [],
    "color": [],
    "xs": [],
    "ys": []
}

for gpu, gpu_dict in kernels.items():
    # print(max([values[-1].max() for values in kernels.values()]))
    for op_type, op_dict in gpu_dict.items():
        for data_type, data_info in op_dict.items():
            # color = gpu_type_colors[gpu]
            y_range = p.y_range
            # color_map = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
            roofline = False
            slope, peak = emp_roofs[gpu][op_type][data_type]
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
            peak_line = p.line('x', 'y', source=source_horizontal, line_width=2, color='black', visible=False)
            slope_line.name = 'roofline'
            peak_line.name = 'roofline'

            # roofline_data = {}

            # roofline_data["slope"].append(slope)
            # roofline_data["peak"].append(peak)

            roofline_sources[gpu][op_type][data_type].append(slope_line)
            roofline_sources[gpu][op_type][data_type].append(peak_line)

            slope, peak = theo_emp_roofs[gpu][op_type][data_type]
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
            slope_line = p.line('x', 'y', source=source_slope, line_width=2, color='black', line_dash='dashed', visible=False)
            peak_line = p.line('x', 'y', source=source_horizontal, line_width=2, color='black', line_dash='dashed', visible=False)
            slope_line.name = 'roofline'
            peak_line.name = 'roofline'

            # roofline_data = {}

            # roofline_data["slope"].append(slope)
            # roofline_data["peak"].append(peak)

            peak_roofline_sources[gpu][op_type][data_type].append(slope_line)
            peak_roofline_sources[gpu][op_type][data_type].append(peak_line)
            for ai, data in data_info.items():
                for perf, power in data:
                    # Create source with all data including power
                    x_left = ai / (10 ** (log_width / 2))
                    x_right = ai * (10 ** (log_width / 2))
                    bar_width = x_right - x_left
                    # color = to_hex(color_map(norm(power)))
                    # print(gpu)
                    slope, peak = emp_roofs[gpu][op_type][data_type]

                    if ai < peak / slope:
                        y_patch = [0.05, slope * x_left, slope * x_right, 0.05]
                    else:
                        # print(op_type, data_type, ai, peak)
                        y_patch = [0.05, peak, peak, 0.05]
                    
                    knee = peak / slope
                    # print(knee)
                    if op_type == 'GEMM':                    
                        scatter_data_gemm["x"].append(float(f"{ai:.10f}"))
                        scatter_data_gemm["y"].append(float(f"{perf:.10f}"))
                        scatter_data_gemm["op"].append(op_type)
                        scatter_data_gemm["data_type"].append(data_type)
                        scatter_data_gemm["gpu"].append(gpu)
                        scatter_data_gemm["power"].append(power)
                        scatter_data_gemm["color"].append(None)
                        # print(gpu, ai, perf)
                        scatter_data_gemm["m"].append(df[(df['GPU'] == gpu) & (df['DATA'] == data_type) & (df['OP'] == op_type) & (df['AI'] == ai) & (df['PERF'] == perf)].iloc[0]['M'])
                        scatter_data_gemm["n"].append(df[(df['GPU'] == gpu) & (df['DATA'] == data_type) & (df['OP'] == op_type) & (df['AI'] == ai) & (df['PERF'] == perf)].iloc[0]['N'])
                        scatter_data_gemm["k"].append(df[(df['GPU'] == gpu) & (df['DATA'] == data_type) & (df['OP'] == op_type) & (df['AI'] == ai) & (df['PERF'] == perf)].iloc[0]['K'])
                    else:
                        if ai > knee and roofline == False:
                            x_left = knee
                            y_patch[1] = peak
                            if op_type in scatter_data['op'] and data_type in scatter_data['data_type'] and scatter_data['xs']:
                                prev_xs = scatter_data['xs'][-1]
                                prev_xs[2] = knee
                                prev_xs[3] = knee

                                prev_ys = scatter_data['ys'][-1]
                                prev_ys[2] = peak

                                scatter_data['xs'][-1] = prev_xs
                                scatter_data['ys'][-1] = prev_ys

                                roofline = True
                        x_patch = [x_left, x_left, x_right, x_right]
                        # if gpu == 'MI250X' and data_type == 'FP64' and op_type == 'MULADD':
                        #     print(ai, x_patch, y_patch)
                        scatter_data["x"].append(float(f"{ai:.10f}"))
                        scatter_data["y"].append(float(f"{perf:.10f}"))
                        scatter_data["op"].append(op_type)
                        scatter_data["data_type"].append(data_type)
                        scatter_data["gpu"].append(gpu)
                        scatter_data["power"].append(power)
                        # print(df[(df['GPU'] == gpu) & (df[f'AI_HBM_{op}_{data_type}'] == ai) & (df['PERF'] == perf)])
                        # scatter_data["m"].append(df[(df['GPU'] == gpu) & (df[f'AI_HBM_{op}_{data_type}'] == ai) & (df['PERF'] == perf)].iloc[0]['M'])
                        # scatter_data["n"].append(df[(df['GPU'] == gpu) & (df[f'AI_HBM_{op}_{data_type}'] == ai) & (df['PERF'] == perf)].iloc[0]['N'])
                        # scatter_data["k"].append(df[(df['GPU'] == gpu) & (df[f'AI_HBM_{op}_{data_type}'] == ai) & (df['PERF'] == perf)].iloc[0]['K'])
                        scatter_data["xs"].append(x_patch)
                        scatter_data["ys"].append(y_patch)
                        scatter_data["color"].append(None)

                    # scatter_data["M"].append(df[[df['GPU'] == gpu] & [df[f'AI_HBM_{op}_{data_type}'] == ai]])
                    # scatter_data["slope"].append(slope)
                    # scatter_data["peak"].append(peak)
                    
                    if gpu not in gpu_sources:
                        gpu_sources[gpu] = []
# print(scatter_data)
for data in zip(*scatter_data.values()):
    ai, perf, op_type, data_type, gpu, power, color, x_patch, y_patch = data
    # print(gpu, data_type, op_type)
    # if gpu == 'MI250X' and data_type == 'FP64' and op_type == 'MULADD':
    #     print(ai, x_patch, y_patch)

for gpu, gpu_df in emp_roofs.items():
    min_power = 0
    max_power = df[df['GPU'] == gpu]['Power'].max()
    # min_power = min_power
    # max_power = max_power
    if gpu == 'H100':
        max_power = 700
    norm = plt.Normalize(
        min_power,
        max_power,
    )
    color_map = LinearSegmentedColormap.from_list(
        'green_to_red', matplotlib.colormaps.get_cmap('hsv')(np.linspace(0.33, 0, 256))
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

    for i, gpu_entry in enumerate(scatter_data_gemm['gpu']):
        if gpu_entry == gpu:
            power = scatter_data_gemm['power'][i]
            scatter_data_gemm['color'][i] = to_hex(color_map(norm(power)))
    
    for i, gpu_entry in enumerate(scatter_data['gpu']):
        if gpu_entry == gpu:
            power = scatter_data['power'][i]
            scatter_data['color'][i] = to_hex(color_map(norm(power)))

    color_bar.name = 'power'
    power_label.name = 'power'
    
    # gpu_sources[gpu].append(slope)
    # gpu_sources[gpu].append(peak)
    gpu_sources[gpu].append(color_bar)
    gpu_sources[gpu].append(power_label)

source_all = ColumnDataSource(data=scatter_data)
source_full = ColumnDataSource(data=scatter_data)

source_all_gemm = ColumnDataSource(data=scatter_data_gemm)
source_full_gemm = ColumnDataSource(data=scatter_data_gemm)

# roofline_all = ColumnDataSource(data=roofline_data)
# roofline_full = ColumnDataSource(data=roofline_data)

scatter_renderer = p.scatter(
    x="x", y="y", source=source_all,
    size=12, color='black', marker='circle', visible=False
)

scatter_renderer_gemm = p.scatter(
    x="x", y="y", source=source_all_gemm,
    fill_alpha=0.5,
    size=6, color="color", marker='circle', visible=False
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

tooltips_gemm = [("AI", "@x"), ("M", "@m"), ("N", "@n"), ("K", "@k"), ("Performance", "@y TFLOPS/sec"), ("Operation", "@op"), ("Data Type", "@data_type"), ("Power", "@power W")]
hover_tool_gemm = HoverTool(
    tooltips=tooltips_gemm,
    renderers=[scatter_renderer_gemm],
)

tooltips = [("AI", "@x"), ("Performance", "@y TFLOPS/sec"), ("Operation", "@op"), ("Data Type", "@data_type"), ("Power", "@power W")]
hover_tool = HoverTool(
    tooltips=tooltips,
    renderers=[scatter_renderer],
)

p.add_tools(hover_tool)
p.add_tools(hover_tool_gemm)

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

corner_label = Label(
    x=20, y=440, x_units='screen', y_units='screen',
    text='- - - Indicates Peak Theoretical Performance\n—   Indicates Peak Empirical Performance',
    text_font_size='16pt',
    text_color='black',
    background_fill_color='white',
    background_fill_alpha=0.7
)
p.add_layout(corner_label)

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
            x: [], y: [], op: [], data_type: [], gpu: [], power: [], xs: [], ys: [], color: []
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

        const full_data_gemm = source_full_gemm.data;
        const filtered_gemm = {
            x: [], y: [], op: [], data_type: [], gpu: [], power: [],
            color: [], m: [], n: [], k: []
        };

        for (let i = 0; i < full_data_gemm.x.length; i++) {
            const op_match = selected_op === null || full_data_gemm.op[i] === selected_op;
            const data_match = selected_data === null || full_data_gemm.data_type[i] === selected_data;
            const gpu_match = selected_gpu === null || full_data_gemm.gpu[i] === selected_gpu;

            if (op_match && data_match && gpu_match) {
                filtered_gemm.x.push(full_data_gemm.x[i]);
                filtered_gemm.y.push(full_data_gemm.y[i]);
                filtered_gemm.op.push(full_data_gemm.op[i]);
                filtered_gemm.data_type.push(full_data_gemm.data_type[i]);
                filtered_gemm.gpu.push(full_data_gemm.gpu[i]);
                filtered_gemm.power.push(full_data_gemm.power[i]);
                filtered_gemm.color.push(full_data_gemm.color[i]);
                filtered_gemm.m.push(full_data_gemm.m[i]);
                filtered_gemm.n.push(full_data_gemm.n[i]);
                filtered_gemm.k.push(full_data_gemm.k[i]);
            }
        }

        // Replace the source data with filtered_gemm view
        source_all_gemm.data = filtered_gemm;
        source_all_gemm.change.emit();
        p.change.emit();
        scatter_renderer_gemm.visible = true;
        power_renderer.visible = true;

        console.log(source_all_gemm.data);

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

        for (const [gpu, ops] of Object.entries(peak_roofline_sources)) {
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
callback = CustomJS(args=dict(op_checkboxes=op_checkboxes, data_checkboxes=data_checkboxes, gpu_checkboxes=gpu_checkboxes, gpu_sources=gpu_sources, roofline_sources=roofline_sources, 
                              peak_roofline_sources=peak_roofline_sources, source_all=source_all, source_full=source_full, scatter_renderer=scatter_renderer, source_all_gemm=source_all_gemm, source_full_gemm=source_full_gemm, scatter_renderer_gemm=scatter_renderer_gemm, power_renderer=power_renderer, p=p), code=callback_code)

# Attach the callback to the 'active' property change
op_checkboxes.js_on_change('active', callback)
data_checkboxes.js_on_change('active', callback)
gpu_checkboxes.js_on_change('active', callback)

from bokeh.events import DocumentReady
curdoc().on_event(DocumentReady, lambda event: callback.execute({}))

# Layout and show
widgets = bl.column(op_checkboxes, data_checkboxes, gpu_checkboxes)

centered_plot = bl.row(
    Spacer(width=0, sizing_mode="stretch_width"),
    p,
    widgets,
    p.toolbar,
    Spacer(width=0, sizing_mode="stretch_width"),
    sizing_mode="stretch_width"
)

# Final layout
layout = bl.column(
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
