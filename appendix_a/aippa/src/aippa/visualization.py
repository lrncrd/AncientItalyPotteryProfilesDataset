import os
from pathlib import Path
import plotly.express as px
import numpy as np
import pandas as pd
import hdbscan
from PIL import Image
import matplotlib.pyplot as plt
from bokeh.plotting import figure, curdoc
#from bokeh.palettes import brewer, Category10, Dark2, Spectral10, Tab10, Viridis3, Viridis256
from bokeh.layouts import column, row
from bokeh.io import show, output_notebook, curdoc, save, output_file
from bokeh.transform import factor_cmap, factor_mark
#from bokeh.palettes import Category10_10, Category10, Category20_12, RdGy
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    TapTool,
    Slider,
    Div,
    RadioButtonGroup, 
    TextInput, 
    CustomJS,
    Range1d,
    CategoricalColorMapper
    )
from io import BytesIO
import base64
#from bokeh.plotting import figure, show, output_notebook
import seaborn as sns

import plotly.io as pio
import plotly.express as px

pio.renderers.default = "notebook_connected"






def embeddable_image(data):
    img_data = 255 - 15 * data.astype(np.uint8)
    image = Image.fromarray(img_data, mode='L')
    thresh = 250
    fn = lambda x : 255 if x > thresh else 0
    image = image.convert('L').point(fn, mode='L')
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

def ScatterImage(reduced, info_selected, potds, hue="black", title = "ScatterImage", point_size = 5, hover_data = [], hover_size = 100, figsize = (500, 400), fill_alpha = 1, palette = px.colors.qualitative.Set1, output = "notebook"):
    """
    Plot an interactive 2D Scatterplot, which shows pots profile and some archaeological information

    Parameters:
        reduced (numpy.ndarray): Input data
        info_selected (pandas.Dataframe or pandas.Series): Additional information or labels associated with the data.
        potds (aippa.utils.PotDataset): An aippa.utils.PotDataset
        hue (str, list or pandas.Series): Categorical membership variable used to color the scatter plot (default: "black")
        title (str): Title of the scatter plot (default: "ScatterImage")
        point_size (int): Size of the data points (default: 5)
        hover_data (list): Information to display in the hover tooltip. Elements in the list should be column names from "info_selected". Up to 4 information fields can be specified (default: [])
        hover_size (int): Size of the hover image (default: 100)
        figsize (tuple): Figure size (width, height) in pixels (default: (500, 400)).
        fill_alpha (float): Alpha value for the fill color of the data points (default: 1)
        palette (px.colors.qualitative): Color palette for categorical data (https://plotly.com/python/discrete-color/) (default: px.colors.qualitative.Set1)
        output (str): Output style for the plot. Valid options are 'notebook' or 'file' (default: "notebook")
        """
    reduced = reduced[:,:2]
    plot_df = pd.DataFrame(reduced, info_selected.index, columns=('x', 'y'))
    plot_df = plot_df.join(info_selected)


    if isinstance(hue, pd.Series):
        my_legend_name = str(hue.name)
        plot_df = plot_df.loc[hue.index]
        labels = hue.unique().astype(str)
        color_mapping = CategoricalColorMapper(factors=labels, palette=palette)
    else:
        pass
    
    plot_df["image"] = list(map(embeddable_image, potds[plot_df.index]))  

    
    plot_figure = figure(
    title=title,
    width=figsize[0],
    height=figsize[1],
    tools=('pan, wheel_zoom, reset, save'))

    if any(hover_data):
        assert len(hover_data) <= 4, "Warnings: hover data is longer than 4"
        len_list = len(hover_data)
        elem = [None] * len_list
        
        for i, n in enumerate(hover_data):
            elem[i] = n
        if len(elem) == 1:
            plot_figure.add_tools(HoverTool(tooltips=f"""
                <div>
                    <div>
                        <img src='@image' width="{hover_size}" style='float: left; margin: 5px 5px 5px 5px'/>
                    </div>
                    <div>
                        <span style='font-size: 16px; color: #224499'>@ids:</span>
                    </div>
                        <span style='font-size: 10px'>@{elem[0]}</span>
                </div>
                """))
        elif len(elem) == 2:
            plot_figure.add_tools(HoverTool(tooltips=f"""
                <div>
                    <div>
                        <img src='@image' width="{hover_size}" style='float: left; margin: 5px 5px 5px 5px'/>
                    </div>
                    <div>
                        <span style='font-size: 16px; color: #224499'>@ids:</span>
                    </div>
                        <span style='font-size: 10px'>@{elem[0]}</span>
                    <div>
                        <span style='font-size: 10px'>@{elem[1]}</span>
                </div>
                """))
        elif len(elem) == 3:
            plot_figure.add_tools(HoverTool(tooltips=f"""
                <div>
                    <div>
                        <img src='@image' width="{hover_size}" style='float: left; margin: 5px 5px 5px 5px'/>
                    </div>
                    <div>
                        <span style='font-size: 16px; color: #224499'>@ids:</span>
                    </div>
                        <span style='font-size: 10px'>@{elem[0]}</span>
                    <div>
                        <span style='font-size: 10px'>@{elem[1]}</span>
                    <div>
                        <span style='font-size: 10px'>@{elem[2]}</span>
                </div>
                """))
        elif len(elem) == 4:
            plot_figure.add_tools(HoverTool(tooltips=f"""
                <div>
                    <div>
                        <img src='@image' width="{hover_size}" style='float: left; margin: 5px 5px 5px 5px'/>
                    </div>
                    <div>
                        <span style='font-size: 16px; color: #224499'>@ids:</span>
                    </div>
                        <span style='font-size: 10px'>@{elem[0]}</span>
                    <div>
                        <span style='font-size: 10px'>@{elem[1]}</span>
                    <div>
                        <span style='font-size: 10px'>@{elem[2]}</span>
                    <div>
                        <span style='font-size: 10px'>@{elem[3]}</span>
                </div>
                """))
    else:
        plot_figure.add_tools(HoverTool(tooltips=f"""
            <div>
                <div>
                    <img src='@image' width="{hover_size}" style='float: left; margin: 5px 5px 5px 5px'/>
                </div>
                <div>
                    <span style='font-size: 16px; color: #224499'>@ids:</span>
            </div>
            """))
        


    if isinstance(hue, pd.Series):
        for h in hue.unique():
            selected_hue = plot_df.loc[plot_df[hue.name] == h]
            datasource = ColumnDataSource(selected_hue)
            plot_figure.scatter(
            'x',
            'y',
            source=datasource,
            color=dict(field=my_legend_name, transform=color_mapping),
            legend_group=my_legend_name,
            line_alpha=1,
            fill_alpha=fill_alpha,
            size=point_size)
    else:
        datasource = ColumnDataSource(plot_df)
        plot_figure.scatter(
        'x',
        'y',
        source=datasource,
        color=hue,
        line_alpha=1,
        fill_alpha=fill_alpha,
        size=point_size)

    if output == "notebook":
        output_notebook()
    if output == "file":
        output_file("Scatterimage.html")
    else:
        assert "Select output style : 'notebook' or 'file'"
        

    plot_figure.legend.click_policy = 'mute'

    show(plot_figure)






