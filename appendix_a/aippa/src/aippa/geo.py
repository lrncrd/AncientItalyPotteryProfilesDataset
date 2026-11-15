import plotly.express as px
import numpy as np

def GeoPlot(geodf_joined, title= None):
    geo_df = geodf_joined


    geo_df = geo_df.to_crs(epsg=4326)

    geo_df["x"] = geo_df.geometry.x
    geo_df["y"] = geo_df.geometry.y



    fig = px.scatter_map(geo_df,
                            lat=geo_df["y"], 
                            lon=geo_df["x"],
                            center={"lat": np.mean(geo_df.geometry.y), "lon": np.mean(geo_df.geometry.x)},
                            map_style="open-street-map",
                            zoom=4.5,
                            hover_name="site",
                            hover_data={"site": False, "x": False, "y": False,  
                            },
                            title= title,
                            color_discrete_sequence=px.colors.qualitative.Set1,
                            opacity=1,
                            width=700, height=500
                            )
                            

    fig.update_traces()
    fig.update_layout()

    fig.show()