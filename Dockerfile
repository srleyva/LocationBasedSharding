FROM jupyter/scipy-notebook

RUN pip install s2sphere s2

RUN conda install matplotlib shapely cartopy plotly folium

COPY shard.json /tmp/shard.json