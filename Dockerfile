FROM jupyter/scipy-notebook

USER root

RUN pip install s2sphere

RUN conda install matplotlib shapely cartopy plotly

COPY shard.json /tmp/shard.json