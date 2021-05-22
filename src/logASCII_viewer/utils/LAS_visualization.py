import pandas as pd
import numpy as np
import seaborn as sns
from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams.update({
                    'axes.labelsize': 12,
                    'xtick.labelsize': 12,
                    'legend.fontsize': 'small'
                    })


def plotly_log(df, cols):
    feat = cols
    track = len(cols)
    index = df.DEPTH
    fig = make_subplots(rows=1, cols=track, shared_yaxes=True, horizontal_spacing=0.03,
                        specs=[list([{'secondary_y': True}]*track)]
                        )

    fig.add_trace(
        go.Scatter(x=df[feat[0]], y=index, name='GR',
                   line_color='green'), row=1, col=1)
    fig.update_xaxes(range=[0, 150], row=1, col=1)

    fig.add_trace(
        go.Scatter(x=df[feat[1]], y=index, name='RESISTIVITY',
                   line_color='red'), row=1, col=2)
    fig.update_xaxes(range=[np.log10(.2), np.log10(200)],
                     type='log', row=1, col=2)

    fig.add_trace(
        go.Scatter(x=df[feat[2]], y=index, name='DENSITY',  # fill='tonextx',
                   line_color='blue', ), row=1, col=3, secondary_y=False)
    fig.update_xaxes(range=[1.85, 2.85], row=1, col=3)

    fig.add_trace(
        go.Scatter(x=df[pred], y=index, name=pred, line_color='orange',
                   fill='tozerox'), row=1, col=4)
    fig.update_xaxes(range=[0, np.max(df[pred])], row=1, col=4)

    fig.add_trace(
        go.Scatter(x=df[feat[3]], y=index, name='NEUTRON',
                   line_color='red'), row=1, col=3, secondary_y=True)
    fig.update_xaxes(range=[.45, -.15], row=1, col=3)
    fig.data[4].update(xaxis='x5')

    fig.update_layout(
        xaxis1=dict(title="GR", titlefont=dict(color="green"), tickfont=dict(color="green"),
                    side='top', anchor='free', position=.9, title_standoff=.1),
        xaxis2=dict(title="RT", titlefont=dict(color="red"), tickfont=dict(color="red"),
                    side='top', anchor='free', position=0.9, title_standoff=.1),
        xaxis3=dict(title="RHOB", titlefont=dict(color="blue"), tickfont=dict(color="blue"),
                    side='top', anchor='free', position=.9, title_standoff=.1),
        xaxis4=dict(title=pred, titlefont=dict(color="orange"), tickfont=dict(color="orange"),
                    side='top', anchor='free', position=.9, title_standoff=.1),
        xaxis5=dict(title="NPHI", titlefont=dict(color="red"), tickfont=dict(color="red"),
                    side='top', anchor='free', position=.97, title_standoff=.1, overlaying='x3'),
    )

    # make room to display double x-axes
    fig.update_layout(
        yaxis=dict(domain=[0, .9]),
        yaxis2=dict(domain=[0, .9]),
        yaxis3=dict(domain=[0, .9]),
        yaxis4=dict(domain=[0, .9]),
        yaxis5=dict(domain=[0, .9]),
        yaxis6=dict(domain=[0, .9], visible=False),
        yaxis7=dict(domain=[0, .9]),
        yaxis8=dict(domain=[0, .9]),
    )

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(matches='y', constrain='domain', autorange="reversed")
    fig.update_layout(height=900, width=900, margin=dict(t=25),
                      title={
        'text': "%s Logs" % df.WELLNAME.unique()[0],
        'y': 1,
        'x': 0,
        'xanchor': 'left',
        'yanchor': 'top',
    }
    )
    return(fig)
