import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams.update({
                    'axes.labelsize': 12,
                    'xtick.labelsize': 12,
                    'legend.fontsize': 'small'
                    })


def plotly_log(df, cols, unit='imperial'):
    track = 3
    index = df.DEPTH

    fig = make_subplots(rows=1, cols=track, shared_yaxes=True, horizontal_spacing=0.05,
                        specs=[list([{'secondary_y': True}] * track)])

    fig.add_trace(
        go.Scatter(x=df[cols[0]], y=index, name='GR',
                   line_color='green'), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=df[cols[1]], y=index, name='RESISTIVITY',
                   line_color='red'), row=1, col=2)

    fig.add_trace(
        go.Scatter(x=df[cols[2]], y=index, name='DENSITY',
                   line_color='blue', fill='tonextx', fillcolor='rgba(0,0,255,.1)'), row=1, col=3, secondary_y=False)

    fig.add_trace(
        go.Scatter(x=df[cols[3]], y=index, name='NEUTRON',
                   line_color='red', fill='tonextx', fillcolor='rgba(255,0,0,.1)'), row=1, col=3, secondary_y=True)
    fig.data[3].update(xaxis='x4')

    fig.update_layout(
        xaxis1=dict(title="GR", titlefont=dict(color="green"), tickfont=dict(color="green"),
                    side='top', anchor='free', position=.9, title_standoff=.1, range=[0, 150]),

        xaxis2=dict(title="RT", titlefont=dict(color="red"), tickfont=dict(color="red"),
                    side='top', anchor='free', position=.9, title_standoff=.1,
                    range=[np.log10(.2), np.log10(200)], type='log'),

        xaxis3=dict(title="RHOB", titlefont=dict(color="blue"), tickfont=dict(color="blue"),
                    side='top', anchor='free', position=.9, title_standoff=.1,
                    range=[1.85, 2.85]),

        xaxis4=dict(title="NPHI", titlefont=dict(color="red"), tickfont=dict(color="red"),
                    side='top', anchor='free', position=.97, title_standoff=.1, overlaying='x3',
                    range=[.45, -.15]),

        # make room to display double x-axes
        yaxis=dict(domain=[0, .9], title='DEPTH '+'(ft MD)' if unit == 'imperial' else '(m MD)'),
        yaxis2=dict(domain=[0, .9]),
        yaxis3=dict(domain=[0, .9], visible=False),
        yaxis4=dict(domain=[0, .9], visible=False),
        yaxis5=dict(domain=[0, .9], visible=False),
        yaxis6=dict(domain=[0, .9], visible=False),

        height=900,
        width=800,
        margin=dict(t=20),
        showlegend=False,
        title={
            'text': "%s Logs" % df.WELLNAME.unique()[0],
            'y': .99,
            'x': 0.09,
            'xanchor': 'left',
            'yanchor': 'top',
        }
    )

    fig.update_xaxes(
        fixedrange=True
    )
    fig.update_yaxes(matches='y', constrain='domain', autorange="reversed")

    st.write(fig)
