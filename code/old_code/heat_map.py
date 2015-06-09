import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np

py.sign_in("mqwilber", "wr5qhbw15p")
a_vals = np.loadtxt("a_vals.txt")
b_vals = np.loadtxt("b_vals.txt")
mort_diff = np.loadtxt("mort_diff.txt") * 100

#a_vals[a_vals > 10] = 10
#b_vals[b_vals < -6] = -6
bdiff = b_vals - -2
bdiff[bdiff < -5] = -5
adiff = a_vals - 6
adiff[adiff > 10] = 10

mort_diff[mort_diff > 2] = 2
mort_diff[mort_diff < -2] = -2
#mort_ratio[1 / mort_ratio > 5] = 5


data = Data([
    Heatmap(
        z = adiff,
        x = np.loadtxt("trun_vals.txt"),
        y = np.loadtxt("ks.txt"),
        colorscale='Jet'
    )
])

myx = XAxis(title='Truncation Value (% of LD50)')
myy = YAxis(title='k values')

layout = Layout(
    title='Power Analysis for a-values',
    xaxis=myx,
    yaxis=myy,
)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='a_vals')

data = Data([
    Heatmap(
        z = bdiff,
        x = np.loadtxt("trun_vals.txt"),
        y = np.loadtxt("ks.txt"),
        colorscale='Jet'
    )
])

myx = XAxis(title='Truncation value (% of LD50)')
myy = YAxis(title='k value')

layout = Layout(
    title='Power Analysis for b-values',
    xaxis=myx,
    yaxis=myy,
)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='b_vals')

# data = Data([
#     Heatmap(
#         z = mort_diff,
#         x = np.loadtxt("trun_vals.txt"),
#         y = np.loadtxt("ks.txt"),
#         colorscale='Jet'
#     )
# ])

# myx = XAxis(title='Truncation value (% of LD50)')
# myy = YAxis(title='k value')

# layout = Layout(
#     title='Power Analysis for Mortality Difference * 100',
#     xaxis=myx,
#     yaxis=myy,
# )

# fig = Figure(data=data, layout=layout)
# plot_url = py.plot(fig, filename='mort_diff')