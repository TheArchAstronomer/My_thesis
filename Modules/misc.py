import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

def plot_OD_gaussian(x, y, bins, sigma, xaxis, yaxis): # x coord, y coord, nr of bins, extent of plot, sigma for gaussian filter, title of plot, x axis title, y axis title
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    fig,ax = plt.subplots(figsize=(12,8), facecolor='white')
    
    def overdensity(x, y, bins):  # generating the overdensity map
        pre_OD, xedges, yedges = np.histogram2d(x, y, bins)
        OD = (pre_OD / np.mean(pre_OD)-1)
        return OD, xedges, yedges
    
    OD, xedges, yedges = overdensity(x, y, bins) # calling out the function
    im = ax.imshow(OD.T, origin='lower', cmap = "seismic")
    c = plt.colorbar(im, ax=ax)
    c.ax.tick_params(labelsize=14)
    plt.xlabel(str(xaxis), size=16)
    plt.ylabel(str(yaxis), size=16)
    ax.tick_params(axis='both', labelsize=14)
    c.set_label('Overdensity', labelpad=20, size=16)
    hist_smoothed = gaussian_filter(OD.T, sigma=sigma)
    image = plt.imshow(hist_smoothed, origin='lower', extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="seismic")
    return image


def plot_OD_gaussian_interactive(x, y, bins, sigma, xaxis, yaxis):
    # Reuse the same overdensity logic
    pre_OD, xedges, yedges = np.histogram2d(x, y, bins)
    OD = (pre_OD / np.mean(pre_OD)) - 1
    hist_smoothed = gaussian_filter(OD.T, sigma=sigma)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    fig = go.Figure(data=go.Heatmap(
        z=hist_smoothed,
        x=xcenters,
        y=ycenters,
        colorscale='seismic',
        colorbar=dict(title='Overdensity'),
        hovertemplate=(
            f'{xaxis}: %{{x:.3f}}<br>'
            f'{yaxis}: %{{y:.3f}}<br>'
            f'OD: %{{z:.3f}}<extra></extra>'
        )
    ))

    fig.update_layout(
        xaxis_title=str(xaxis),
        yaxis_title=str(yaxis),
        font=dict(family='serif', size=14),
        width=800, height=600
    )

    fig.show()
    return fig