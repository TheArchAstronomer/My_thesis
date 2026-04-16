import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize, PowerNorm
from matplotlib import colors
#import plotly.graph_objects as go

def overdensity(x, y, bins):  # generating the overdensity map
        pre_OD, xedges, yedges = np.histogram2d(x, y, bins)
        OD = ((pre_OD / np.mean(pre_OD))-1)
        return OD, xedges, yedges

def plot_OD_gaussian(x, y, bins, sigma, xaxis, yaxis): # x coord, y coord, nr of bins, extent of plot, sigma for gaussian filter, title of plot, x axis title, y axis title
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    fig,ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    def overdensity(x, y, bins):  # generating the overdensity map
        pre_OD, xedges, yedges = np.histogram2d(x, y, bins)
        OD = (pre_OD / np.mean(pre_OD)-1)
        return OD, xedges, yedges
    
    OD, xedges, yedges = overdensity(x, y, bins) # calling out the function
    hist_smoothed = gaussian_filter(OD.T, sigma=sigma)
    im = ax.imshow(hist_smoothed, origin='upper', cmap = "seismic", 
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   norm=Normalize(vmin=-np.max(np.abs(hist_smoothed)), vmax=1.0, clip=False))
    ax.scatter(0,0, color='white', marker='x', s=150, linewidths=3,label='LMC')
    c = plt.colorbar(im, ax=ax)
    c.ax.tick_params(labelsize=30)
    plt.xlabel(str(xaxis), size=30)
    plt.ylabel(str(yaxis), size=30)
    ax.tick_params(axis='both', labelsize=30)
    ax.tick_params(which='major', length=10, width=2, direction='in', color='black')
    ax.set_xticks(np.arange(xedges[0], xedges[-1], 50), minor=True)
    ax.set_yticks(np.arange(yedges[0], yedges[-1], 20), minor=True)
    ax.tick_params(which='minor', length=5, width=1.5, direction='in', color='black',
                    labelbottom=False, labelleft=False)
    c.set_label(r'$\delta \rho$', labelpad=20, size=30)
    half_max = hist_smoothed.max() / 2
    OD_half_max = np.max(OD) / 2
    print(f"Half max overdensity: {half_max:.4f}")
    print(f"OD half max overdensity: {np.max(OD)/2:.4f}")
    # ax.contour(hist_smoothed, levels=[half_max], colors='white', linewidths=1.5, 
    #            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='upper')
    ax.contour(hist_smoothed, levels=[OD_half_max], colors='white', linewidths=1.5, 
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='upper')
    ax.text(-225, -250, f'Contour Level = {OD_half_max:.2f}', fontsize=30, color='white')
    return im


def plot_OD_gaussian_foote(x, y, bins, sigma, xaxis, yaxis): # x coord, y coord, nr of bins, extent of plot, sigma for gaussian filter, title of plot, x axis title, y axis title
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    fig,ax = plt.subplots(figsize=(12,8), facecolor='white')
    
    def overdensity(x, y, bins):  # generating the overdensity map
        pre_OD, xedges, yedges = np.histogram2d(x, y, bins)
        OD = (pre_OD / np.mean(pre_OD)-1)
        return OD, xedges, yedges
    
    OD, xedges, yedges = overdensity(x, y, bins) # calling out the function
    im = ax.imshow(OD.T, origin='upper', cmap = "viridis")
    c = plt.colorbar(im, ax=ax)
    c.ax.tick_params(labelsize=14)
    plt.xlabel(str(xaxis), size=16)
    plt.ylabel(str(yaxis), size=16)
    ax.tick_params(axis='both', labelsize=14)
    c.set_label('Overdensity', labelpad=20, size=16)
    hist_smoothed = gaussian_filter(OD.T, sigma=sigma)
    image = plt.imshow(hist_smoothed, origin='lower', extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis")
    half_max = hist_smoothed.max() / 2
    ax.contour(hist_smoothed, levels=[half_max], colors='white', linewidths=1.5, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='upper')
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
        colorscale='RdBu_r',
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
        width=800, height=800
    )

    fig.show()
    return fig


def plot_gaussian_interactive(x, y, bins, xaxis, yaxis):
    # Reuse the same overdensity logic
    counts, xedges, yedges = np.histogram2d(x, y, bins)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    fig = go.Figure(data=go.Heatmap(
        z=counts.T,
        x=xcenters,
        y=ycenters,
        colorscale='RdBu_r',
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
        width=800, height=800
    )

    fig.show()
    return fig


def resultant_OD_gaussian(data1, data2, bins, sigma, data1_label, data2_label):
    
    def overdensity(x, y, bins):  # generating the overdensity map
        pre_OD, xedges, yedges = np.histogram2d(x, y, bins)
        OD = ((pre_OD / np.mean(pre_OD))-1)
        return OD, xedges, yedges
    
    x1, y1, z1 = data1[:, 0], data1[:, 1], data1[:, 2]
    x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]

    OD1, xedges1, yedges1 = overdensity(x1, y1, bins)
    OD2, xedges2, yedges2 = overdensity(x2, y2, bins)

    hist_smoothed1 = gaussian_filter(OD1.T, sigma=sigma)
    hist_smoothed2 = gaussian_filter(OD2.T, sigma=sigma)

    resultant_OD = hist_smoothed1 - hist_smoothed2

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    im1 = axs[0].imshow(hist_smoothed1, origin='upper', cmap="viridis", 
                        extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]],
                          norm=Normalize(vmin=np.min(hist_smoothed1), vmax=np.max(hist_smoothed1),
                                          clip=False))
    im2 = axs[1].imshow(hist_smoothed2, origin='upper', cmap="viridis", 
                        extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]],
                          norm=Normalize(vmin=np.min(hist_smoothed2), vmax=np.max(hist_smoothed2),
                                          clip=False))
    im3 = axs[2].imshow(resultant_OD, origin='upper', cmap="viridis", 
                        extent=[-300, 100, -100, 100],
                          norm=Normalize(vmin=np.min(resultant_OD), vmax=np.max(resultant_OD), clip=False))
    c = plt.colorbar(im1, ax=axs[0])
    c = plt.colorbar(im2, ax=axs[1])
    c = plt.colorbar(im3, ax=axs[2])
    axs[0].set_title(f'{data1_label} Overdensity', size=16)
    axs[1].set_title(f'{data2_label} Overdensity', size=16)
    axs[2].set_title('Resultant Overdensity', size=16)

    return fig, axs


# def resultant_OD_gaussian(data1, data2, bins, sigma, data1_label, data2_label):
    
#     def overdensity(x, y, bins):  # generating the overdensity map
#         pre_OD, xedges, yedges = np.histogram2d(x, y, bins)
#         OD = ((pre_OD / np.mean(pre_OD))-1)
#         return OD, xedges, yedges
    
#     x1, y1, z1 = data1[:, 0], data1[:, 1], data1[:, 2]
#     x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]

#     OD1, xedges1, yedges1 = overdensity(x1, y1, bins)
#     OD2, xedges2, yedges2 = overdensity(x2, y2, bins)

#     hist_smoothed1 = gaussian_filter(OD1.T, sigma=sigma)
#     hist_smoothed2 = gaussian_filter(OD2.T, sigma=sigma)

#     resultant_OD = hist_smoothed1 - hist_smoothed2

#     fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
#     im1 = axs[0].imshow(hist_smoothed1, origin='upper', cmap="seismic", 
#                         extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]],
#                           norm=Normalize(vmin=-np.max(np.abs(hist_smoothed1)), vmax=np.max(np.abs(hist_smoothed1)),
#                                           clip=False))
#     im2 = axs[1].imshow(hist_smoothed2, origin='upper', cmap="seismic", 
#                         extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]],
#                           norm=Normalize(vmin=-np.max(np.abs(hist_smoothed2)), vmax=np.max(np.abs(hist_smoothed2)),
#                                           clip=False))
#     im3 = axs[2].imshow(resultant_OD, origin='upper', cmap="seismic", 
#                         extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]],
#                           norm=Normalize(vmin=-np.max(np.abs(resultant_OD)), vmax=np.max(np.abs(resultant_OD)), clip=False))
#     c = plt.colorbar(im1, ax=axs[0])
#     c = plt.colorbar(im2, ax=axs[1])
#     c = plt.colorbar(im3, ax=axs[2])
#     axs[0].set_title(f'{data1_label} Overdensity', size=16)
#     axs[1].set_title(f'{data2_label} Overdensity', size=16)
#     axs[2].set_title('Resultant Overdensity', size=16)

#     return fig, axs
