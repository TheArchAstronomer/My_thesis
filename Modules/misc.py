import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def plot_OD_gaussian(x, y, bins, sigma, title, xaxis, yaxis): # x coord, y coord, nr of bins, extent of plot, sigma for gaussian filter, title of plot, x axis title, y axis title
    fig,ax = plt.subplots()
    
    def overdensity(x, y, bins):  # generating the overdensity map
        pre_OD, xedges, yedges = np.histogram2d(x, y, bins)
        OD = (pre_OD / np.mean(pre_OD)-1)
        return OD, xedges, yedges
    
    OD, xedges, yedges = overdensity(x, y, bins) # calling out the function
    im = ax.imshow(OD.T, origin='lower', cmap = "seismic")
    c = plt.colorbar(im, ax=ax)
    plt.title(str(title))
    plt.xlabel(str(xaxis))
    plt.ylabel(str(yaxis))
    c.set_label('difference from average star density', rotation=270, labelpad=20)
    hist_smoothed = gaussian_filter(OD.T, sigma=sigma)
    image = plt.imshow(hist_smoothed, origin='lower', extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="seismic")
    return image