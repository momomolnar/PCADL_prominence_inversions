# General (computations):
import numpy as np
import scipy as sp
# Plotting
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density  # adds projection='scatter_density'


# Absolute errors
def absolute_errors_scalar(v1, v2):
    return np.sqrt((v1 - v2) ** 2)


# Relative errors
def relative_errors_scalar(v1, v2):
    return np.sqrt((v1 - v2) ** 2) / np.sqrt(v1 ** 2 + 1.e-12)


# Absolute errors
def absolute_errors_vector(v1_x, v1_y, v2_x, v2_y):
    return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2)


# Relative errors
def relative_errors_vector(v1_x, v1_y, v2_x, v2_y):
    return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2) / np.sqrt(v1_x ** 2 + v1_y ** 2)


# Cosine similatiry
def cosine_similarity_vector(v1_x, v1_y, v2_x, v2_y):
    return (v2_x * v1_x + v2_y * v1_y) / (np.sqrt(v2_x ** 2 + v2_y ** 2) * np.sqrt(v1_x ** 2 + v1_y ** 2))


def img_minmax(img, img_coord=None, img_shape=None):
    # Default values
    if img_coord is None:
        img_coord = (0, 0)
    if img_shape is None:
        img_shape = img.shape

    # Compute min/max
    min_img = np.nanmin(img[img_coord[0]:img_coord[0] + img_shape[0], img_coord[1]:img_coord[1] + img_shape[1]])
    max_img = np.nanmax(img[img_coord[0]:img_coord[0] + img_shape[0], img_coord[1]:img_coord[1] + img_shape[1]])

    return min_img, max_img


def colorbar_minmax(img, img_coord=None, img_shape=None):
    # Default values
    if img_coord is None:
        img_coord = (0, 0)
    if img_shape is None:
        img_shape = img.shape

    # Compute min/max
    min_img, max_img = img_minmax(
        img[img_coord[0]:img_coord[0] + img_shape[0], img_coord[1]:img_coord[1] + img_shape[1]])

    # Adjust min/max to be symetrical if diverging
    if min_img * max_img < 0:
        return -np.nanmax([np.abs(min_img), np.abs(max_img)]), np.nanmax([np.abs(min_img), np.abs(max_img)])
    return min_img, max_img


def make_list(var, ndim=0):
    if np.ndim(var) == ndim: var = [var]
    if np.ndim(var) == ndim + 1: var = [var]
    return var


def make_tuple(var, ndim=0):
    if np.ndim(var) == ndim: var = (var)
    if np.ndim(var) == ndim + 1: var = (var)
    return var


def plot_bars(bar, bar_axscale=None,
              bar_xticks=None, bar_yticks=None, bar_labels=None, bar_title=None,
              bar_labelspad=(5, 3), bar_tickw=1, bar_tickl=2.5, bar_width=0.75,
              bar_tickdir='out', bar_titlepad=1.005, bar_color=None, bar_legend=None,
              legend_loc=None, legend_font=10, legend_ncol=1,
              legend_npoints=1, legend_scale=4.0, legend_spacing=0.2,
              fig_filename=None, fig_show=False, fig_format='png', fig_dpi=300,
              fig_transparent=False, fig_lx=4.0, fig_ly=4.0, fig_ratio=None, fig_font=12,
              fig_left=0.8, fig_right=0.8, fig_bottom=0.48, fig_top=0.32,
              fig_wspace=0.0, fig_hspace=0.0, fig_grid=False, fig_gridlinew=0.5):
    # Make lists/tuples
    bar = make_list(bar, ndim=2)
    if bar_xticks is not None: bar_xticks = make_list(bar_xticks, ndim=1)
    if bar_yticks is not None: bar_yticks = make_list(bar_yticks)
    if bar_axscale is not None: bar_axscale = make_list(bar_axscale)
    if bar_labels is not None: bar_labels = make_list(bar_labels, ndim=1)
    if bar_title is not None: bar_title = make_list(bar_title)
    if bar_color is not None: bar_color = make_list(bar_color)
    if bar_legend is not None: bar_legend = make_list(bar_legend, ndim=1)
    if legend_loc is not None: legend_loc = make_list(legend_loc)

    # Dimensions
    nrows, ncols, nb_xticks, nb_bars = np.shape(bar)

    # Set up default lists and values
    # if bar_yticks is None: bar_yticks = [[1, ] * ncols, ] * nrows
    if bar_axscale is None: bar_axscale = [['linear', ] * ncols, ] * nrows
    if bar_labels is None: bar_labels = [[('y-axis', 'x-axis'), ] * ncols, ] * nrows
    if legend_loc is None: legend_loc = [['best', ] * ncols, ] * nrows
    if fig_ratio is None:
        tmp_lx = nb_xticks * (bar_width + bar_width / nb_bars)
        if tmp_lx > fig_ly:
            fig_ratio = fig_ly / tmp_lx
        else:
            fig_ratio = 1

    # (fig_ly - bar_width*3)

    # Layout
    font_size = fig_font
    fig_sizex = 0.
    fig_sizey = 0.
    fig_specx = np.zeros((nrows, ncols + 1)).tolist()
    fig_specy = np.zeros((nrows + 1, ncols)).tolist()
    for row in range(nrows):
        for col in range(ncols):
            if fig_ratio > 1:
                fig_specy[row + 1][col] = fig_specy[row][col] + fig_bottom + fig_ly * fig_ratio + fig_top
                fig_specx[row][col + 1] = fig_specx[row][col] + fig_left + fig_lx + fig_right
            else:
                fig_specy[row + 1][col] = fig_specy[row][col] + fig_bottom + fig_ly + fig_top
                fig_specx[row][col + 1] = fig_specx[row][col] + fig_left + fig_lx / fig_ratio + fig_right
            if row == 0 and col == ncols - 1: fig_sizex = fig_specx[row][col + 1]
            if row == nrows - 1 and col == 0: fig_sizey = fig_specy[row + 1][col]
    fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout=False)

    # Rows
    for row in tqdm(range(nrows)):
        # Cols
        for col in tqdm(range(ncols)):

            # Adjust subplot layout
            spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left / fig_sizex + fig_specx[row][col] / fig_sizex,
                                    right=-fig_right / fig_sizex + fig_specx[row][col + 1] / fig_sizex,
                                    bottom=fig_bottom / fig_sizey + fig_specy[(nrows - 1) - row][col] / fig_sizey,
                                    top=-fig_top / fig_sizey + fig_specy[(nrows - 1) - row + 1][col] / fig_sizey,
                                    wspace=fig_wspace, hspace=fig_hspace)

            # Colors
            if bar_color is None:
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Indices fox x-axis
            nb_xticks = len(bar[row][col])
            nb_bars = len(bar[row][col][0])
            ind = np.arange(nb_xticks)

            # Plot scatter
            ax = fig.add_subplot(spec[:, :])
            for b in range(nb_bars):
                for x in range(nb_xticks):
                    if x == 0:
                        plt.bar(x + b * bar_width / nb_bars, bar[row][col][x][b], bar_width / nb_bars,
                                label=bar_legend[row][col][b], color=colors[b])
                    else:
                        plt.bar(x + b * bar_width / nb_bars, bar[row][col][x][b], bar_width / nb_bars, color=colors[b])
            ax.set_yscale(bar_axscale[row][col])

            # Grid
            if fig_grid:
                ax.grid(fig_grid, linewidth=fig_gridlinew)
            # Title
            if bar_title is not None: ax.set_title(bar_title[row][col], fontsize=font_size, y=bar_titlepad, wrap=True)
            # x/y-axis layout
            ax.get_yaxis().set_tick_params(which='both', direction=bar_tickdir, width=bar_tickw, length=bar_tickl,
                                           labelsize=font_size,
                                           left=True, right=False)
            ax.get_xaxis().set_tick_params(which='both', direction=bar_tickdir, width=bar_tickw, length=bar_tickl,
                                           labelsize=font_size,
                                           bottom=True, top=False)
            ax.set_ylabel(bar_labels[row][col][0], fontsize=font_size, labelpad=bar_labelspad[0])
            ax.set_xlabel(bar_labels[row][col][1], fontsize=font_size, labelpad=bar_labelspad[1])
            # Ticks
            if bar_yticks is not None:
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(bar_yticks[row][col]))
            if bar_xticks is not None:
                # x-axis
                plt.xticks(ind + 0.5 * (nb_bars - 1) * bar_width / nb_bars, bar_xticks[row][col])
            else:
                plt.xticks(ind + 0.5 * (nb_bars - 1) * bar_width / nb_bars, np.arange(nb_xticks))

            # Legend
            ax.legend(loc=legend_loc[row][col], fontsize=legend_font, numpoints=legend_npoints,
                      markerscale=legend_scale,
                      labelspacing=legend_spacing, ncol=legend_ncol, fancybox=False)

    # Show and Save
    if fig_filename is not None: plt.savefig(fig_filename, format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
    if fig_show is False:
        plt.close('all')
    else:
        plt.draw()


def plot_scatter(reference, inference,
                 img_coord=None, img_shape=None,
                 scat_alpha=None, scat_ticks=None, scat_labels=None, scat_title=None,
                 scat_labelspad=(5, 3), scat_tickw=1, scat_tickl=2.5,
                 scat_tickdir='out', scat_titlepad=1.005, scat_proj=None,
                 scat_marker='.', scat_markersize=0.9, scat_color=None,
                 scat_grid=True, scat_gridlinew=0.5, scat_axscale=None,
                 ref_label='Reference (1:1)', ref_color=None, ref_linew=0.5, ref_lines='--',
                 fit=None, fit_color=None, fit_linew=0.25, fit_lines='-',
                 legend_loc=None, legend_font=10, legend_ncol=1,
                 legend_npoints=1, legend_scale=4.0, legend_spacing=0.05,
                 cb_label=None, cb_minmax=None, cb_cmap=None,
                 cb_pad=0, cb_tickw=1, cb_tickl=2.5, cb_font=None,
                 cb_dir='out', cb_rot=270, cb_labelpad=16, cb_side='right',
                 fig_filename=None, fig_show=False, fig_format='png', fig_dpi=300,
                 fig_transparent=False, fig_lx=4.0, fig_ly=4.0, fig_lcb=5, fig_font=12,
                 fig_left=0.8, fig_right=0.8, fig_bottom=0.48, fig_top=0.32,
                 fig_wspace=0.0, fig_hspace=0.0):
    # Make lists/tuples
    reference = make_list(reference, ndim=2)
    inference = make_list(inference, ndim=2)
    if img_coord is not None: img_coord = make_list(img_coord, ndim=1)
    if img_shape is not None: img_shape = make_list(img_shape, ndim=1)
    if scat_alpha is not None: scat_alpha = make_list(scat_alpha)
    if scat_axscale is not None: scat_axscale = make_list(scat_axscale, ndim=1)
    if scat_ticks is not None: scat_ticks = make_list(scat_ticks, ndim=1)
    if scat_labels is not None: scat_labels = make_list(scat_labels, ndim=1)
    if scat_proj is not None: scat_proj = make_list(scat_proj)
    if scat_title is not None: scat_title = make_list(scat_title)
    if fit is not None: fit = make_list(fit)
    if legend_loc is not None: legend_loc = make_list(legend_loc)
    if cb_label is not None: cb_label = make_list(cb_label)
    if cb_minmax is not None: cb_minmax = make_list(cb_minmax)
    if cb_cmap is not None: cb_cmap = make_list(cb_cmap)

    # Dimensions
    nrows, ncols, _, _ = np.shape(reference)

    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    # "Viridis-like" colormap with white background
    if cb_cmap is None:
        white_viridis = LinearSegmentedColormap.from_list('white_viridis',
                                                          [(0, '#ffffff'),
                                                           (1e-20, '#440053'),
                                                           (0.2, '#404388'),
                                                           (0.4, '#2a788e'),
                                                           (0.6, '#21a784'),
                                                           (0.8, '#78d151'),
                                                           (1, '#fde624'),
                                                           ], N=256)

    # Set up default lists and values
    if img_coord is None: img_coord = [[(0, 0), ] * ncols, ] * nrows
    if img_shape is None: img_shape = [[reference[row][col].shape for col in range(ncols)] for row in range(nrows)]
    if scat_alpha is None: scat_alpha = [[1.0, ] * ncols, ] * nrows
    if scat_ticks is None: scat_ticks = [[(1, 1), ] * ncols, ] * nrows
    if scat_proj is None: scat_proj = [[None, ] * ncols, ] * nrows
    if scat_labels is None: scat_labels = [[('Inference', 'Reference'), ] * ncols, ] * nrows
    if scat_axscale is None: scat_axscale = [[('linear', 'linear'), ] * ncols, ] * nrows
    if scat_color is None: scat_color = colors[0]
    if ref_color is None: ref_color = 'black'
    if fit_color is None: fit_color = colors[1]
    if fit is None: fit = [[True, ] * ncols, ] * nrows
    if legend_loc is None: legend_loc = [['best', ] * ncols, ] * nrows
    if cb_font is None: cb_font = fig_font

    # Layout
    font_size = fig_font
    fig_sizex = 0.
    fig_sizey = 0.
    fig_specx = np.zeros((nrows, ncols + 1)).tolist()
    fig_specy = np.zeros((nrows + 1, ncols)).tolist()
    for row in range(nrows):
        for col in range(ncols):
            fig_specy[row + 1][col] = fig_specy[row][col] + fig_bottom + fig_ly + fig_top
            fig_specx[row][col + 1] = fig_specx[row][col] + fig_left + fig_lx + fig_right
            if row == 0 and col == ncols - 1: fig_sizex = fig_specx[row][col + 1]
            if row == nrows - 1 and col == 0: fig_sizey = fig_specy[row + 1][col]
    fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout=False)

    # Rows
    for row in tqdm(range(nrows)):
        # Cols
        for col in tqdm(range(ncols)):

            # Adjust subplot layout
            spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left / fig_sizex + fig_specx[row][col] / fig_sizex,
                                    right=-fig_right / fig_sizex + fig_specx[row][col + 1] / fig_sizex,
                                    bottom=fig_bottom / fig_sizey + fig_specy[(nrows - 1) - row][col] / fig_sizey,
                                    top=-fig_top / fig_sizey + fig_specy[(nrows - 1) - row + 1][col] / fig_sizey,
                                    wspace=fig_wspace, hspace=fig_hspace)

            # Metrics
            x_val = reference[row][col].copy().flatten()
            y_val = inference[row][col].copy().flatten()
            x_min, x_max = np.nanmin(x_val), np.nanmax(x_val)
            y_min, y_max = np.nanmin(y_val), np.nanmax(y_val)
            xy_range = [np.nanmin([x_min, y_min]), np.nanmax([x_max, y_max])]

            # Plot scatter
            if scat_proj[row][col] == 'scatter_density':
                ax = fig.add_subplot(spec[:, :], projection=scat_proj[row][col])
                I = ax.scatter_density(x_val, y_val, cmap=white_viridis)
            else:
                ax = fig.add_subplot(spec[:, :])
                I = ax.scatter(x_val, y_val, c=scat_color, alpha=scat_alpha[row][col], marker=scat_marker,
                               s=scat_markersize)
            # Reference
            ax.set_aspect(1)
            ax.plot(xy_range, xy_range, label=ref_label, color=ref_color, linewidth=ref_linew, linestyle=ref_lines)
            # Fit
            if fit[row][col] is True:
                # Fit
                slope, origin = np.polyfit(x_val.flatten(), y_val.flatten(), 1)
                if origin >= 0:
                    fit_label = r'y = {0:.3f}x + {1:.3f}'.format(slope, origin)
                else:
                    fit_label = r'y = {0:.3f}x - {1:.3f}'.format(slope, np.abs(origin))
                ax.plot(xy_range, [xy_range[0] * slope + origin, xy_range[1] * slope + origin],
                        label=fit_label, color=fit_color, linewidth=fit_linew, linestyle=fit_lines,
                        xscale=scat_axscale[row][col][1], yscale=scat_axscale[row][col][0])

            # Grid
            ax.grid(scat_grid, linewidth=scat_gridlinew)
            ax.set_xlim(xy_range)
            ax.set_ylim(xy_range)
            ax.set_xscale(scat_axscale[row][col][1])
            ax.set_yscale(scat_axscale[row][col][0])

            # Title
            if scat_title is not None: ax.set_title(scat_title[row][col], fontsize=font_size, y=scat_titlepad,
                                                    wrap=True)
            # x/y-axis layout
            ax.get_yaxis().set_tick_params(which='both', direction=scat_tickdir, width=scat_tickw, length=scat_tickl,
                                           labelsize=font_size,
                                           left=True, right=True)
            ax.get_xaxis().set_tick_params(which='both', direction=scat_tickdir, width=scat_tickw, length=scat_tickl,
                                           labelsize=font_size,
                                           bottom=True, top=True)
            if scat_labels is None:
                ax.set_ylabel('Inference', fontsize=font_size, labelpad=scat_labelspad[0])
                ax.set_xlabel('Reference', fontsize=font_size, labelpad=scat_labelspad[1])
            else:
                ax.set_ylabel(scat_labels[row][col][0], fontsize=font_size, labelpad=scat_labelspad[0])
                ax.set_xlabel(scat_labels[row][col][1], fontsize=font_size, labelpad=scat_labelspad[1])
            # Number of ticks
            if scat_ticks[row][col] is not None:
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(scat_ticks[row][col][0]))
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(scat_ticks[row][col][1]))

            # Legend
            plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(sp.stats.spearmanr(x_val, y_val)[0]))
            plt.plot([], [], ' ', label=r"MAE: {0:.3f}".format(np.nanmean(absolute_errors_scalar(x_val, y_val))))
            plt.plot([], [], ' ',
                     label=r"MAPE: {0:.3f}%".format(100. * np.nanmedian(relative_errors_scalar(x_val, y_val))))
            ax.legend(loc=legend_loc[row][col], fontsize=legend_font, numpoints=legend_npoints,
                      markerscale=legend_scale,
                      labelspacing=legend_spacing, ncol=legend_ncol, fancybox=False)

            # Colorbar
            divider = make_axes_locatable(ax)
            if scat_proj[row][col] == 'scatter_density':
                cax = divider.append_axes(cb_side, size="{0}%".format(fig_lcb * fig_lx * ncols / fig_sizex), pad=cb_pad)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction=cb_dir, labelsize=cb_font, width=cb_tickw, length=cb_tickl)
                if cb_label is not None: cb.set_label(cb_label[row][col], labelpad=cb_labelpad, rotation=cb_rot,
                                                      size=cb_font)

    # Show and Save
    if fig_filename is not None: plt.savefig(fig_filename, format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
    if fig_show is False:
        plt.close('all')
    else:
        plt.draw()