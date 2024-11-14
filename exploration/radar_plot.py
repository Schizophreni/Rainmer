"""
plot radar properties refers to https://www.matplotlib.org.cn/gallery/specialty_plots/radar_chart.html
"""

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import numpy as np


pro2label = {
    "gt_entropies": "Total GT entropy",
    "rain_entropies": "Total rain entropy",
    "densities": "Average rain density",
    "y_gradients": "Average Y_grad",
    "non_monos": "Average non-monochromatic",
    "gt_illus": "Average GT illuminance",
    "rain_illus": "Average rain illuminance",
    "psnrs": "Average PSNR(rain, GT)"
}


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)
        
        def fill_between(self, *args, closed=True, **kwargs):
            return super().fill_between(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def my_data():
    # obtain properties for different datasets
    # load data
    synrain_prop = np.load("properties/SynRain_prop.npz", allow_pickle=True)["arr_0"].item()
    gtrain_prop = np.load("properties/GT-Rain_prop.npz", allow_pickle=True)["arr_0"].item()
    gtav_prop = np.load("properties/GTAV-balance_prop.npz", allow_pickle=True)["arr_0"].item()
    keys = list(synrain_prop.keys())
    synrain_vals, gtrain_vals, gtav_vals = [[], []], [[], []], [[], []]
    for k in keys:
        synrain_k = synrain_prop[k]
        gtrain_k = gtrain_prop[k]
        gtav_k = gtav_prop[k]
        if k in ["gt_entropies", "rain_entropies"]:
            synrain_k = synrain_k.sum()
            gtrain_k = gtrain_k.sum()
            gtav_k = gtav_k.sum() 
            max_k = max(synrain_k, gtrain_k, gtav_k)
            synrain_vals[0].append(0)
            gtrain_vals[0].append(0)
            gtav_vals[0].append(0)
            synrain_vals[1].append(synrain_k / max_k)
            gtrain_vals[1].append(gtrain_k / max_k)
            gtav_vals[1].append(gtav_k / max_k)
        elif  k in ["gt_illus", "rain_illus", "y_gradients", "non_monos", "psnrs", "densities"]:
            synrain_k = synrain_k.mean()
            gtrain_k = gtrain_k.mean()
            gtav_k = gtav_k.mean() 
            max_k = max(synrain_k, gtrain_k, gtav_k)
            synrain_vals[0].append(0)
            gtrain_vals[0].append(0)
            gtav_vals[0].append(0)
            synrain_vals[1].append(synrain_k / max_k)
            gtrain_vals[1].append(gtrain_k / max_k)
            gtav_vals[1].append(gtav_k / max_k)
            print(k, gtrain_k, synrain_k)
        else:
            print((synrain_k))
            synrain_min, synrain_max = synrain_k.min(), synrain_k.max()
            gtrain_min, gtrain_max = gtrain_k.min(), gtrain_k.max()
            gtav_min, gtav_max = gtav_k.min(), gtav_k.max()
            # use max value to normalize
            max_k = max(synrain_max, gtrain_max, gtav_max)
            synrain_vals[0].append(synrain_min / max_k)
            gtrain_vals[0].append(gtrain_min / max_k)
            gtav_vals[0].append(gtav_min / max_k)
            synrain_vals[1].append(synrain_max / max_k)
            gtrain_vals[1].append(gtrain_max / max_k)
            gtav_vals[1].append(gtav_max / max_k)
    data = [
        keys,
        synrain_vals,
        gtrain_vals, 
        gtav_vals
    ]
    print(keys)
    return data

def example_data():
    # The following data is from the Denver Aerosol Sources and Health study.
    # See doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolized Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase species is present...
    data = [
        ['Sulfate', 'Nitrate', 'EC', 'OC1', 'OC2', 'OC3', 'OP', 'CO', 'O3'],
        ('Basecase', [
            [0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00],
            [0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00],
            [0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00],
            [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00],
            [0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]]),
        ('With CO', [
            [0.88, 0.02, 0.02, 0.02, 0.00, 0.05, 0.00, 0.05, 0.00],
            [0.08, 0.94, 0.04, 0.02, 0.00, 0.01, 0.12, 0.04, 0.00],
            [0.01, 0.01, 0.79, 0.10, 0.00, 0.05, 0.00, 0.31, 0.00],
            [0.00, 0.02, 0.03, 0.38, 0.31, 0.31, 0.00, 0.59, 0.00],
            [0.02, 0.02, 0.11, 0.47, 0.69, 0.58, 0.88, 0.00, 0.00]]),
        ('With O3', [
            [0.89, 0.01, 0.07, 0.00, 0.00, 0.05, 0.00, 0.00, 0.03],
            [0.07, 0.95, 0.05, 0.04, 0.00, 0.02, 0.12, 0.00, 0.00],
            [0.01, 0.02, 0.86, 0.27, 0.16, 0.19, 0.00, 0.00, 0.00],
            [0.01, 0.03, 0.00, 0.32, 0.29, 0.27, 0.00, 0.00, 0.95],
            [0.02, 0.00, 0.03, 0.37, 0.56, 0.47, 0.87, 0.00, 0.00]]),
        ('CO & O3', [
            [0.87, 0.01, 0.08, 0.00, 0.00, 0.04, 0.00, 0.00, 0.01],
            [0.09, 0.95, 0.02, 0.03, 0.00, 0.01, 0.13, 0.06, 0.00],
            [0.01, 0.02, 0.71, 0.24, 0.13, 0.16, 0.00, 0.50, 0.00],
            [0.01, 0.03, 0.00, 0.28, 0.24, 0.23, 0.00, 0.44, 0.88],
            [0.02, 0.00, 0.18, 0.45, 0.64, 0.55, 0.86, 0.00, 0.16]])
    ]
    return data


if __name__ == '__main__':
    N = 8
    theta = radar_factory(N, frame='polygon')
    theta_close = radar_factory(N, frame="polygon")
    theta_close = np.concatenate([theta_close, theta_close[[0]]], axis=0)

    # data = example_data()
    data = my_data()
    spoke_labels = data.pop(0)
    spoke_labels = [pro2label[item] for item in spoke_labels]

    fig, ax = plt.subplots(figsize=(7, 5), nrows=1, ncols=1,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'y', 'm']# 'r', 'g']
    labels = ('SynRain', 'GT-Rain', 'GTAV-NB')
    # Plot the four cases from the example data on separate axes
    for idx, (d, color) in enumerate(zip(data, colors)):
        # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])  # set radius values
        # x.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
        #              horizontalalignment='center', verticalalignment='center')
        print(theta.shape, (d[1]))
        ax.plot(theta, d[0], color=color, label=labels[idx])
        ax.plot(theta, d[1], color=color)
        ax.fill_between(theta_close, d[0]+[d[0][0]], d[1]+[d[1][0]], facecolor=color, alpha=0.2)
    ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    # ax = axes[0, 0]
    # ax.set_yticklabels([])
    legend = ax.legend(loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small')

    # fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')

    # plt.show()
    plt.savefig("radar_plot.pdf", dpi=400, pad_inches=0.1, bbox_inches="tight")