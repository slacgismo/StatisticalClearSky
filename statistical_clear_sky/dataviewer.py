# -*- coding: utf-8 -*-
"""
This module contains the a data viewer class for data set investigation.
"""
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from statistical_clear_sky.solver_type import SolverType
from statistical_clear_sky.configuration import CONFIG1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set(context='paper', style='darkgrid', palette='colorblind')
import s3fs
import logging, warnings, time, os
logging.basicConfig(filename='data_viewer.log', level=logging.INFO)

class PointBrowser(object):
    """
    See "Event Handling" example from matplotlib documentation:
    https://matplotlib.org/examples/event_handling/data_browser.html

    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'a'
    and 's' keys to browse through the next and previous points along x-axis (ordered by RdTools estimate).
    """

    def __init__(self, data, xlim=None, ylim=None, prcntl=95):
        logging.info('NEW SESSION')
        warnings.filterwarnings("ignore")
        self._scsf_cache_dir = './local_cache/'
        if not os.path.exists(self._scsf_cache_dir):
            os.makedirs(self._scsf_cache_dir)

        ordering = np.argsort(data['rd']).values
        self._data = data.iloc[ordering]
        self._xs = self._data['rd'].values
        self._ys = self._data['deg'].values

        gs = GridSpec(4, 3)
        fig = plt.figure('DataViewer', figsize=(8, 16))
        ax = [plt.subplot(gs[0, :2])]  # Main scatter plot
        with sns.axes_style('white'):
            ax.append(plt.subplot(gs[0, -1]))  # Record viewing panel
            ax[-1].set_axis_off()
            ax.append(plt.subplot(gs[1, :]))  # Timeseries heatmap view
            ax.append(plt.subplot(gs[2, :]))  # ClearSky heatmap view
        ax.append(plt.subplot(gs[3, :]))  # Daily Energy view
        self._fig = fig
        self._ax = ax

        self._ax[0].set_title('click on point to view record')
        self._ax[0].set_xlabel('RdTools Estimate YoY deg (%)')
        self._ax[0].set_ylabel('SCSF Estimate YoY deg (%)')
        self._ax[2].set_title('Measured power')
        self._ax[2].set_xlabel('Day number')
        self._ax[2].set_yticks([])
        self._ax[2].set_ylabel('(sunset)        Time of day        (sunrise)')

        self._line, = self._ax[0].plot(self._xs, self._ys, '.', picker=5)  # 5 points tolerance
        m = np.logical_and(
            np.logical_and(
                self._data['res-median'] < np.percentile(self._data['res-median'], prcntl),
                self._data['res-var'] < np.percentile(self._data['res-var'], prcntl)
            ),
            self._data['res-L0norm'] < np.percentile(self._data['res-L0norm'], prcntl)
        )
        m = np.logical_not(m.values)
        self._ax[0].plot(self._xs[m], self._ys[m], '.')
        if xlim is None:
            xlim = self._ax[0].get_xlim()
        if ylim is None:
            ylim = self._ax[0].get_ylim()
        pts = (
            min(xlim[0], ylim[0]),
            max(xlim[1], ylim[1])
        )
        self._ax[0].plot(pts, pts, ls='--', color='red')
        self._ax[0].set_xlim(xlim)
        self._ax[0].set_ylim(ylim)
        self._text = self._ax[0].text(0.05, 0.95, 'system ID: none',
                                    transform=self._ax[0].transAxes, va='top')
        self._selected, = self._ax[0].plot([self._xs[0]], [self._ys[0]], 'o',
            ms=6, alpha=0.4, color='yellow', visible=False)

        with sns.axes_style('white'):
            ax.append(plt.axes([.77, .5 * (1 + .57), .2, .05 / 2]))  # Text box entry
            ax.append(plt.axes([.82, .5 * (1 + .5), .1, .05 / 2]))  # run SCSF button
        self._text_box = TextBox(self._ax[-2], 'ID Number')
        self._button = Button(self._ax[-1], 'run SCSF', color='red')
        self._lastind = None
        self._power_signals_d = None
        self._iterative_fitting = None
        self._cb = None
        self._cb2 = None
        self._local_cash = {}
        self._prcntl = prcntl
        plt.tight_layout()

        self._fig.canvas.mpl_connect('pick_event', self._onpick)
        self._fig.canvas.mpl_connect('key_press_event', self._onpress)
        self._text_box.on_submit(self._submit)
        self._button.on_clicked(self._clicked)

        plt.show()

    def submit(self, text):
        logging.info('submit: ' + str(text))
        asrt = np.argsort(np.abs(self._data.index - float(text)))
        sysid = self._data.index[asrt[0]]
        bool_list = self._data.index == sysid
        # bool_list = self._data.index == int(text)
        index_lookup = np.arange(self._data.shape[0])
        self._lastind = int(index_lookup[bool_list])
        logging.info('selected index: ' + str(self._lastind))
        self.update()

    def clicked(self, event):
        if self._lastind is None:
            logging.info('button click: nothing selected!')
            return
        sysid = self._data.iloc[self._lastind].name
        logging.info('button click: current ID: {}'.format(sysid))

        self._ax[3].cla()
        self._ax[3].text(0.05, 0.95, 'initializing algorithm...', transform=self._ax[3].transAxes,
                        va='top', fontname='monospace')
        self._ax[3].set_xlabel('Day number')
        self._ax[3].set_yticks([])
        self._ax[3].set_ylabel('(sunset)        Time of day        (sunrise)')
        plt.tight_layout()
        self._fig.canvas.draw()

        self._ax[4].cla()

        D = self._power_signals_d
        cached_files = os.listdir(self._scsf_cache_dir)
        fn = 'pvo_' + str(sysid) + '.scsf'
        if fn in cached_files:
            iterative_fitting = IterativeFitting.load_instance(
                self._scsf_cache_dir + fn)
            self._iterative_fitting = iterative_fitting
            self._ax[4].plot(np.sum(iterative_fitting.power_signals_d, axis=0)
                 * 24 / iterative_fitting.power_signals_d.shape[0],
                 linewidth=1, label='raw data')
            use_day = iterative_fitting.weights > 1e-1
            days = np.arange(iterative_fitting.power_signals_d.shape[1])
            self._ax[4].scatter(days[use_day],
                np.sum(iterative_fitting.power_signals_d, axis=0)[use_day]
                    * 24 / iterative_fitting.power_signals_d.shape[0],
                color='orange', alpha=0.7, label='days selected')
            self._ax[4].legend()
            self._ax[4].set_title('Daily Energy')
            self._ax[4].set_xlabel('Day Number')
            self._ax[4].set_ylabel('kWh')
            self._ax[3].cla()
            self._ax[3].text(0.05, 0.95, 'loading cached results...',
                             transform=self._ax[3].transAxes,
                             va='top', fontname='monospace')
            self._ax[3].set_xlabel('Day number')
            self._ax[3].set_yticks([])
            self._ax[3].set_ylabel('(sunset)        Time of day        (sunrise)')
            self.show_ticks(self._ax[2])
            plt.tight_layout()
            self._fig.canvas.draw()
        else:
            iterative_fitting = IterativeClearSky(power_signals_d)
            self._iterative_fitting = iterative_fitting
            self._ax[4].plot(np.sum(iterative_fitting.power_signals_d, axis=0) * 24 / iterative_fitting.power_signals_d.shape[0], linewidth=1, label='raw data')
            use_day = iterative_fitting.weights > 1e-1
            days = np.arange(iterative_fitting.power_signals_d.shape[1])
            self._ax[4].scatter(days[use_day], np.sum(iterative_fitting.power_signals_d, axis=0)[use_day] * 24 / iterative_fitting.power_signals_d.shape[0],
                               color='orange', alpha=0.7, label='days selected')
            self._ax[4].legend()
            self._ax[4].set_title('Daily Energy')
            self._ax[4].set_xlabel('Day Number')
            self._ax[4].set_ylabel('kWh')
            self._ax[3].cla()
            self._ax[3].text(0.05, 0.95, 'running algorithm...', transform=self._ax[3].transAxes,
                            va='top', fontname='monospace')
            self._ax[3].set_xlabel('Day number')
            self._ax[3].set_yticks([])
            self._ax[3].set_ylabel('(sunset)        Time of day        (sunrise)')
            self.show_ticks(self._ax[2])
            plt.tight_layout()
            self._fig.canvas.draw()
            logging.info('starting algorithm')
            config_l = CONFIG1.copy()
            config_l['max_iter'] = 1
            obj_vals = iterative_fitting.calculate_objective(False)
            old_obj = np.sum(obj_vals)
            ti = time.time()
            for cntr in range(CONFIG1['max_iter']):
                iterative_fitting.execute(**config_l)
                logging.info('min iteration {} complete'.format(cntr + 1))
                obj_vals = iterative_fitting.calculate_objective(False)
                new_obj = np.sum(obj_vals)
                improvement = (old_obj - new_obj) * 1. / old_obj

                self._ax[3].cla()
                self._ax[3].set_xlabel('Day number')
                self._ax[3].set_yticks([])
                self._ax[3].set_ylabel('(sunset)        Time of day        (sunrise)')
                s1 = 'Iteration {} complete: obj = {:.2f}, f1 = {:.2f}'.format(cntr + 1, new_obj, obj_vals[0])
                s2 = 'Improvement: {:.2f}%'.format(100 * improvement)
                tf = time.time()
                s3 = 'Time elapsed: {:.2f} minutes'.format((tf - ti) / 60.)
                textout = '\n'.join([s1, s2, s3])
                logging.info(textout)
                self._ax[3].text(0.05, 0.95, textout, transform=self._ax[3].transAxes,
                                va='top', fontname='monospace')
                plt.tight_layout()
                self._fig.canvas.draw()
                old_obj = new_obj
                if improvement <= CONFIG1['eps']:
                    break
            iterative_fitting.save_instance(self._scsf_cache_dir + fn)

        logging.info('algorithm complete')
        self._ax[4].plot((iterative_fitting.r_cs_value[0] *
            np.sum(iterative_fitting.l_cs_value[:, 0])) * 24 /
            iterative_fitting.power_signals_d.shape[0],
            linewidth=1, label='clear sky estimate')
        self._ax[4].legend()
        logging.info('first plot complete')
        with sns.axes_style('white'):
            self._ax[3].cla()
            bar = self._ax[3].imshow(
                iterative_fitting.clear_sky_signals(), cmap='hot',
                vmin=0, vmax=np.max(iterative_fitting.power_signals_d),
                interpolation='none', aspect='auto')
            if self._cb2 is not None:
                self._cb2.remove()
            self._cb2 = plt.colorbar(bar, ax=self._ax[3], label='kW')
        self.show_ticks(self._ax[3])
        self._ax[3].set_title('Estimated clear sky power')
        self._ax[3].set_xlabel('Day number')
        self._ax[3].set_yticks([])
        self._ax[3].set_ylabel('(sunset)        Time of day        (sunrise)')
        logging.info('second plot complete')
        plt.tight_layout()
        self._fig.canvas.draw()
        return

    def show_ticks(self, ax):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        use_day = self._iterative_fitting.weights > 1e-1
        days = np.arange(self._iterative_fitting.power_signals_d.shape[1])
        y1 = np.ones_like(days[use_day]) * self._power_signals_d.shape[0] * .99
        ax.scatter(days[use_day], y1, marker='|', color='yellow', s=2)
        ax.scatter(days[use_day], .995 * y1, marker='|', color='yellow', s=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return

    def onpress(self, event):
        if self._lastind is None:
            return

        logging.info('press event: ' + str(event.key))

        if event.key == 'a':
            inc = -1
            self._lastind += inc
            self._lastind = np.clip(self._lastind, 0, len(self._xs) - 1)
        elif event.key == 's':
            inc = 1
            self._lastind += inc
            self._lastind = np.clip(self._lastind, 0, len(self._xs) - 1)
        else:
            return

        self.update()

    def onpick(self, event):
        if event.artist != self._line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        logging.info('pick: ' + str(x) + ', ' + str(y))

        distances = np.hypot(x - self._xs[event.ind], y - self._ys[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self._lastind = dataind
        self.update()

    def update(self):
        if self._lastind is None:
            return

        dataind = self._lastind
        prcntl = self._prcntl

        logging.info('updating, ID = {}'.format(self._data.iloc[dataind].name))

        self._selected.set_visible(True)
        self._selected.set_data(self._xs[dataind], self._ys[dataind])

        out1 = 'system ID: {:d}'.format(self._data.iloc[dataind].name)
        out2 = str(self._data.iloc[dataind])
        # self._text_box.set_val('')

        idxs = np.arange(len(self._data.columns))
        if self._data.iloc[dataind]['res-median'] > np.percentile(self._data['res-median'], prcntl):
            l1 = out2.split('\n')
            i = idxs[self._data.columns == 'res-median'][0]
            l1[i] = '*' + l1[i][:-2] + '*'
            out2 = '\n'.join(l1)
        if self._data.iloc[dataind]['res-var'] > np.percentile(self._data['res-var'], prcntl):
            l1 = out2.split('\n')
            i = idxs[self._data.columns == 'res-var'][0]
            l1[i] = '*' + l1[i][:-2] + '*'
            out2 = '\n'.join(l1)
        if self._data.iloc[dataind]['res-L0norm'] > np.percentile(self._data['res-L0norm'], prcntl):
            l1 = out2.split('\n')
            i = idxs[self._data.columns == 'res-L0norm'][0]
            l1[i] = '*' + l1[i][:-2] + '*'
            out2 = '\n'.join(l1)

        self._text.set_text(out1)
        self._ax[1].cla()
        self._ax[1].text(0.00, 0.95, out2, transform=self._ax[1].transAxes, va='top', fontname='monospace')
        self._ax[1].set_axis_off()
        self._ax[2].cla()
        self._ax[2].text(0.05, 0.95, 'data loading...', transform=self._ax[2].transAxes, va='top', fontname='monospace')
        self._ax[2].set_xlabel('Day number')
        self._ax[2].set_yticks([])
        self._ax[2].set_ylabel('(sunset)        Time of day        (sunrise)')
        self._ax[3].cla()
        self._ax[4].cla()
        self._iterative_fitting = None
        plt.tight_layout()
        self._fig.canvas.draw()

        with sns.axes_style('white'):
            idnum = self._data.iloc[dataind].name
            if idnum in self._local_cash.keys():
                df = self._local_cash[idnum]
            else:
                df = load_sys(idnum=idnum, local=False)
                self._local_cash[idnum] = df
            days = df.resample('D').max().index[1:-1]
            start = days[0]
            end = days[-1]
            power_signals_d = df.loc[start:end].iloc[:-1].values.reshape(
                288, -1, order='F')
            self._power_signals_d = power_signals_d
            self._ax[2].cla()
            foo = self._ax[2].imshow(power_signals_d, cmap='hot', interpolation='none', aspect='auto')
            if self._cb is not None:
                self._cb.remove()
            self._cb = plt.colorbar(foo, ax=self._ax[2], label='kW')

        self._ax[2].set_xlabel('Day number')
        self._ax[2].set_yticks([])
        self._ax[2].set_ylabel('(sunset)        Time of day        (sunrise)')
        self._ax[2].set_title('Measured power')

        self._text_box.set_val('')
        self._fig.canvas.draw()

    @property
    def iterative_fitting(self):
        return self._iterative_fitting

def view_ts(pb, clear_day_start=None, day_start=None):
    if pb.iterative_fitting is not None:
        clear_days = np.arange(
            len(pb.iterative_fitting.weights))[
                pb.iterative_fitting.weights >= 1e-3]
        fig = pb.iterative_fitting.ts_plot_with_weights(
            num_days=len(pb.iterative_fitting.weights), figsize=(9, 6),
            fig_title='System ID: {}'.format(pb.data.iloc[pb.lastind].name))
        if clear_day_start is not None:
            N = clear_day_start
            plt.xlim(clear_days[N] - 2, clear_days[N] - 2 + 5)
        elif day_start is not None:
            plt.xlim(day_start, day_start+5)
        else:
            plt.xlim(0, 5)
        plt.show()
