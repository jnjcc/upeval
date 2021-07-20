# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .uputil import ensure_dir

class UpBaseEval:
    """ Uplift Evaluation Base Class """
    def __init__(self, col_treatment, col_outcome, col_uscore, ckpt_dir):
        """ `col_uscore': predicted uplift score """
        self.col_treatment_ = col_treatment
        self.col_outcome_ = col_outcome
        self.col_uscore_ = col_uscore
        if ckpt_dir is not None:
            ensure_dir(ckpt_dir)
        self.ckpt_dir_ = ckpt_dir

    def plot(self, fname = None):
        if (self.ckpt_dir_ is not None) and (fname is not None):
            plt.savefig(self.ckpt_dir_ + "/" + fname, bbox_inches = "tight")
            plt.close()
        else:
            plt.show()

class UpContEval(UpBaseEval):
    """ Uplift Evaluation with continuous outcome """
    def __init__(self, dframe, col_treatment = "Treatment", col_outcome = "Outcome", col_uscore = "Uscore",
            ckpt_dir = "./", ascending = False):
        """ `dframe': `col_treatment', `col_outcome', `col_uscore' needed """
        super().__init__(col_treatment, col_outcome, col_uscore, ckpt_dir)
        self.ascending_ = ascending

        self.df_ = dframe[[col_treatment, col_outcome, col_uscore]]
        self.df_cumsumed_, self.col_scuplift_, self.col_scgains_, self.col_sqini_, \
                self.col_rcgains_, self.col_rqini_, self.col_sbalance_ = self._cumsumed()

    def _cumsumed(self):
        """ cumsum by `col_uscore' """
        ## Without shuffling, the following DataFrame might seem to have imbalance problems in cumsum plots
        ##     | Treatment | Uscore |
        ##     |-----------|--------|
        ##     | 0         | 0.6    |
        ##     | 1         | 0.5    |
        ##     | ...       | 0.5    |
        ##     | 1         | 0.5    |
        ##     | 0         | 0.5    |
        ##     | ...       | 0.5    |
        ##     | 0         | 0.5    |
        ##     | 0         | 0.2    |
        df_sort = self.df_.sample(frac = 1.0) ## shuffle the DataFrame before sorting
        df_sort = df_sort.sort_values(self.col_uscore_, ascending = self.ascending_).reset_index(drop = True)
        df_sort.index = df_sort.index + 1
        df_sort["Rt"] = (df_sort[self.col_outcome_] * df_sort[self.col_treatment_]).cumsum()
        df_sort["Nt"] = df_sort[self.col_treatment_].cumsum()
        df_sort["Rc"] = (df_sort[self.col_outcome_] * (1 - df_sort[self.col_treatment_])).cumsum()
        df_sort["Nc"] = df_sort.index.values - df_sort["Nt"]

        cuplift, cgains, qini, rcgains, rqini, balance = ("cuplift", "cgains", "qini", "rcgains", "rqini", "balance")
        df_sort[cuplift] = df_sort["Rt"] / df_sort["Nt"] - df_sort["Rc"] / df_sort["Nc"]
        ## cgains: (Rt/Nt - Rc/Nc) * (Nt + Nc)
        df_sort[cgains] = df_sort[cuplift] * df_sort.index.values
        ## qini: Rt - (Rc * Nt) / Nc
        df_sort[qini] = df_sort["Rt"] - (df_sort["Rc"] * df_sort["Nt"]) / df_sort["Nc"]
        ## in pylift:
        ##   cgains = cgains / N_all
        ##   qini = Rt / Nt_all - Rc / Nc_all
        ## Nt_all = df_sort["Nt"].iloc[-1]
        ## Nc_all = df_sort["Nc"].iloc[-1]
        ## N_all = Nt_all + Nc_all
        ## df_sort[cgains] = df_sort[cgains] / N_all
        ## df_sort[qini] = df_sort["Rt"] / Nt_all - df_sort["Rc"] / Nc_all
        df_sort[balance] = df_sort["Nt"] * 1.0 / (df_sort["Nt"] + df_sort["Nc"])

        df_sort.loc[0] = np.zeros((df_sort.shape[1], ))
        df_sort = df_sort.sort_index().interpolate()

        popu = df_sort.index.values[-1]
        cgains_last = df_sort[cgains].iloc[-1]
        qini_last = df_sort[qini].iloc[-1]
        df_sort[rcgains] = df_sort.index.values * cgains_last / popu
        df_sort[rqini] = df_sort.index.values * qini_last / popu
        return df_sort[[cuplift, cgains, qini, rcgains, rqini, balance]], cuplift, cgains, qini, rcgains, rqini, balance

    def auuc_score(self, normalized = True):
        cgain = self.df_cumsumed_[self.col_scgains_]
        if normalized:
            ## the fraction of gain from treating the entire population, to normalize y-axis
            cgain = cgain / np.abs(cgain.iloc[-1])
        ## cgain.shape[0]: # populations, to normalize x-axis
        return cgain.sum() / cgain.shape[0]

    def qini_score(self, normalized = True):
        qini = self.df_cumsumed_[self.col_sqini_]
        rqini = self.df_cumsumed_[self.col_rqini_]
        if normalized:
            qini = qini / np.abs(qini.iloc[-1])
            rqini = rqini / np.abs(rqini.iloc[-1])
        return (qini.sum() - rqini.sum()) / qini.shape[0]

    def _sampling(self, n = 100):
        dfplot = self.df_cumsumed_
        if (n is not None) and (n < dfplot.shape[0]):
            dfplot = dfplot.iloc[np.linspace(0, dfplot.index[-1], n, endpoint = True)]
        return dfplot

    def plot_cumsum(self, plot_type, dfplot = None, n = 100):
        if dfplot is None:
            dfplot = self._sampling(n)
        axs = plt.gca()
        ylabel = None
        if plot_type == "cuplift":
            axs.plot(dfplot.index.values[1:], dfplot[self.col_scuplift_][1:], linewidth = 1)
            ylabel = "Cumulative Uplift"
        elif plot_type == "cgains":
            axs.plot(dfplot.index.values, dfplot[self.col_scgains_], linewidth = 1)
            axs.plot(dfplot.index.values, dfplot[self.col_rcgains_], linewidth = 1, linestyle = "--")
            ylabel = "Cumulative Gains"
        elif plot_type == "qini":
            axs.plot(dfplot.index.values, dfplot[self.col_sqini_], linewidth = 1)
            axs.plot(dfplot.index.values, dfplot[self.col_rqini_], linewidth = 1, linestyle = "--")
            ylabel = "Qini"
        elif plot_type == "balance":
            axs.scatter(dfplot.index.values[1:], dfplot[self.col_sbalance_][1:], s = 2)
            ylabel = "% Treatment"
        axs.set_xlabel("Population")
        axs.set_ylabel(ylabel)

    def plot_cumsums(self, fname = None, n = 100, plot_types = ["cuplift", "cgains", "balance"]):
        dfplot = self._sampling(n)
        nplots = len(plot_types)
        nrows = nplots
        fig, axes = plt.subplots(nrows = nrows, ncols = 1)
        title = "auuc: %.3f qini: %.3f" % (self.auuc_score(), self.qini_score())
        fig.suptitle(title)
        fig.tight_layout()
        for i in range(nplots):
            plt.subplot(nrows, 1, i+1)
            self.plot_cumsum(plot_types[i], dfplot, n)
        self.plot(fname)

class UpContEvalQ(UpContEval):
    """ Uplift Evaluation with Quantiles """
    def __init__(self, dframe, col_treatment = "Treatment", col_outcome = "Outcome", col_uscore = "Uscore",
            ckpt_dir = "./", nquantiles = 10):
        """ if `nquantiles' == 10, then deciles """
        super().__init__(dframe, col_treatment, col_outcome, col_uscore, ckpt_dir)

        self.df_quantiled_, self.col_qgrp_, self.col_quplift_, self.col_qcuplift_, \
                self.col_qcgains_, self.col_qcupratio_, self.col_qbalance_ = self._quantiled(nquantiles)
        if self.df_quantiled_.shape[0] < nquantiles:
            logging.warning("%s not enough uscores to be quantiled" % (ckpt_dir))

    def _quantiled(self, nquantiles):
        col_qgrp = "ux_grp"
        self.grp_range_, self.grp_bins_ = pd.qcut(self.df_[self.col_uscore_], q = nquantiles, retbins = True,
                duplicates = "drop")
        self.grp_range_.name = col_qgrp

        grp, _ = pd.qcut(self.df_[self.col_uscore_], q = nquantiles, retbins = True, duplicates = "drop",
                labels = False)
        grp = nquantiles - grp ## 9 reverse to 1, which means top 10%
        grp.name = col_qgrp
        self.df_ = pd.concat([self.df_, grp], axis = 1)

        agged = self.df_.groupby([self.col_treatment_, col_qgrp]).agg({self.col_outcome_: ["sum", "count"]})
        agged.columns = ["sum", "count"] # , "mean"]
        agged = agged.reset_index()

        col_rt, col_nt, col_rc, col_nc = ("rt", "nt", "rc", "nc")
        col_Rt, col_Nt, col_Rc, col_Nc = ("Rt", "Nt", "Rc", "Nc")
        treat_mask = (agged[self.col_treatment_] == 1)
        treated = agged[treat_mask].drop(self.col_treatment_, axis = 1)
        treated.columns = [col_qgrp, col_rt, col_nt]
        control = agged[~treat_mask].drop(self.col_treatment_, axis = 1)
        control.columns = [col_qgrp, col_rc, col_nc]
        dfmerge = pd.merge(treated, control, on = col_qgrp, how = "inner")

        ## TODO: already sorted by `groupby(self.col_treatment_, col_qgrp)'?
        dfmerge = dfmerge.sort_values(col_qgrp, ascending = True)
        dfcum = dfmerge.drop(col_qgrp, axis = 1).cumsum()
        dfcum.columns = [col_Rt, col_Nt, col_Rc, col_Nc]

        uplift, cuplift, cupratio, cgains, balance = ("uplift", "cuplift", "cupratio", "cgains", "balance")
        dfmerge[uplift] = dfmerge[col_rt] / dfmerge[col_nt] - dfmerge[col_rc] / dfmerge[col_nc]

        dfcum[cuplift] = dfcum[col_Rt] / dfcum[col_Nt] - dfcum[col_Rc] / dfcum[col_Nc]
        dfcum[cupratio] = dfcum[cuplift] / (dfcum[col_Rc] / dfcum[col_Nc])
        dfcum[cgains] = dfcum[cuplift] * (dfcum[col_Nt] + dfcum[col_Nc])
        dfcum[balance] = dfcum[col_Nt] / (dfcum[col_Nt] + dfcum[col_Nc])
        return pd.concat([dfmerge, dfcum], axis = 1), col_qgrp, uplift, cuplift, cupratio, cgains, balance

    def plot_quantile(self, plot_type, **kwargs):
        axs = plt.gca()
        if plot_type == "hist":
            ## histogram of u(x) score
            axs.hist(self.df_[self.col_uscore_], **kwargs)
            axs.set_xlabel("u(x) score")
            axs.set_ylabel("% population")
        elif plot_type == "uplift":
            ## true uplift in each quantile bin
            dfplot = self.df_quantiled_
            axs.bar(dfplot[self.col_qgrp_], dfplot[self.col_quplift_])
            axs.set_xlabel("u(x) top %")
            axs.set_ylabel("True Uplift")
        elif plot_type == "count":
            ## treatment/control size in each quantile bin
            df_count = self.df_.groupby([self.col_qgrp_, self.col_treatment_])[self.col_treatment_].count()
            df_count = df_count.unstack(-1)
            df_count.plot.bar(ax = axs)
        elif plot_type == "balance":
            df_count = self.df_.groupby([self.col_qgrp_])[self.col_treatment_].mean()
            df_count = df_count.reset_index()
            axs.bar(df_count[self.col_qgrp_], df_count[self.col_treatment_])
            axs.set_ylabel("% Treatment")
        elif plot_type == "u2u":
            ## Uplift-to-Uscore: True Uplift vs. mean u(x) score
            uplift = self.df_quantiled_[self.col_quplift_]
            uscore = self.df_.groupby(self.col_qgrp_)[self.col_uscore_].mean()
            minval = min(uplift.min(), uscore.min())
            maxval = max(uplift.max(), uscore.max())
            axs.plot([minval, maxval], [minval, maxval], linestyle = "--")
            try:
                ## TODO: it is possible there are no treated or control samples in a certain ux_grp,
                ##   so that (uscore.shape != uplift.shape)
                axs.scatter(uscore, uplift)
            except:
                pass
            axs.set_xlabel("u(x) score")
            axs.set_ylabel("True Uplift")
        elif plot_type == "cuplift":
            dfplot = self.df_quantiled_
            axs.bar(dfplot[self.col_qgrp_], dfplot[self.col_qcuplift_])
            axs.set_xlabel("u(x) top %")
            axs.set_ylabel("Cumulative Uplift")
        elif plot_type == "cgains":
            dfplot = self.df_quantiled_
            axs.bar(dfplot[self.col_qgrp_], dfplot[self.col_qcgains_])
            axs.set_xlabel("u(x) top %")
            axs.set_ylabel("Cumulative Gains")
        elif plot_type == "cupratio":
            ## Cumulative Uplift Ratio
            dfplot = self.df_quantiled_
            axs.bar(dfplot[self.col_qgrp_], dfplot[self.col_qcupratio_])
            axs.set_xlabel("u(x) top %")
            axs.set_ylabel("Cumulative Uplift Ratio")

    def plot_quantiles(self, fname = None, plot_types = ["hist", "cuplift", "balance", "u2u", "cupratio", "cgains"], **kwargs):
        nplots = len(plot_types)
        nrows = 2 if nplots >= 2 else 1
        ncols = int(np.ceil(nplots / nrows))
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols)
        title = "auuc: %.3f qini: %.3f" % (self.auuc_score(), self.qini_score())
        fig.suptitle(title)
        fig.tight_layout()
        for i in range(nplots):
            plt.subplot(nrows, ncols, i + 1)
            self.plot_quantile(plot_types[i], **kwargs)
        self.plot(fname)

def make_eval(treatment, outcome, uscore, quantile = True, ckpt_dir = None):
    col_treatment, col_outcome, col_uscore = ("Treatment", "Outcome", "Uscore")
    df = pd.DataFrame({
        col_treatment: treatment, col_outcome: outcome, col_uscore: uscore
    })
    if quantile:
        return UpContEvalQ(df, col_treatment = col_treatment, col_outcome = col_outcome,
                col_uscore = col_uscore, ckpt_dir = ckpt_dir)
    else:
        return UpContEval(df, col_treatment = col_treatment, col_outcome = col_outcome,
                col_uscore = col_uscore, ckpt_dir = ckpt_dir)

def plot_eval(treatment, outcome, uscore, ckpt_dir = None):
    evaler = make_eval(treatment, outcome, uscore, quantile = True, ckpt_dir = ckpt_dir)
    evaler.plot_cumsums("cumsum.png")
    evaler.plot_quantiles("quantile.png", bins = 50, density = True)
    return evaler

if __name__ == "__main__":
    df = pd.DataFrame({
        "Treatment": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        "Outcome":   [6, 7, 1, 1, 7, 8, 2, 2, 9, 10],
        "Uscore":    [0.8, 0.7, 0.8, 0.7, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5]
    })
    upq = UpContEvalQ(df, nquantiles = 3)
    upq.plot_cumsums("cumsum.png")
    upq.plot_quantiles("quantile.png", bins = 50, density = True)
