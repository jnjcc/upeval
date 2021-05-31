# -*- coding: utf-8 -*-

import shap
from .upeval import UpBaseEval

class UpTreeShap(UpBaseEval):
    def __init__(self, model, col_treatment = "Treatment", col_outcome = "Outcome", ckpt_dir = "./"):
        super().__init__(col_treatment, col_outcome, None, ckpt_dir)
        self.model_ = model
        self.explainer_ = shap.TreeExplainer(self.model_)

    def plot_shap(self, dframe, fname = None):
        dframe = dframe.drop([self.col_treatment_, self.col_outcome_], axis = 1, errors = "ignore")
        shap_values = self.explainer_.shap_values(dframe, check_additivity = False)
        shap.summary_plot(shap_values, dframe, show = False)
        self.plot(fname)

if __name__ == "__main__":
    pass
