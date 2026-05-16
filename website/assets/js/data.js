/* Data extracted from results/tables/*.csv and the paper.
   Loaded synchronously before charts.js. */

window.DGA_DATA = {

  /* ks_test_domain_shift.csv */
  ks: [
    { feature: 'CO',         stat: 0.2904, p: 2.05e-9, sig: true,  mag: 'MED' },
    { feature: 'CO₂',        stat: 0.2646, p: 7.09e-8, sig: true,  mag: 'MED' },
    { feature: 'TCG',        stat: 0.1321, p: 2.75e-2, sig: true,  mag: 'LOW' },
    { feature: 'CO₂ / CO',   stat: 0.1053, p: 0.129,   sig: false, mag: 'LOW' },
    { feature: 'C₂H₆',       stat: 0.0906, p: 0.261,   sig: false, mag: 'LOW' },
    { feature: 'CH₄ / H₂',   stat: 0.0850, p: 0.331,   sig: false, mag: 'LOW' },
    { feature: 'CH₄',        stat: 0.0766, p: 0.457,   sig: false, mag: 'LOW' },
    { feature: 'C₂H₄',       stat: 0.0673, p: 0.621,   sig: false, mag: 'LOW' },
    { feature: 'C₂H₂ / C₂H₄',stat: 0.0566, p: 0.814,   sig: false, mag: 'LOW' },
    { feature: 'C₂H₂',       stat: 0.0543, p: 0.849,   sig: false, mag: 'LOW' },
    { feature: 'C₂H₆ / CH₄', stat: 0.0540, p: 0.855,   sig: false, mag: 'LOW' },
    { feature: 'H₂',         stat: 0.0500, p: 0.908,   sig: false, mag: 'LOW' }
  ],

  /* results_7class.csv */
  results7: [
    { method: 'Random Forest',           type: 'Baseline',          acc: 0.906, accStd: 0.068, f1: 0.833, f1Std: 0.137, prec: 0.869, rec: 0.838 },
    { method: 'MLP + TL (Full Finetune)',type: 'Transfer Learning', acc: 0.811, accStd: 0.033, f1: 0.760, f1Std: 0.080, prec: 0.796, rec: 0.801 },
    { method: 'XGBoost',                 type: 'Baseline',          acc: 0.859, accStd: 0.056, f1: 0.756, f1Std: 0.126, prec: 0.771, rec: 0.771 },
    { method: 'MLP + TL (Progressive)',  type: 'Transfer Learning', acc: 0.797, accStd: 0.021, f1: 0.747, f1Std: 0.070, prec: 0.785, rec: 0.784 },
    { method: 'MLP (source only)',       type: 'Transfer Learning', acc: 0.764, accStd: 0.000, f1: 0.743, f1Std: 0.000, prec: 0.780, rec: 0.781 },
    { method: 'MLP + TL (Freeze Head)',  type: 'Transfer Learning', acc: 0.783, accStd: 0.029, f1: 0.738, f1Std: 0.070, prec: 0.778, rec: 0.778 },
    { method: 'CNN + TL (Progressive)',  type: 'Transfer Learning', acc: 0.702, accStd: 0.036, f1: 0.674, f1Std: 0.041, prec: 0.707, rec: 0.700 },
    { method: 'CNN + TL (Freeze Head)',  type: 'Transfer Learning', acc: 0.716, accStd: 0.057, f1: 0.672, f1Std: 0.096, prec: 0.694, rec: 0.706 },
    { method: 'CNN + TL (Full Finetune)',type: 'Transfer Learning', acc: 0.702, accStd: 0.059, f1: 0.670, f1Std: 0.075, prec: 0.704, rec: 0.701 },
    { method: 'MLP (no TL)',             type: 'Transfer Learning', acc: 0.784, accStd: 0.081, f1: 0.670, f1Std: 0.138, prec: 0.700, rec: 0.676 },
    { method: 'CNN (source only)',       type: 'Transfer Learning', acc: 0.709, accStd: 0.000, f1: 0.656, f1Std: 0.000, prec: 0.673, rec: 0.684 },
    { method: 'SVM (RBF)',               type: 'Baseline',          acc: 0.791, accStd: 0.065, f1: 0.655, f1Std: 0.100, prec: 0.698, rec: 0.656 },
    { method: 'KNN (k=5)',               type: 'Baseline',          acc: 0.689, accStd: 0.083, f1: 0.505, f1Std: 0.124, prec: 0.570, rec: 0.497 }
  ],

  /* results_4class.csv */
  results4: [
    { method: 'Random Forest',           type: 'Baseline',          acc: 0.960, accStd: 0.025, f1: 0.883, f1Std: 0.092, prec: 0.889, rec: 0.889 },
    { method: 'XGBoost',                 type: 'Baseline',          acc: 0.926, accStd: 0.038, f1: 0.845, f1Std: 0.072, prec: 0.887, rec: 0.843 },
    { method: 'MLP (source only)',       type: 'Transfer Learning', acc: 0.845, accStd: 0.000, f1: 0.781, f1Std: 0.000, prec: 0.788, rec: 0.819 },
    { method: 'MLP (no TL)',             type: 'Transfer Learning', acc: 0.892, accStd: 0.044, f1: 0.774, f1Std: 0.098, prec: 0.787, rec: 0.775 },
    { method: 'MLP + TL (Freeze Head)',  type: 'Transfer Learning', acc: 0.845, accStd: 0.041, f1: 0.769, f1Std: 0.059, prec: 0.786, rec: 0.818 },
    { method: 'MLP + TL (Progressive)',  type: 'Transfer Learning', acc: 0.845, accStd: 0.041, f1: 0.769, f1Std: 0.059, prec: 0.786, rec: 0.818 },
    { method: 'MLP + TL (Full Finetune)',type: 'Transfer Learning', acc: 0.838, accStd: 0.049, f1: 0.758, f1Std: 0.067, prec: 0.772, rec: 0.792 },
    { method: 'CNN + TL (Progressive)',  type: 'Transfer Learning', acc: 0.804, accStd: 0.034, f1: 0.741, f1Std: 0.033, prec: 0.751, rec: 0.781 },
    { method: 'CNN + TL (Full Finetune)',type: 'Transfer Learning', acc: 0.797, accStd: 0.019, f1: 0.735, f1Std: 0.019, prec: 0.745, rec: 0.777 },
    { method: 'CNN + TL (Freeze Head)',  type: 'Transfer Learning', acc: 0.797, accStd: 0.019, f1: 0.735, f1Std: 0.019, prec: 0.745, rec: 0.777 },
    { method: 'CNN (source only)',       type: 'Transfer Learning', acc: 0.784, accStd: 0.000, f1: 0.718, f1Std: 0.000, prec: 0.730, rec: 0.751 },
    { method: 'SVM (RBF)',               type: 'Baseline',          acc: 0.866, accStd: 0.055, f1: 0.698, f1Std: 0.033, prec: 0.694, rec: 0.714 },
    { method: 'KNN (k=5)',               type: 'Baseline',          acc: 0.811, accStd: 0.062, f1: 0.665, f1Std: 0.054, prec: 0.699, rec: 0.666 }
  ],

  /* Data scarcity ablation — extracted from the paper. */
  scarcity: {
    sizes:  [50,   75,    100,   150  ],
    mlpTL:  [0.773, 0.766, 0.756, 0.760],
    rf:     [0.778, 0.676, 0.739, 0.833]
  },

  /* anova_feature_ranking.csv */
  anova: [
    { feature: 'TCG',          f: 52.80, p: 1.11e-33, sig: true  },
    { feature: 'C₂H₂',         f: 48.95, p: 4.03e-32, sig: true  },
    { feature: 'C₂H₄',         f: 35.69, p: 4.98e-26, sig: true  },
    { feature: 'C₂H₆',         f: 25.61, p: 1.97e-20, sig: true  },
    { feature: 'H₂',           f: 21.37, p: 9.52e-18, sig: true  },
    { feature: 'C₂H₂ / C₂H₄',  f: 21.07, p: 1.50e-17, sig: true  },
    { feature: 'CO₂',          f: 18.78, p: 5.52e-16, sig: true  },
    { feature: 'CO',           f: 13.81, p: 2.59e-12, sig: true  },
    { feature: 'CH₄',          f: 10.45, p: 1.42e-09, sig: true  },
    { feature: 'CH₄ / H₂',     f:  5.13, p: 8.48e-05, sig: true  },
    { feature: 'C₂H₆ / CH₄',   f:  3.97, p: 1.06e-03, sig: true  },
    { feature: 'CO₂ / CO',     f:  0.61, p: 0.725,    sig: false }
  ]
};
