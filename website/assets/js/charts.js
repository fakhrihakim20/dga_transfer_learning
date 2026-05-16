/* Plotly charts — call window.DGA_CHARTS.renderAll() after DOM ready. */

(function () {
  'use strict';

  const PALETTE = {
    ink:    '#0B0B0D',
    ink3:   '#5C5C63',
    accent: '#D94F1A',
    teal:   '#1F4E5F',
    line:   '#E5E2D9',
    line2:  '#D5D1C4',
    bg:     '#FAFAF7',
    bg2:    '#F3F0E7',
    muted:  '#B8B5AB'
  };

  const FONT = {
    family: 'Inter, system-ui, sans-serif',
    color:  PALETTE.ink,
    size:   13
  };

  const LAYOUT_BASE = {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    font: FONT,
    margin: { l: 110, r: 32, t: 16, b: 56 },
    hoverlabel: {
      bgcolor: PALETTE.ink,
      bordercolor: PALETTE.ink,
      font: { color: PALETTE.bg, family: FONT.family, size: 13 }
    },
    xaxis: {
      gridcolor: PALETTE.line,
      zerolinecolor: PALETTE.line2,
      tickfont: { color: PALETTE.ink3, size: 12 },
      titlefont: { color: PALETTE.ink3, size: 12 }
    },
    yaxis: {
      gridcolor: PALETTE.line,
      zerolinecolor: PALETTE.line2,
      tickfont: { color: PALETTE.ink3, size: 12 },
      titlefont: { color: PALETTE.ink3, size: 12 }
    }
  };

  const CONFIG = {
    displayModeBar: false,
    responsive: true
  };

  /* ─────────────────────────────────────────────────────────────
     1. KS-test domain shift bar chart
     ───────────────────────────────────────────────────────────── */
  function renderKS() {
    const data = window.DGA_DATA.ks.slice().sort((a, b) => a.stat - b.stat);

    const trace = {
      type: 'bar',
      orientation: 'h',
      x: data.map(d => d.stat),
      y: data.map(d => d.feature),
      marker: {
        color: data.map(d => d.sig ? PALETTE.accent : PALETTE.muted),
        line: { color: 'transparent', width: 0 }
      },
      customdata: data.map(d => [d.p.toExponential(2), d.sig ? 'Significant (p < 0.05)' : 'Not significant']),
      hovertemplate:
        '<b>%{y}</b><br>' +
        'KS statistic: %{x:.4f}<br>' +
        'p-value: %{customdata[0]}<br>' +
        '%{customdata[1]}<extra></extra>',
      text: data.map(d => d.sig ? d.stat.toFixed(3) : ''),
      textposition: 'outside',
      textfont: { color: PALETTE.accent, family: FONT.family, size: 12 },
      cliponaxis: false
    };

    const layout = Object.assign({}, LAYOUT_BASE, {
      height: 460,
      margin: { l: 130, r: 60, t: 16, b: 50 },
      xaxis: Object.assign({}, LAYOUT_BASE.xaxis, {
        title: 'KS statistic (higher = larger shift)',
        range: [0, Math.max(...data.map(d => d.stat)) * 1.18]
      }),
      yaxis: Object.assign({}, LAYOUT_BASE.yaxis, {
        automargin: true,
        tickfont: { color: PALETTE.ink, size: 12, family: FONT.family }
      }),
      shapes: [{
        type: 'line',
        x0: 0.0937, x1: 0.0937,        // approximate KS critical value at α=0.05
        y0: -0.5, y1: data.length - 0.5,
        line: { color: PALETTE.ink3, width: 1, dash: 'dot' }
      }],
      annotations: [{
        x: 0.0937, y: data.length - 0.5,
        xref: 'x', yref: 'y',
        text: 'α = 0.05 threshold',
        showarrow: false,
        font: { color: PALETTE.ink3, size: 11 },
        xanchor: 'left', yanchor: 'top',
        xshift: 6
      }]
    });

    Plotly.newPlot('chart-ks', [trace], layout, CONFIG);
  }

  /* ─────────────────────────────────────────────────────────────
     2. Method comparison (tab toggle 7 / 4 class)
     ───────────────────────────────────────────────────────────── */
  function renderMethods(scheme) {
    const raw = scheme === 4 ? window.DGA_DATA.results4 : window.DGA_DATA.results7;
    const data = raw.slice().sort((a, b) => a.f1 - b.f1);

    const trace = {
      type: 'bar',
      orientation: 'h',
      x: data.map(d => d.f1),
      y: data.map(d => d.method),
      error_x: {
        type: 'data',
        array: data.map(d => d.f1Std),
        color: PALETTE.ink3,
        thickness: 1,
        width: 4
      },
      marker: {
        color: data.map(d => d.type === 'Baseline' ? PALETTE.teal : PALETTE.accent),
        line: { color: 'transparent', width: 0 }
      },
      customdata: data.map(d => [d.type, d.acc, d.accStd, d.prec, d.rec, d.f1Std]),
      hovertemplate:
        '<b>%{y}</b><br>' +
        '%{customdata[0]}<br>' +
        'Macro F1: %{x:.3f} ± %{customdata[5]:.3f}<br>' +
        'Accuracy: %{customdata[1]:.3f} ± %{customdata[2]:.3f}<br>' +
        'Precision: %{customdata[3]:.3f}<br>' +
        'Recall: %{customdata[4]:.3f}<extra></extra>',
      text: data.map(d => d.f1.toFixed(3)),
      textposition: 'outside',
      textfont: { color: PALETTE.ink, family: FONT.family, size: 12 },
      cliponaxis: false
    };

    const layout = Object.assign({}, LAYOUT_BASE, {
      height: 620,
      margin: { l: 200, r: 80, t: 16, b: 50 },
      xaxis: Object.assign({}, LAYOUT_BASE.xaxis, {
        title: 'Macro F1 score',
        range: [0.4, 1.0]
      }),
      yaxis: Object.assign({}, LAYOUT_BASE.yaxis, {
        automargin: true,
        tickfont: { color: PALETTE.ink, size: 12, family: FONT.family }
      }),
      showlegend: true,
      legend: {
        x: 1, y: 0.02, xanchor: 'right', yanchor: 'bottom',
        bgcolor: 'rgba(250,250,247,0.85)',
        bordercolor: PALETTE.line,
        borderwidth: 1,
        font: { size: 12 }
      }
    });

    // legend proxy traces
    const legendBase = {
      type: 'bar', orientation: 'h', x: [null], y: [null],
      showlegend: true, hoverinfo: 'skip'
    };
    const legendBaseline = Object.assign({}, legendBase, {
      name: 'Classical baseline',
      marker: { color: PALETTE.teal }
    });
    const legendTL = Object.assign({}, legendBase, {
      name: 'Transfer-learning variant',
      marker: { color: PALETTE.accent }
    });

    Plotly.newPlot('chart-methods', [trace, legendBaseline, legendTL], layout, CONFIG);
  }

  /* ─────────────────────────────────────────────────────────────
     3. Data scarcity line chart
     ───────────────────────────────────────────────────────────── */
  function renderScarcity() {
    const d = window.DGA_DATA.scarcity;

    const traceTL = {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'MLP + Transfer Learning',
      x: d.sizes,
      y: d.mlpTL,
      line: { color: PALETTE.accent, width: 3, shape: 'spline' },
      marker: { size: 11, color: PALETTE.accent, line: { color: PALETTE.bg, width: 2 } },
      hovertemplate: '<b>MLP + TL</b><br>Target n = %{x}<br>Macro F1 = %{y:.3f}<extra></extra>'
    };
    const traceRF = {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Random Forest (target only)',
      x: d.sizes,
      y: d.rf,
      line: { color: PALETTE.teal, width: 3, shape: 'spline', dash: 'solid' },
      marker: { size: 11, color: PALETTE.teal, line: { color: PALETTE.bg, width: 2 } },
      hovertemplate: '<b>Random Forest</b><br>Target n = %{x}<br>Macro F1 = %{y:.3f}<extra></extra>'
    };

    const layout = Object.assign({}, LAYOUT_BASE, {
      height: 440,
      margin: { l: 70, r: 40, t: 24, b: 60 },
      xaxis: Object.assign({}, LAYOUT_BASE.xaxis, {
        title: 'Number of target (PLN) training samples',
        tickvals: d.sizes,
        ticktext: d.sizes.map(s => 'n = ' + s)
      }),
      yaxis: Object.assign({}, LAYOUT_BASE.yaxis, {
        title: 'Macro F1 score',
        range: [0.6, 0.88]
      }),
      legend: {
        orientation: 'h',
        x: 0.5, xanchor: 'center', y: 1.06,
        font: { size: 12 }
      },
      shapes: [{
        type: 'rect',
        x0: 70, x1: 80,
        y0: 0.6, y1: 0.88,
        fillcolor: PALETTE.accent,
        opacity: 0.06,
        line: { width: 0 },
        layer: 'below'
      }],
      annotations: [{
        x: 75, y: 0.766,
        ax: 0, ay: -50,
        xref: 'x', yref: 'y', axref: 'pixel', ayref: 'pixel',
        arrowhead: 2, arrowsize: 1, arrowwidth: 1, arrowcolor: PALETTE.ink,
        text: '<b>+9.0 pp</b> gap at n = 75',
        font: { color: PALETTE.ink, size: 12, family: FONT.family },
        bgcolor: PALETTE.bg,
        bordercolor: PALETTE.ink,
        borderwidth: 1,
        borderpad: 6
      }]
    });

    Plotly.newPlot('chart-scarcity', [traceTL, traceRF], layout, CONFIG);
  }

  /* ─────────────────────────────────────────────────────────────
     4. ANOVA feature ranking
     ───────────────────────────────────────────────────────────── */
  function renderAnova() {
    const data = window.DGA_DATA.anova.slice().sort((a, b) => a.f - b.f);

    const trace = {
      type: 'bar',
      orientation: 'h',
      x: data.map(d => d.f),
      y: data.map(d => d.feature),
      marker: {
        color: data.map(d => d.sig ? PALETTE.teal : PALETTE.muted),
        line: { color: 'transparent', width: 0 }
      },
      customdata: data.map(d => [d.p.toExponential(2), d.sig ? 'Significant' : 'Not significant']),
      hovertemplate:
        '<b>%{y}</b><br>' +
        'F-score: %{x:.2f}<br>' +
        'p-value: %{customdata[0]}<br>' +
        '%{customdata[1]}<extra></extra>',
      text: data.map(d => d.f.toFixed(1)),
      textposition: 'outside',
      textfont: { color: PALETTE.ink, family: FONT.family, size: 12 },
      cliponaxis: false
    };

    const layout = Object.assign({}, LAYOUT_BASE, {
      height: 460,
      margin: { l: 130, r: 60, t: 16, b: 50 },
      xaxis: Object.assign({}, LAYOUT_BASE.xaxis, {
        title: 'ANOVA F-score (higher = more useful for fault discrimination)',
        range: [0, Math.max(...data.map(d => d.f)) * 1.15]
      }),
      yaxis: Object.assign({}, LAYOUT_BASE.yaxis, {
        automargin: true,
        tickfont: { color: PALETTE.ink, size: 12, family: FONT.family }
      })
    });

    Plotly.newPlot('chart-anova', [trace], layout, CONFIG);
  }

  /* ─────────────────────────────────────────────────────────────
     Public
     ───────────────────────────────────────────────────────────── */
  window.DGA_CHARTS = {
    renderAll: function () {
      renderKS();
      renderMethods(7);
      renderScarcity();
      renderAnova();
    },
    renderMethods: renderMethods
  };
})();
