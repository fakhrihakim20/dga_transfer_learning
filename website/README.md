# Tropical DGA Transfer Learning — Report Website

A single-page report website for the ICT-PEP 2026 paper *"Tropical Domain Shift in Dissolved Gas Analysis: Transfer Learning from the IEC TC 10 Benchmark to PLN Indonesia."*

## Local preview

No build step. Just open `index.html` in a browser, **or** start a tiny static server (recommended so fonts and Plotly load from the CDN cleanly):

```bash
# from this directory
python -m http.server 8080
# then visit http://localhost:8080
```

## Stack

- Vanilla HTML + CSS + JS (no framework, no bundler)
- [Plotly.js 2.35](https://plotly.com/javascript/) via CDN for interactive charts
- Google Fonts (Fraunces, Inter, JetBrains Mono)

## File map

```
website/
├── index.html              · one-page report
├── assets/
│   ├── css/styles.css      · all styling
│   ├── js/data.js          · CSV-derived JSON
│   ├── js/charts.js        · Plotly chart factories
│   ├── js/app.js           · nav, scroll reveal, lightbox, tabs
│   ├── figures/            · 8 PNG figures (copies of ../results/figures)
│   ├── data/               · 4 CSV tables (copies of ../results/tables)
│   └── paper/              · paper PDF
└── .nojekyll               · so GitHub Pages serves _ paths
```

## Deployment

Automatic via GitHub Actions — see `.github/workflows/deploy-pages.yml` in the repo root. Pushing to `main` publishes this directory to GitHub Pages.

Live URL: <https://fakhrihakim20.github.io/dga_transfer_learning/>
