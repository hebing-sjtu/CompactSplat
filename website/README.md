# CompactSplat project website

A static academic project page, following SurfSplat's title/authors/resources → teaser → abstract → overview → results → citation structure. The page uses original CompactSplat manuscript figures, with keyboard-accessible resolution switching and citation copying. No build or external JavaScript dependencies are needed.

## Run locally

From the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory website
```

Open [the local page](http://127.0.0.1:8765/). Files are served only on the local machine.

## Edit

- `index.html`: content, author metadata, native-resolution result table and citation.
- `styles.css`: responsive academic layout.
- `script.js`: resolution-specific manuscript values and copy interaction.
- `assets/`: manuscript figure renderings; provenance is in [SOURCES.md](assets/SOURCES.md).

The native-resolution table in HTML remains readable with JavaScript disabled. When changing the results, keep its values synchronized with `script.js`; retain protocol and provenance notes. Bold values are computed among the displayed rows, including ties.

The result tables report paper values. See the [reproducibility notes](../docs/REPRODUCIBILITY.md) for implementation differences and the [model status](../MODEL_ZOO.md) for checkpoint availability.

## Hosting

This source has no automatic deployment workflow. The included `noindex` metadata and `robots.txt` discourage indexing; they do not authenticate visitors. Configure access control on the hosting service when deploying privately.
