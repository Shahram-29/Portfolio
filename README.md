# Shahram Sajawal — Portfolio

Single-file portfolio site (`index.html`) — no build step, no dependencies beyond Google Fonts.

## Files
- `index.html` — the whole site (HTML + CSS + JS)
- `resume.pdf` — linked from the "Résumé" button in the hero
- `README.md` — this file

## Edit
Everything is plain HTML. Search for these to update content quickly:
- `<!-- Replace these two with your real profile URLs -->` — LinkedIn / GitHub buttons in the Contact section
- `<section id="projects"` — project cards; change each card's `href` to the repo or report link
- `.pip` spans inside `.player` — the orbiting chips in the hero
- `.ticker__row` — the scrolling tool strip (list is repeated twice so the loop is seamless)
- `data-count="…"` — animated numbers in the hero stat tiles

## Publish on GitHub Pages
1. Create a new public repo on GitHub named `Portfolio` (or `<username>.github.io` for a root URL).
2. From this folder:
   ```bash
   git init
   git add .
   git commit -m "Portfolio site"
   git branch -M main
   git remote add origin https://github.com/Shahram-29/Portfolio.git
   git push -u origin main
   ```
3. In the repo: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main` / `(root)` → Save**.
4. After a minute the site is live at `https://shahram-29.github.io/Portfolio/`.
