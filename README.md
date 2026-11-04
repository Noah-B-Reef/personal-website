# Noah Reef's personal website

The site uses **Typst + Tola 0.8.1** and the [NASApunk theme](https://github.com/KKKZOZ/tola-theme-nasapunk), replacing Hugo. Production lives at <https://noah-reef.net/>.

## Run locally

Use Node.js 22 and Tola 0.8.1 (which embeds Typst 0.15.1). Install Tola from its [official release](https://github.com/tola-rs/tola-ssg/releases/tag/v0.8.1), or with Rust:

```sh
cargo install --locked tola --version 0.8.1
npm ci
npm run preview
```

Open **http://127.0.0.1:5277/**. A checksum-verified Tola 0.8.1 binary is already installed in `.tools/bin/tola` in this working copy. The npm commands prefer that local binary and otherwise use `tola` on PATH. `.tools/` is intentionally not committed.

Commands:

- `npm run dev`: live reload while writing; search indexing is skipped.
- `npm run preview`: build and serve with working search.
- `npm run build`: production output in `public/`, including search.
- `npm run check`: compile and validate references without publishing output.
- `npm run verify`: check a completed build's local links, retained routes, feeds, and migrated equation counts.
- `npm test`: theme math-copy tests; requires the separate Typst 0.15.1 CLI and downloads its pinned MiTeX test package on first run.

## Edit the site

- `content/about.typ`: biography, education, research interests, and contact links.
- `content/blog/*.typ`: posts. Keep filenames stable to preserve public URLs.
- `tola.toml`: title, author, description, origin, deployment base path, and build settings.
- `templates/config.typ`: document prefix, social links, and theme settings.
- `templates/layout.typ`: navigation and page chrome.
- `assets/styles/site.css`: theme styling.
- `static/resume.pdf` and `static/images/`: résumé and retained images.

New posts follow this format:

```typst
#import "@tola/source:0.0.0": tola-meta

#tola-meta((
  title: "Post title",
  date: "2026-11-04",
  summary: "A short description.",
  tags: ("research",),
))

= Introduction

Write in Typst. Inline mathematics: $x^2$. Display mathematics:

$ sum_(i=1)^n x_i $
```

Use Tola's `asset-url` for published assets so links include the deployment base path:

```typst
#import "@tola/address:0.0.0": asset-url
#link(asset-url("/resume.pdf"))[Curriculum Vitae]
```

For Typst `image(...)`, use the actual source path, such as `/static/images/perceptron.png`.

## Deployment

`.github/workflows/pages.yml` builds and checks pull requests. Pushes to `main` and manual runs on `main` also deploy the resulting `public/` artifact through GitHub Pages. The workflow pins Tola 0.8.1 and Typst 0.15.1, verifies the Tola release checksum, and uses the npm lockfile. Repository Pages settings should use **GitHub Actions** (as with the previous workflow).

The custom-domain origin and `/` base path are configured in `tola.toml`. Change both when moving to another domain or repository. Publishing changes to `main` triggers the deployment workflow.

## Migration and theme provenance

Vendored NASApunk revision: `0492ef3ce63c6baf025b86ff4f8bda829f92fddb`. Theme sources are in `templates/`, `utils/`, `assets/`, and `typst-fonts/`; its MIT license is in `LICENSE-NASAPUNK`, and bundled fonts retain their licenses. Theme changes here include Noah's identity and social links, `/blog/` post discovery/navigation, and legacy RSS outputs. The npm lockfile uses the official npm registry.

The About page and all three posts were converted to Typst. The blog slugs, individual tag pages, résumé URL, and `/index.xml` and `/blog/index.xml` RSS URLs are retained; `/feed.xml` is also available. The 139 source equations are rendered as SVG with copyable LaTeX. The decision-tree source's invalid `s^\*` was corrected to a star superscript during conversion. The MLP figure and Python examples are retained.

Original Markdown, Hugo configuration, layouts, and archetypes are preserved under `legacy/hugo/`. The old `themes/hugo-xterm` submodule and `static/` remain in place, including pre-existing local modifications. The original portrait was restored for the About page; the unused photo.svg deletion remains a local change. These files are not used by Tola. The previous generated `public/` output was moved to `/tmp/noah-hugo-public-backup` locally because Tola refuses to overwrite output it does not own.

For a temporary Hugo preview of the backup:

```sh
hugo --source legacy/hugo --themesDir ../../themes --staticDir ../../static --destination /tmp/noah-hugo-preview
```

The theme uses Typst's experimental HTML export. Review a local preview before publishing content with new layout features.
