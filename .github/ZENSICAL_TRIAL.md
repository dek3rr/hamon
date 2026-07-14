# Zensical trial — compatibility status

We plan to eventually move the docs from **MkDocs + Material for MkDocs** to
[**Zensical**](https://zensical.org) (their consolidated successor). Zensical is
pre-1.0 and still building toward feature parity, so we are **not** cutting over
yet. Instead:

- **Live site is still built and deployed by MkDocs** — see
  [`docs.yml`](workflows/docs.yml). Nothing about the live docs changes.
- **A non-deploying trial job builds the same `mkdocs.yml` with Zensical** — see
  [`docs-zensical-trial.yml`](workflows/docs-zensical-trial.yml). It uploads the
  rendered `site/` + build log as an artifact on docs PRs and on manual dispatch,
  so we can watch Zensical's output on *our* content as it matures.

Run it locally the same way CI does:

```bash
pip install .[docs,docs-zensical]
zensical build --clean      # reads mkdocs.yml; writes ./site
zensical serve              # live preview at localhost:8000
```

## Status as of Zensical 0.0.50 (2026-07-13)

Trial build against our unmodified `mkdocs.yml` **exits 0 in ~4s**. Our docs are
~25 pages: 4 hand-written Markdown, 6 Jupyter notebooks, 15 mkdocstrings API pages.

### Works today ✅

- **Hand-written Markdown** — `index`, `getting-started`, `concepts`,
  `architecture` all render.
- **Math** — `pymdownx.arithmatex` (generic) + MathJax via `extra_javascript`
  render fine.
- **API reference (mkdocstrings)** — all 15 `api/*.md` pages render with full
  docstrings, signatures, and parameters. **Internal cross-references resolve**
  (e.g. ~164 `#hamon…` autoref links on the NRPT page) — better than Zensical's
  docs imply; the "backlinks" limitation is narrower than "all cross-refs".
- **Theme** — `material` theme, `slate` palette + custom primary, Inter/JetBrains
  Mono fonts, search, code-copy.

### Broken / degraded today ❌ — cutover blockers

1. **Notebook examples (6 pages) do not render.** Two compounding causes:
   - `docs/examples` is a **symlink** to `../examples`; Zensical copies it
     verbatim as an 11-byte text file instead of following it.
   - Zensical has **no `.ipynb` → HTML renderer** yet (the `mkdocs-ipynb`
     equivalent is on their backlog: zensical/zensical#96).
   The nav still lists all 6 notebook entries, so they render as **dangling
   links**. This is the single biggest gap.
2. **External API cross-references are lost.** `hippogriffe` (which resolves refs
   to `jax` / `equinox` / `numpy` types via `extra_public_objects`) is not
   supported — 0 external cross-links in the Zensical output. Those types render
   as plain text.
3. **`_overrides/` and `_static/` leak into the site root** as raw copied
   directories. Low priority and partly pre-existing (no `theme.custom_dir` is
   wired in `mkdocs.yml`), but worth cleaning up before any cutover.

> The `griffe: …` warnings in the build log are docstring-lint notices from
> introspection — they also appear under the current MkDocs build and are not
> Zensical-specific.

## Cutover checklist

Revisit a full migration once these clear (roughly Zensical Phase 3):

- [ ] Zensical renders `.ipynb` pages **or** we convert the 6 notebooks to Markdown.
- [ ] `docs/examples` symlink is followed **or** replaced with real files.
- [ ] External cross-references reach `hippogriffe` parity (or an accepted downgrade).
- [ ] Internal API cross-refs remain complete after any mkdocstrings changes.
- [ ] `custom_dir` / partial overrides (`_overrides/partials/source.html`) confirmed.
- [ ] `include_exclude_files` behavior (`.htaccess` inclusion, artifact exclusion) covered.

When all boxed items hold, swap `docs.yml` to `zensical build` + deploy and retire
the MkDocs config.
