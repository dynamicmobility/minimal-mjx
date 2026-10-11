# Developer guide

## Tests

```bash
pip install -e ".[dev]"
pytest            # fast suite (~15 s, CPU)
pytest -m slow    # rendering, video, and a tiny end-to-end PPO run
```

GitHub Actions runs the fast suite on every push to `main` and on every pull request
(`.github/workflows/tests.yml`).

## Building these docs

The site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. The API reference is
generated from docstrings by [mkdocstrings](https://mkdocstrings.github.io/).

```bash
pip install -e ".[docs]"
mkdocs serve           # live preview at http://127.0.0.1:8000
mkdocs build --strict  # what CI runs; fails on broken links and references
```

The configuration lives in `mkdocs.yml` at the repository root, and the pages in
`docs/`:

```text
docs/
├── index.md              # home page
├── examples/             # step-by-step walkthroughs of example/*.py
├── guides/               # the config file, writing an environment
├── api/                  # one page per subpackage, generated from docstrings
├── developer.md          # this page
└── assets/               # plots and videos shown on the site
```

`.github/workflows/docs.yml` builds the site on every pull request and publishes it to
GitHub Pages on every push to `main`.

### Writing docstrings

The API pages pick up every public function, class and method automatically: a new
function appears as soon as it exists in a module the reference covers. Underscored
names are left out. Docstrings use the
[Google style](https://mkdocstrings.github.io/griffe/reference/docstrings/#google-style):

```python
def load_policy(run_dir, deterministic=True, checkpoint_path=None):
    """Load a policy from a checkpoint.

    Args:
        run_dir: Run directory to take the newest checkpoint from.
        deterministic: If True, the policy returns the mean action.
        checkpoint_path: A specific checkpoint to load instead.

    Returns:
        A function ``(obs, rng) -> (action, extras)``.
    """
```

To link to an API entry from any page, use its full path as a Markdown reference:
`` [`train`][minimal_mjx.learning.training.train] ``.

A new module needs a `::: minimal_mjx.<subpackage>.<module>` line on its subpackage's
page in `docs/api/`.

### Keeping the examples in sync

The step-by-step pages quote excerpts of `example/train.py` and `example/rollout.py`.
The full listing at the bottom of each page is included straight from the file, so it
never goes stale, but the excerpts and their explanations need updating by hand when
the scripts change.

The plot and video in `docs/assets/` come from the run in
`example/results/cheetah/test`. To refresh them after a new run:

```bash
ffmpeg -i example/results/cheetah/test/rollout.mp4 -vf "scale=540:540,fps=50" \
    -c:v libx264 -crf 28 -pix_fmt yuv420p -movflags +faststart -an docs/assets/cheetah-rollout.mp4
ffmpeg -ss 4 -i docs/assets/cheetah-rollout.mp4 -frames:v 1 docs/assets/cheetah-rollout-poster.jpg
python docs/scripts/plot_progress.py example/results/cheetah/test/progress.csv docs/assets/cheetah-progress.png
```
