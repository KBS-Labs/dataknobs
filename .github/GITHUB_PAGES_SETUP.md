# GitHub Pages Setup for MkDocs

## The Issue

GitHub Pages was trying to build the site with Jekyll (default behavior), which conflicts with our MkDocs setup. The Jekyll build fails because:
- Our docs contain Jinja2 template syntax (e.g., `{% elif %}`)
- Jekyll uses Liquid templating and doesn't understand Jinja2
- Jekyll tries to process the source `docs/` folder before our custom workflow runs

## The Solution

### 1. Disable Jekyll with `.nojekyll` file

We added `.nojekyll` to the repository root to tell GitHub Pages:
- ❌ Don't run automatic Jekyll builds
- ✅ Use our custom GitHub Actions workflow instead

### 2. Verify GitHub Pages Settings

**IMPORTANT**: Check your repository settings to ensure the correct deployment source:

1. Go to your GitHub repository
2. Navigate to **Settings** → **Pages**
3. Under **Build and deployment** → **Source**, select:
   - ✅ **GitHub Actions** (correct)
   - ❌ NOT "Deploy from a branch" (would trigger Jekyll)

### 3. Our Custom Workflow

Our `.github/workflows/docs.yml` handles everything:
- ✅ Builds docs with MkDocs (not Jekyll)
- ✅ Adds `.nojekyll` to the built site
- ✅ Uploads the artifact
- ✅ Deploys to GitHub Pages

## How It Works

```mermaid
graph LR
    A[Push to main] --> B[docs.yml workflow]
    B --> C[Install Python + deps]
    C --> D[mkdocs build]
    D --> E[Add .nojekyll to site/]
    E --> F[Upload artifact]
    F --> G[Deploy to GitHub Pages]
    G --> H[Site live!]
```

## Why Two `.nojekyll` Files?

1. **Repository root** (`.nojekyll`):
   - Prevents GitHub from auto-building source with Jekyll
   - Tells GitHub: "I'm using a custom workflow"

2. **Built site** (`site/.nojekyll`):
   - Created during workflow build step
   - Ensures deployed site isn't processed with Jekyll
   - Redundant but safe practice

## Testing the Fix

After pushing `.nojekyll` to the repository:

1. Push changes to trigger the docs workflow
2. Check Actions tab for workflow run
3. Verify the "Build with Jekyll" error is gone
4. Check that docs deploy successfully

## Troubleshooting

### Jekyll Still Running?

If you still see Jekyll errors after adding `.nojekyll`:

1. **Check GitHub Pages settings**:
   - Settings → Pages → Source must be "GitHub Actions"

2. **Clear GitHub cache**:
   - Sometimes GitHub caches the old build method
   - Try a force push or wait a few minutes

3. **Check `.nojekyll` is committed**:
   ```bash
   git ls-files .nojekyll
   # Should show: .nojekyll
   ```

### Workflow Fails?

If the docs.yml workflow itself fails:

1. Check the Actions tab for error details
2. Look for mkdocs build errors
3. Verify all packages are installed (see workflow file)

## References

- [GitHub Pages Custom Workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [Bypassing Jekyll on GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages#static-site-generators)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
