# update version in src/muse_xai/__init__.py
git add src/muse_xai/__init__.py
git commit -m "Bump package version to 0.1.0"
git tag -a v0.1.0 -m "v0.1.0 release"
git push origin main --tags
