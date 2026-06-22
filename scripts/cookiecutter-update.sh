#!/usr/bin/env bash

set -e

original_branch=$(git branch --show-current)

# Switch to template branch and clear all files
git switch template
git pull origin template
git rm -rf .
git clean -fd
git checkout HEAD -- .template.json

# Update template using cookiecutter
yes | cookiecutter gh:avcopan/cookiecutter-pixi-python --replay-file .template.json
rsync -av --remove-source-files autochem/ ./
rm -r autochem/

# Commit and push the changes to the template branch
git add --all
git commit --no-verify -m "Update template"
git push origin template
commit_hash=$(git rev-parse --short HEAD)

# Update the current branch by cherry-picking the template update commit
git switch "$original_branch"
git cherry-pick "$commit_hash" -x
