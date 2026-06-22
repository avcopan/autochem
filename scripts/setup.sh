#!/usr/bin/env bash

# Set up direnv, if present
if command -v direnv >/dev/null 2>&1; then
  direnv allow
else
  echo "Warning: direnv is not installed."
  echo "You can install direnv and then run: direnv allow"
fi

# Initialize git repository
git init
git add --all
if git config --get user.name >/dev/null 2>&1 && git config --get user.email >/dev/null 2>&1; then
  git commit -m "Initial commit"
else
  echo "Warning: Git author identity is not configured."
  echo "You can configure it as follows:"
  echo '  git config --global user.name "Your Name"'
  echo '  git config --global user.email "you@example.com"'
  echo "Afterwards, you can make the initial commit with:"
  echo '  git commit -m "Initial commit"'
fi

if git show-ref --verify --quiet refs/heads/template; then
  echo "Branch 'template' already exists"
else
  git branch template
fi

# Install git hooks with lefthook
lefthook install
