#!/usr/bin/env bash

# Set up remote repository
read -p "Press enter to continue if you have already set up a remote repository for autochem on GitHub. Otherwise, please set it up before continuing."

DEFAULT_USERNAME=$(git config --global user.name)
read -p "  Git username (${DEFAULT_USERNAME} [default] or enter alternative): " INPUT
USERNAME=${INPUT:-$DEFAULT_USERNAME}
git remote add origin git@github.com:$USERNAME/autochem.git
git push -u origin main
git push -u origin template:template
