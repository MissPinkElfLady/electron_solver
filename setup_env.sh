#!/usr/bin/env bash
# Set up the tool chain on Debian/Ubuntu. Run from the repository root.
set -euo pipefail

# FreeFEM: the "freefem++" package only has the executables; the 3D mesh plugins
# (msh3, gmsh, tetgen, medit) are in "libfreefem++". gmsh's python wheel needs a few X libraries.
sudo apt-get install -y --no-install-recommends freefem++ libfreefem++ \
    libxft2 libxinerama1 libxcursor1 libglu1-mesa libxrender1 libxfixes3

python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
mkdir -p external
[ -d external/zeroheliumkit ] || git clone https://github.com/eeroqlab/zeroheliumkit external/zeroheliumkit
[ -d external/quantum_electron ] || git clone https://github.com/gkoolstra/quantum_electron external/quantum_electron
pip install -r external/zeroheliumkit/requirements.txt
pip install -e external/zeroheliumkit -e external/quantum_electron

echo
echo "Done. Before running:  . .venv/bin/activate && export FF_LOADPATH=/usr/lib/freefem++"
