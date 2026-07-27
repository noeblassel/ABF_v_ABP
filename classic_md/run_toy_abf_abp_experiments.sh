#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
python_bin="/home/jhimbert/miniconda3/envs/ML4MD-py311/bin/python"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$script_dir/.mplconfig}"

cd "$repo_root"
"$python_bin" classic_md/toy_abf_abp_experiments.py "$@"
