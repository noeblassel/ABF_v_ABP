#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sweep the ABF & ABP integrators over effective dimension d and tempering alpha.
# Data is tagged by (alpha,d) in sim_results/, so every run coexists; make_plots.py
# then renders all of them.  ABF and ABP for every combo run as a bounded pool of
# parallel processes.
#
# Override anything via the environment, e.g.:
#     NX=200 T=8 NJOBS=6 ./run_sweep.sh                 # finer / more parallel
#     SWEEP_D="2 6 10" SWEEP_ALPHA="1 4" ./run_sweep.sh # a sub-grid
#     RENDER=0 ./run_sweep.sh                           # simulate only, don't render
# ---------------------------------------------------------------------------
cd "$(dirname "$0")"

# --- sweep grid (space-separated; override with SWEEP_D / SWEEP_ALPHA) ------
DS=(${SWEEP_D:-2 3 4 5 6 7 8 9 10})
ALPHAS=(${SWEEP_ALPHA:-0.125 0.25 0.5 1 2 4 8 16})

# --- solver settings (coarser/shorter than production; override via env) ----
NX=${NX:-150}; DT=${DT:-0.005}; T=${T:-5}; STRIDE=${STRIDE:-1}; S0=${S0:-0.05}

# --- parallelism: NJOBS processes, each OMP_NUM_THREADS threads -------------
NJOBS=${NJOBS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

mkdir -p sim_results
total=$(( ${#DS[@]} * ${#ALPHAS[@]} * 2 )); n=0
echo "sweep: ${#DS[@]} d x ${#ALPHAS[@]} alpha x 2 methods = $total runs"
echo "  nx=$NX dt=$DT T=$T stride=$STRIDE s0=$S0 | $NJOBS jobs x $OMP_NUM_THREADS threads"

for a in "${ALPHAS[@]}"; do
  for d in "${DS[@]}"; do
    for m in abf abp; do
      while (( $(jobs -rp | wc -l) >= NJOBS )); do wait -n; done   # throttle
      n=$((n + 1)); echo "[$n/$total] $m  alpha=$a  d=$d"
      FreeFem++ -nw "$m.edp" -nx "$NX" -dt "$DT" -T "$T" -stride "$STRIDE" -s0 "$S0" \
                -alphaTemp "$a" -d "$d" > "sim_results/log_${m}_a${a}_d${d}.out" 2>&1 &
    done
  done
done
wait
echo "simulations done ($total runs)."

if [[ "${RENDER:-1}" == 1 ]]; then
  echo "rendering all runs..."
  python3 make_plots.py --fps "${FPS:-10}"
else
  echo "skipping render (RENDER=0). Run:  python3 make_plots.py --fps 15"
fi
