#!/bin/sh

set -eu

python_bin="/home/jhimbert/miniconda3/envs/ML4MD-py311/bin/python"

d=10.0
num_trajectories=22
num_steps=1000000
dt=0.0001
alpha=10.0
theta=60.0
burn_in=1.0
save_stride=100

make_gifs=1
gif_fps=12
gif_frame_stride=10
gif_max_particles=32
gif_trail_length=8
gif_grid_size=180

abf_output_path="classic_md/results/abf_d_${d}_alpha_${alpha}_theta_${theta}_dt_${dt}_burnin_${burn_in}_estimator_running_mean.npz"
abp_output_path="classic_md/results/abp_d_${d}_alpha_${alpha}_theta_${theta}_dt_${dt}_burnin_${burn_in}_estimator_running_mean.npz"
abf_gif_path="${abf_output_path%.npz}.gif"
abp_gif_path="${abp_output_path%.npz}.gif"
plot_output_dir="classic_md/plots/d_${d}_alpha_${alpha}_theta_${theta}_dt_${dt}_burnin_${burn_in}_estimator_running_mean"

"$python_bin" classic_md/jax_abf_abp.py \
  --method abf \
  --num-trajectories "$num_trajectories" \
  --num-steps "$num_steps" \
  --dt "$dt" \
  --alpha "$alpha" \
  --theta "$theta" \
  --burn-in "$burn_in" \
  --save-stride "$save_stride" \
  --output "$abf_output_path" \
  $(if [ "$make_gifs" -eq 1 ]; then
      printf '%s ' \
        --make-gif \
        --gif-output "$abf_gif_path" \
        --gif-fps "$gif_fps" \
        --gif-frame-stride "$gif_frame_stride" \
        --gif-max-particles "$gif_max_particles" \
        --gif-trail-length "$gif_trail_length" \
        --gif-grid-size "$gif_grid_size"
    fi)

"$python_bin" classic_md/jax_abf_abp.py \
  --method abp \
  --num-trajectories "$num_trajectories" \
  --num-steps "$num_steps" \
  --dt "$dt" \
  --alpha "$alpha" \
  --theta "$theta" \
  --burn-in "$burn_in" \
  --save-stride "$save_stride" \
  --output "$abp_output_path" \
  $(if [ "$make_gifs" -eq 1 ]; then
      printf '%s ' \
        --make-gif \
        --gif-output "$abp_gif_path" \
        --gif-fps "$gif_fps" \
        --gif-frame-stride "$gif_frame_stride" \
        --gif-max-particles "$gif_max_particles" \
        --gif-trail-length "$gif_trail_length" \
        --gif-grid-size "$gif_grid_size"
    fi)

echo "Plotting results..."
"$python_bin" classic_md/plot_results.py \
  "$abf_output_path" \
  "$abp_output_path" \
  --output-dir "$plot_output_dir"
