python FinalPaper\grid_search_normalization_topology.py --aggregate-only --output-dir FinalPaper\results\normalization_grid # Aggregate files in results from multiple parallel runs


python FinalPaper\analyze_normalization_grid.py --results-dir FinalPaper\results\normalization_grid --output-dir FinalPaper\results\normalization_grid_analysis   # Analyzes files in FinalPaper/results/normalization_grid


Launch parallel grid jobs in separate PowerShell windows:

powershell -ExecutionPolicy Bypass -File FinalPaper\run_scipts\launch_grid_jobs.ps1 -NumJobs 4 -OutputDir FinalPaper\results\normalization_grid -GridArgs "--j-values 0.1 0.2 0.4 0.75 1.0 1.25 --g-values 0.6 1.0 1.5 2.0 3.0 5.0 --input-amplitudes 0.0 0.02 0.05 0.1 0.2 0.5 1.0 3.0 --seed-count 5"

After all launched jobs finish, run the generated aggregate script printed by the launcher, for example:

powershell -ExecutionPolicy Bypass -File FinalPaper\run_scipts\generated\grid_YYYYMMDD_HHMMSS_aggregate_and_analyze.ps1

Useful launcher options:
-NumJobs 4 sets --job-index 0..3 and --num-jobs 4 automatically.
-BlasThreads 1 sets OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, and MKL_NUM_THREADS in every job.
-NoExit keeps each job window open after completion.
-DryRun generates scripts without launching them.


Add new values to an existing completed/partial grid without rerunning finished conditions:

powershell -ExecutionPolicy Bypass -File FinalPaper\run_scipts\launch_grid_jobs.ps1 -NumJobs 8 -OutputDir FinalPaper\results\normalization_grid -GridArgs "--augment-existing --add-j-values 0.6 --add-g-values 0.4"

This reads grid_config_job*.json from the output folder, expands the old axes with the new values, and skips condition JSON files that are already marked status=ok. Adding both a new J and a new g expands the full rectangular grid, so it runs the new J row and the new g column.



python FinalPaper\grid_search_normalization_topology.py --sweep b-target-normalization --x-values 0.5 1.0 1.5 --y-values 0.5 1.0 2.0

python FinalPaper\grid_search_normalization_topology.py --sweep delay --topologies fixed spatial --fixed-e-delay-values 5 10 15 --fixed-i-delay-values 1 1.5 2 --spatial-e-lambda-values 0.1 0.2 0.4 --spatial-i-lambda-values 0.05 0.1 0.2 --delay-std-fraction 0.3

For delay sweeps:
- fixed topology uses E/I delay means in ms, with std = --delay-std-fraction * mean.
- spatial topology uses E/I lambda_by_preclass values, not ms delays.
- --x-values/--y-values still work as a generic fallback, but the topology-specific delay/lambda flags are preferred when sweeping both topologies.

python FinalPaper\grid_search_normalization_topology.py --sweep threshold-heterogeneity --x-values 0 0.5 1 3 7 --y-values 0 0.5 1 3 7

python FinalPaper\grid_search_normalization_topology.py --sweep size-inhibition --x-values 500 1000 1500 --y-values 0.1 0.2 0.3
