#!/bin/bash
#SBATCH --job-name=python_test_job       # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --output=text_output_unet.txt     # Output file
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --time=48:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mem=64G                        # Memory allocation (64 GB)
#SBATCH --gres=gpu:1                     # Request 1 GPU (adjust based on availability)
#SBATCH --cpus-per-task=16               # Number of CPU cores per task

# Load necessary modules (adjust or uncomment based on your environment)
# module load python  # Uncomment if system Python is needed

# Activate the virtual environment
source /home/RSGIS-st121616/tony_env/bin/activate  # Adjust the path if needed

# Email notifications (ensure emails are separated by commas, no spaces)
#SBATCH --mail-user=akkarapon.chaiyana@gmail.com,67130749@dpu.ac.th
#SBATCH --mail-type=END,FAIL                # Send email when job ends or fails

# Run the Python script using SLURM's srun
srun python unet_gpu.py

# Deactivate virtual environment (optional in most cases)
deactivate
