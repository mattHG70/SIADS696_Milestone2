#!/bin/bash
# Slurm script to run the transfer learning job.
# Uses one node + one gpu
# Slurm PARAMETERS
#SBATCH --job-name=team10_cv02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2g
#SBATCH --account=siads696s24_class
#SBATCH --mail-user mhuebsch@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/siads696/%x-%j.log
#SBATCH --error=/home/%u/siads696/error-%x-%j.log

# run bashrc
source /home/mhuebsch/.bashrc
DATA_DIR=/home/mhuebsch/siads696/data
# switch to directory containing all scripts
cd /home/mhuebsch/SIADS696_Milestone2/scripts/supervised
# load a python+conda module if necessary
# module load <ptyhon module>
# module load cuda cudnn
# activate custom conda environment
conda activate ms2

# some fancy logging
START=`date +%s`; STARTDATE=`date`;
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] Starting the workflow
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] We got the following cores: $CUDA_VISIBLE_DEVICES
python ms2_crossval_script_2.py -infile $DATA_DIR/full_train_moa.parquet -outfile $DATA_DIR/nn_Net3072_cv_results_03.parquet

EXITCODE=$?
# some fancy logging
END=`date +%s`; ENDDATE=`date`
echo [INFO] [$END] [$ENDDATE] [$$] [$JOB_ID] Workflow finished with code $EXITCODE
echo [INFO] [$END] [`date`] [$$] [$JOB_ID] Workflow execution time \(seconds\) : $(( $END-$START ))
