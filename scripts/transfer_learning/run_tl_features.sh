#!/bin/bash
# UGE PARAMETERS
#$ -N slc
#$ -pe smp 1
#$ -binding linear:1
#$ -cwd
#$ -S /bin/bash
#$ -l m_mem_free=64G
#$ -l h_rt=24:00:00
#$ -l gpu_card=1
#$ -j y

# UGE PARAMETERS END
# run bashrc
source /home/ghisuma1/.bashrc
cd /da/isld/ghisuma1/shared/cm_pheno/jlifesci_cm
conda activate cm_pheno_v2

# some fancy logging
START=`date +%s`; STARTDATE=`date`;
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] Starting the workflow
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] We got the following cores: $CUDA_VISIBLE_DEVICES
python create_tl_features.py -dev cuda -image_list linux_imageList -channels "ACTIN,ALPHA_ACTININ,NUCLEI"

EXITCODE=$?
# some fancy logging
END=`date +%s`; ENDDATE=`date`
echo [INFO] [$END] [$ENDDATE] [$$] [$JOB_ID] Workflow finished with code $EXITCODE
echo [INFO] [$END] [`date`] [$$] [$JOB_ID] Workflow execution time \(seconds\) : $(( $END-$START ))