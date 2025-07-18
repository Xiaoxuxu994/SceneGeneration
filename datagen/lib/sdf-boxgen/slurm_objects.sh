#!/bin/bash

#SBATCH --job-name voxelize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=0
##SBATCH --mem-per-cpu=12G
#SBATCH --exclude=char,pegasus,seti,sorona,tarsonis,umoja

cd /rhome/ysiddiqui/sdf-gen
python process_objects.py --mesh_dir /cluster/moria/ysiddiqui/3DFrontMeshesNoStructs/ --df_lowres_dir /cluster/moria/ysiddiqui/3DFrontDistanceFieldsObjects/complete_lowres --df_highres_dir /cluster/moria/ysiddiqui/3DFrontDistanceFieldsObjects/complete_highres --df_if_dir /cluster/moria/ysiddiqui/3DFrontDistanceFieldsObjects/complete_semantics --chunk_lowres_dir /cluster/moria/ysiddiqui/3DFrontDistanceFieldsObjects/chunk_lowres --chunk_highres_dir /cluster/moria/ysiddiqui/3DFrontDistanceFieldsObjects/chunk_highres --chunk_if_dir /cluster/moria/ysiddiqui/3DFrontDistanceFieldsObjects/chunk_semantics --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID