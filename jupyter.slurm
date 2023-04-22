#!/bin/bash

#SBATCH --job-name=jupyter
## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2023sp_14

## change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

##SBATCH --partition=n1c24m128-v100-4
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=24

#SBATCH --time=1:00:00
#SBATCH --output=logs/jupyter_%j.out
#SBATCH --exclusive
#SBATCH --requeue

port=$(shuf -i 10000-65500 -n 1)

opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -N -f -R $port:localhost:$port"

for((i=1; i<=3; i++)); do
    /usr/bin/ssh $opts $USER@log-$i.nyu.cluster
done

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to greene with command

ssh $USER@greene

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on greene compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL23SP/dataset_v2.sqsh /tmp
echo "Dataset is copied to /tmp"

singularity exec $nv \
	    --bind /scratch \
	    --overlay /tmp/dataset_v2.sqsh \
/share/apps/images/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
/bin/bash -c "
jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
"
