# Collision Prediction

## Setup

1. Download and connect Cisco VPN
2. ssh `NetID`@greene.hpc.nyu.edu 
3. ssh burst
4.  srun --partition=interactive --account csci_ga_2572_2023sp_14 --pty /bin/bash 
5. cd /scratch/<netid>
6. git clone https://github.com/leodup/collision-prediction.git
7. cd collision-prediction
8. mkdir logs
9. sbatch jupyter.slurm
10. cd logs
11. WAIT FOR 5 minutes atleast - jupyter_xxxxx.out file should show up. 
12. cat jupyter_xxxxx.out (The end should say jupyter is running. If it is not, wait and try after a few minutes)
13. Read the output, and it'll give exact instructions on what to do (You need to do it on another terminal, and open in your regular broswer)
14. Dataset is at /dataset
