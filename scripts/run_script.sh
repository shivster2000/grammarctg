#!/bin/sh
module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy
source $HOME/gctg/bin/activate
python "$SCRIPT_NAME"