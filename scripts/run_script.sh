#!/bin/bash
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy
source /cluster/work/sachan/dglandorf/gctg/bin/activate
python "$SCRIPT_NAME" "$@"