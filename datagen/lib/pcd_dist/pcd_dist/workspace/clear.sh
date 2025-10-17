
#!/bin/bash

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR
rm -rf build 
rm -rf input 
rm -rf out 
rm -rf output 
rm ascendc_kernels_bbit

