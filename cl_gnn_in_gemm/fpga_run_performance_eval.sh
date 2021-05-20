#!/bin/bash

unset CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA
unset CL_CONTEXT_EMULATOR_DEVICE_ALTERA

cd bin

aocl program acl0 gnn.aocx
aocl program acl1 gnn.aocx

echo "Pt: 5 GeV"
./gnn_fpga /tigress/et7417/data/model_weights_LP_5.hdf5 /tigress/et7417/data/test_LP_5 100 1
