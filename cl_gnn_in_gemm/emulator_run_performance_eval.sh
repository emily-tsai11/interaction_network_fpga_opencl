#!/bin/bash
cd bin
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./gnn_fpga /tigress/et7417/data/model_weights_LP_5.hdf5 /tigress/et7417/data/test_LP_5 1 1
