#!/bin/bash

rm -rf ~/.cache/keops*

METHOD=wgan bash experiments/run_train_ks_once.sh
