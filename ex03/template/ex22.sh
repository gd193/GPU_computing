#!/usr/bin/env bash
#SBATCH --gres=gpu:1

bin/memCpy --global-coalesced