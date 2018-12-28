#!/usr/bin/env bash
conda env export --no-builds | sed '/prefix:/d' > condaenv.yml
