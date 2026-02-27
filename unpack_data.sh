#!/usr/bin/env bash

# unpack the two archives under 1-AlLi-GRACE-2LAYER/0-data
set -euo pipefail

base="1-AlLi-GRACE-2LAYER/0-data"

echo "Extracting ${base}/AlLi_Materials_Project.tar.gz …"
tar -xzf "${base}/AlLi_Materials_Project.tar.gz" -C "${base}"

echo "Extracting ${base}/AlLi_vasp_data.tar.gz …"
tar -xzf "${base}/AlLi_vasp_data.tar.gz" -C "${base}"

echo "Done."