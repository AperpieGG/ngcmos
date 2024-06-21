#!/bin/bash

# Variables
REMOTE_USER="ops"
REMOTE_HOST="10.2.5.115"
REMOTE_DIR="/home/u5500483/ngcmos/"
LOCAL_DIR="/home/ops/ngcmos"

# Rsync command to synchronize files
rsync -avz --delete $LOCAL_DIR $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR