#!/bin/bash

REMOTE_USER="ops"
REMOTE_HOST="10.2.5.115"
REMOTE_SCRIPT="/home/ops/ngcmos/send_diagnostics.py"
REMOTE_FILE_LIST="/home/ops/ngcmos/files_to_download.txt"
LOCAL_FILE_LIST="/home/u5500483/ngcmos/files_to_download.txt"
LOCAL_DOWNLOAD_SCRIPT="/home/u5500483/ngcmos/download.py"
LOCAL_DEST_DIR="/path/to/local/destination"

# Step 1: Run the file identification script on the remote machine
ssh $REMOTE_USER@$REMOTE_HOST "python3 $REMOTE_SCRIPT"

# Step 2: Copy the file list from the remote machine to the local machine
scp $REMOTE_USER@$REMOTE_HOST:$REMOTE_FILE_LIST $LOCAL_FILE_LIST

# Step 3: Run the download script locally to transfer the files
python3 $LOCAL_DOWNLOAD_SCRIPT $LOCAL_FILE_LIST $LOCAL_DEST_DIR