#!/bin/bash

REMOTE_USER="ops"
REMOTE_HOST="10.2.5.115"
REMOTE_SCRIPT="/home/ops/ngcmos/Diagnostics/send_diagnostics.py"
REMOTE_FILE_LIST="/home/ops/ngcmos/Diagnostics/files_to_download.txt"
LOCAL_FILE_LIST="/home/u5500483/ngcmos/Diagnostics/files_to_download.txt"
LOCAL_DOWNLOAD_SCRIPT="/home/u5500483/ngcmos/Diagnostics/download_files.py"
LOCAL_DEST_DIR="/home/u5500483/shifts_plots_cmos/"

# Step 1: Run the file identification script on the remote machine
ssh $REMOTE_USER@$REMOTE_HOST "python3 $REMOTE_SCRIPT"

# Step 2: Copy the file list from the remote machine to the local machine
scp $REMOTE_USER@$REMOTE_HOST:$REMOTE_FILE_LIST $LOCAL_FILE_LIST

# Step 3: Check if the file list is empty
if [ ! -s $LOCAL_FILE_LIST ]; then
  echo "No files to download. Exiting."
  exit 0
fi

# Step 4: Run the download script locally to transfer the files
python3 $LOCAL_DOWNLOAD_SCRIPT $LOCAL_FILE_LIST $LOCAL_DEST_DIR

# Step 5: Delete the file list after transfer
rm $LOCAL_FILE_LIST