import os
import shutil
import time
def get_drive_free_space(drive):
    total, used, free = shutil.disk_usage(drive)
    return {
        "total": total,
        "used": used,
        "free": free
    }
def move_files(source_dir, destination_dir):
    # Ensure the destination directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist. Exiting.")
        return
    os.makedirs(destination_dir, exist_ok=True)
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Construct the relative path from the source directory
        relative_path = os.path.relpath(root, source_dir)
        
        # Construct the corresponding destination directory path
        dest_path = os.path.join(destination_dir, relative_path)
        
        # Ensure the destination subdirectory exists
        os.makedirs(dest_path, exist_ok=True)
        
        # Move files to the destination subdirectory
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            shutil.move(src_file, dest_file)

source_directory_GLU = "N:\\SPDC_Data_Diagonsis\\GLU"
source_directory_MLU = "N:\\SPDC_Data_Diagonsis\\MLU"
destination_directory_GLU = 'D:\\SPDC_Data_Diagonsis\\GLU\\'
destination_directory_MLU = 'D:\\SPDC_Data_Diagonsis\\MLU\\'




def run_at_intervals(interval, source_directory_GLU,source_directory_MLU, destination_directory_GLU, destination_directory_MLU):
    while True:
        # if free_space_info['free']/(1024**3)<1:
        move_files(source_directory_GLU,destination_directory_GLU) 
        move_files(source_directory_MLU,destination_directory_MLU) 
        time.sleep(interval)
interval = 10  # Interval in seconds (e.g., 3600 seconds for 1 hour)
run_at_intervals(interval,source_directory_GLU,source_directory_MLU, destination_directory_GLU,destination_directory_MLU)
