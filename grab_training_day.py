import os
import shutil

def copy_most_recent_folder(source_folder, target_folder):
    folders = os.listdir(source_folder)
    if not folders:
        print("No folders found in the source directory.")
        return

    most_recent_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(source_folder, f)))
    source_path = os.path.join(source_folder, most_recent_folder)
    target_path = os.path.join(target_folder, most_recent_folder)

    try:
        shutil.copytree(source_path, target_path)
        print("Successfully copied the most recent folder '{most_recent_folder}' to the target location.")
    except Exception as e:
        print("An error occurred while copying the folder: {e}")

#now a function that iterates over a list of source folders and and runs the function above
def grab_training_files(source_folders, target_folders):
    for source_folder in source_folders:
        target_folder = target_folders[source_folders.index(source_folder)]
        copy_most_recent_folder(source_folder, target_folder)


# Example usage
base_folder = 'D:\\2024\\sandra'
source_folders = [os.path.join(base_folder, 'SR_0000006'), os.path.join(base_folder, 'SR_0000007'),os.path.join(base_folder, 'SR_0000008'), os.path.join(base_folder, 'SR_0000010'), os.path.join(base_folder, 'SR_0000011'), os.path.join(base_folder, 'SR_0000012'),os.path.join(base_folder, 'SR_0000013')]
base_target_folder = 'Y:\\public\\projects\\SaRe_20240219_hfs\\training_data\\v2'
target_folders = [os.path.join(base_target_folder, 'SR_0000006'), os.path.join(base_target_folder, 'SR_0000007'), os.path.join(base_target_folder, 'SR_0000008'), os.path.join(base_target_folder, 'SR_0000010'), os.path.join(base_target_folder, 'SR_0000011'),  os.path.join(base_target_folder, 'SR_0000012'),os.path.join(base_target_folder, 'SR_0000013')]

grab_training_files(source_folders, target_folders)

