"""
Copy text file from the original dataset folder to the target dataset folder.
Usage:
```bash
python copy_text.py -d <original_dataset_path> -o <target_dataset_path>
```
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, required=True,
                    help="The folder containing the text file to copy from")
parser.add_argument("-o", "--output_dir", type=str, required=True,
                    help="The folder to save the text file")
args = parser.parse_args()

ori_dataset_path = args.dir
output_dataset_path = args.output_dir
os.makedirs(output_dataset_path, exist_ok=True)

folders = os.listdir(ori_dataset_path)
folders = [folder for folder in folders if os.path.isdir(os.path.join(ori_dataset_path, folder))]

count = 0

for folder in folders:
    os.makedirs(os.path.join(output_dataset_path, folder), exist_ok=True)
    ori_text_path = os.path.join(ori_dataset_path, folder, f'{folder}.yaml')
    output_text_path = os.path.join(output_dataset_path, folder, f'{folder}.yaml')
    shutil.copy(ori_text_path, output_text_path)
    count += 1

print(f"Copied {count} text files")
    
