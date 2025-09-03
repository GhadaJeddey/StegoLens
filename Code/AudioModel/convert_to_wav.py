import os
import subprocess
from tqdm import tqdm

# Paths to your dataset folders
positive_folder = 'test_set\stego_set'
negative_folder = 'test_set\clean_set'
print('read the paths')
# Output folders
positive_output = r'test_set/positive_wav'
negative_output = r'test_set/negative_wav'

# Create output folders if they don't exist
os.makedirs(positive_output, exist_ok=True)
os.makedirs(negative_output, exist_ok=True)

def convert_to_wav(input_folder, output_folder):
    print('start')
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.g729a'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.wav'
            output_path = os.path.join(output_folder, output_filename)

            # Command to run ffmpeg
            command = ['ffmpeg', '-y', '-f', 'g729', '-i', input_path, output_path]

            # Execute the command
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Convert both positive and negative samples
convert_to_wav(positive_folder, positive_output)
convert_to_wav(negative_folder, negative_output)
