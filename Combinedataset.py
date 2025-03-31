import os
import shutil
import soundfile as sf
from tqdm import tqdm

def load_protocol(file_path):
    """Load ASVspoof2019 protocol file and return a dictionary mapping filenames to labels."""
    protocol = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename, label = parts[1], parts[-1]  # Assuming filename is second column and label is last
            protocol[filename] = "real" if label == "bonafide" else "fake"
    return protocol

def convert_and_move(input_path, output_path):
    """Convert FLAC to WAV and move the file."""
    y, sr = sf.read(input_path)
    sf.write(output_path, y, sr)

def process_dataset(asv_base_path, scene_base_path, protocol_files):
    """Convert and structure ASVspoof2019 dataset to match Scenefake format."""
    for subset, protocol_file in protocol_files.items():
        protocol = load_protocol(protocol_file)
        asv_flac_path = os.path.join(asv_base_path, f'ASVspoof2019_LA_{subset}', 'flac')
        scene_real_path = os.path.join(scene_base_path, subset, 'real')
        scene_fake_path = os.path.join(scene_base_path, subset, 'fake')
        os.makedirs(scene_real_path, exist_ok=True)
        os.makedirs(scene_fake_path, exist_ok=True)
        
        for filename, label in tqdm(protocol.items(), desc=f'Processing {subset}'):
            input_flac = os.path.join(asv_flac_path, f'{filename}.flac')
            output_wav = os.path.join(scene_real_path if label == 'real' else scene_fake_path, f'{filename}.wav')
            if os.path.exists(input_flac):
                convert_and_move(input_flac, output_wav)

def main():
    asv_base_path = "asvpoof-2019-dataset/LA/LA"  # Update this
    scene_base_path = "datasetNEW"  # Update this
    protocol_files = {
        "train": "asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",  # Update paths
        "dev": "asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    }
    
    process_dataset(asv_base_path, scene_base_path, protocol_files)
    print("Dataset processing complete.")

if __name__ == "__main__":
    main()
