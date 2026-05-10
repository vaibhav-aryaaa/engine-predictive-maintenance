import os
import gdown
import zipfile

def download_nasa_from_gdrive():
    # The ID from the script you found
    file_id = '1uNIreOQ1GWxIu_Aw-YWLT-xobcSWV2Kj'
    output_zip = 'data/nasa_dataset.zip'
    extract_path = 'data/nasa_raw'

    os.makedirs('data', exist_ok=True)

    print("🚀 Downloading dataset from Google Drive...")
    gdown.download(id=file_id, output=output_zip, quiet=False)

    print("📦 Unzipping dataset...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Cleanup
    os.remove(output_zip)
    print(f"✅ Success! Data extracted to {extract_path}")
    print("Inside that folder, you should see 'train_FD001.txt'.")

if __name__ == "__main__":
    download_nasa_from_gdrive()
