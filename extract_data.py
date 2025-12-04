import zipfile
import os
import sys
import subprocess

def extract_chunk(chunk_file="STEAD/chunk2.7z", output_dir="STEAD"):
    if not os.path.exists(chunk_file):
        print(f"File {chunk_file} not found. Please wait for download to complete.")
        sys.exit(1)

    print(f"Extracting {chunk_file}...")
    
    # Debug header
    try:
        with open(chunk_file, 'rb') as f:
            header = f.read(6).hex().upper()
        print(f"File Header: {header}")
        
        # Rename if it looks like a zip but has .7z extension
        if header.startswith("504B0304") and chunk_file.endswith(".7z"):
            new_name = chunk_file.replace(".7z", ".zip")
            print(f"Renaming {chunk_file} to {new_name} for compatibility...")
            os.rename(chunk_file, new_name)
            chunk_file = new_name
            
    except Exception as e:
        print(f"Could not read/rename file: {e}")

    # Try ZipFile
    try:
        if zipfile.is_zipfile(chunk_file):
            print("Detected ZIP format. Attempting extraction with zipfile...")
            with zipfile.ZipFile(chunk_file, 'r') as z:
                z.extractall(path=output_dir)
            print("Extraction complete (zipfile).")
            return
    except Exception as e:
        print(f"zipfile extraction failed: {e}")
        print("Falling back to py7zr...")

    # Try py7zr
    try:
        import py7zr
        if py7zr.is_7zfile(chunk_file):
            print("Detected 7z format. Attempting extraction with py7zr...")
            with py7zr.SevenZipFile(chunk_file, mode='r') as z:
                z.extractall(path=output_dir)
            print("Extraction complete (py7zr).")
            return
        else:
            # Force try py7zr if zipfile failed and header looks weird
            print("Not detected as 7z, but trying py7zr anyway...")
            with py7zr.SevenZipFile(chunk_file, mode='r') as z:
                z.extractall(path=output_dir)
            print("Extraction complete (py7zr forced).")
            return
            
    except Exception as e:
        print(f"py7zr extraction failed: {e}")

    # Try PowerShell Expand-Archive (Native Windows)
    print("Trying PowerShell Expand-Archive...")
    try:
        cmd = ["powershell", "-Command", f"Expand-Archive -Path '{chunk_file}' -DestinationPath '{output_dir}' -Force"]
        subprocess.check_call(cmd)
        print("Extraction complete (PowerShell).")
        return
    except Exception as e:
        print(f"PowerShell extraction failed: {e}")
        
    print("CRITICAL ERROR: Could not extract file. It might be corrupted.")
    sys.exit(1)

if __name__ == "__main__":
    extract_chunk()
