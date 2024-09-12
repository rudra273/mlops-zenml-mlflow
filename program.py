import os

# Define the file extensions you want to include (excluding .csv)
extensions = ['.py']
exclude_dirs = ['venv']  # List of directories to exclude

# Create or overwrite the output file
with open('summary.txt', 'w', encoding='utf-8') as output_file:
    # Iterate over the directory and its subdirectories
    for root, dirs, files in os.walk('.'):
        # Modify the dirs list in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            # Check if the file has the desired extension
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                output_file.write(f"\n===== {file_path} =====\n\n")
                
                # Try to open and write the file content to summary.txt
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        output_file.write(f.read())
                except UnicodeDecodeError:
                    # If UTF-8 decoding fails, try using a different encoding
                    with open(file_path, 'r', encoding='latin-1') as f:
                        output_file.write(f.read())

                output_file.write("\n")  # Add a newline for better readability
