import os

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            print("Data from file:")
            print(data)
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    file_path = '/data/demo.txt'
    if os.path.exists(file_path):
        read_file(file_path)
    else:
        print(f"File not found: {file_path}")
