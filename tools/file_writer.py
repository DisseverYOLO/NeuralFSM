# A file_write function takes the string and the file path. Write the string to the file path.
def file_write(string, file_path):
    with open(file_path, 'w') as file:
        file.write(string)