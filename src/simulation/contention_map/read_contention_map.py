import pickle

def read_pickle_file(pickle_filename):
    """
    Reads a pickle file and returns its content.
    
    Args:
    pickle_filename (str): The path to the pickle file.

    Returns:
    The content of the pickle file.
    """
    with open(pickle_filename, 'rb') as file:
        data = pickle.load(file)
    return data

def main():
    pickle_filename = 'contention_map.pkl'  # Replace with your actual pickle file path
    data = read_pickle_file(pickle_filename)
    print(data)

if __name__ == '__main__':
    main()
