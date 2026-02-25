"""Functions to postprocess the results of the model."""

import pickle as pkl


def save_results(output_datafile, dictionary_result):
    """
    Save the dictionary_result to the output_datafile.

    Args:
        output_datafile (str): The path to the output data file.
        dictionary_result (dict): The dictionary to be saved.
    """

    # Check if directory exists, create if not
    if not output_datafile.parent.exists():
        output_datafile.parent.mkdir(parents=True)
    try:
        with open(output_datafile, "wb") as f:
            pkl.dump(dictionary_result, f)
    except Exception as e:
        print(f"Error occurred while saving results: {e}")

def load_results(input_datafile):
    """
    Load the dictionary_result from the input_datafile.

    Args:
        input_datafile (str): The path to the input data file.
    Returns:
        dict: The loaded dictionary.    
    """

    with open(input_datafile, "rb") as f:
        dictionary_result = pkl.load(f)
    return dictionary_result