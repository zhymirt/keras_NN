""" Thompson 2022

    Utility functions useful for reading/writing JSON files.
"""
import json


def load_json_file(filename: str):
    """ Load and return JSON object.

        :param str filename: File path to data.
        :returns: JSON object loaded from file at filename.
    """
    with open(filename, 'r') as json_file:
        obj = json.load(json_file)
    return obj


def save_json_file(data: dict, filename: str):
    """ Save data dictionary as JSON file with filename.

        :param dict data: Dictionary of data to save.
        :param str filename: Path to save data.
    """
    with open(filename, 'w') as new_file:
        json.dump(data, new_file)


def dict_to_json(data: dict, indents: int = 4):
    """ Return JSON string of given data.

        :param dict data: Data to write.
        :param int indents: Number of indents to use in JSON file.
        :returns: JSON string object.
        :rtype: str
    """
    json_object = json.dumps(data, indent=indents)
    return json_object
