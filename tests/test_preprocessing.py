from sys import intern
import pytest
import json

from natcap.root import rootcore
from natcap.root import preprocessing



def test_run_preprocessing():
    args_file = "tests/test_data/correct_ui_args_example.json"
    ui_args = json.load(open(args_file, "r"))["args"]
    internal_args = rootcore.parse_args(ui_args)
    preprocessing.execute(internal_args)
