import pandas as pd
import requests
import os


def get_csv():
    tmp_name = "tmp.csv"
    response = requests.get(
        'https://docs.google.com/spreadsheets/d/1saeZRgg-E5ZUgyvuvd8qeJGYnJQtCSPz8QZSiKkzjdw/export?format=csv&id=1saeZRgg-E5ZUgyvuvd8qeJGYnJQtCSPz8QZSiKkzjdw&gid=0')
    assert response.status_code == 200, 'Wrong status code'

    file = open(tmp_name, "w")
    file.write(response.content)
    file.close()

    df = pd.read_csv(tmp_name)
    os.remove(tmp_name)
    return df
