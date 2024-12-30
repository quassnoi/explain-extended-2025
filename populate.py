import psycopg2
import fire
import numpy as np
from pathlib import Path
from urllib import request
from safetensors.numpy import load_file
from itertools import groupby


def save_to_db(connection, key: str, weight: np.ndarray, bias: np.ndarray):
    with connection.cursor() as cursor:
        print(key, weight.shape, bias.shape)
        cursor.execute(
            "INSERT INTO parameters (key, weight, bias) VALUES (%s, %s, %s)", (key, weight.tolist(), bias.tolist()))
        connection.commit()


def get_model_file(model_url, model_dir):
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(exist_ok=True, parents=True)
    model_file_path = model_dir_path.joinpath("./model.safetensors")
    if not model_file_path.exists():
        request.urlretrieve(model_url, model_file_path)
    return model_file_path


def save_image_to_db(connection, data):
    with connection.cursor() as cursor:
        cursor.execute("INSERT INTO image VALUES (%s)", (data.tolist(),))
        connection.commit()


def create_table(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS parameters (key TEXT NOT NULL PRIMARY KEY, weight REAL[] NOT NULL, bias REAL[] NOT NULL)")
        connection.commit()


def create_functions(connection):
    with open("scripts/functions.sql") as f:
        with connection.cursor() as cursor:
            cursor.execute(f.read())
            connection.commit()


def main(connection_string="",
         model_url="https://huggingface.co/gnokit/ddpm-butterflies-64/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors",
         model_dir="./.models"):
    with psycopg2.connect(connection_string) as connection:
        model_file_path = get_model_file(model_url, model_dir)
        arrays = load_file(model_file_path)
        create_table(connection)
        for group in groupby(arrays.keys(), key=lambda key: key.rsplit(".", 1)[0]):
            save_to_db(
                connection, group[0], arrays[f"{group[0]}.weight"], arrays[f"{group[0]}.bias"])
        create_functions(connection)


if __name__ == "__main__":
    fire.Fire(main)
