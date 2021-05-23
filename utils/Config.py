import json


class Config:
    __data = None

    def __read():
        if Config.__data is None:
            with open("config.json", "r") as infile:
                config = json.loads(infile.read())
            Config.__data = config

    def get():
        Config.__read()
        return Config.__data
