import json
import tempfile
import os
import hashlib


class Cache:

    def process(key, func, *args):
        file_name = Cache.__get_file_name(key)
        if Cache.__key_exists(file_name):
            return Cache.__read_file(file_name)
        else:
            val = func(*args)
            Cache.__write_file(file_name, val)
            return val

    def get(key):
        file_name = Cache.__get_file_name(key)
        if Cache.__key_exists(file_name):
            return Cache.__read_file(file_name)
        else:
            return None

    def delete(key):
        file_name = Cache.__get_file_name(key)
        if Cache.__key_exists(file_name):
            os.remove(file_name)

    def __get_file_name(key):
        clean_key = hashlib.md5(str(key).encode()).hexdigest()
        a = clean_key[0]
        b = clean_key[1]
        file_name = "{}/PyCache/{}/{}/{}.json".format(tempfile.gettempdir(), a, b, clean_key)
        file_name = os.path.abspath(file_name)
        return file_name

    def __key_exists(file_name):
        return os.path.exists(file_name)

    def __read_file(file_name):
        with open(file_name, "r") as infile:
            return json.loads(infile.read())

    def __write_file(file_name, val):
        dir = os.path.dirname(file_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(file_name, "w") as outfile:
            outfile.write(json.dumps(val))
