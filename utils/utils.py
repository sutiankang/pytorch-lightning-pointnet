import os
import yaml
import json
import datetime


def get_size(size):
    if isinstance(size, int):
        new_size = [size, size]
    elif isinstance(size, (list, tuple)):
        new_size = size
    else:
        raise TypeError
    return new_size


def check_repeat_dir(file_dir):
    count = 1
    new_dir = file_dir
    while os.path.exists(new_dir):
        new_dir = f'{file_dir}_{count}'
        count += 1
    return new_dir


def merge_args_with_dict(cfg, data_dict):
    assert isinstance(data_dict, (tuple, list, dict))
    if isinstance(data_dict, (tuple, list)):
        for data in data_dict:
            cfg.__dict__.update(data)
    else:
        cfg.__dict__.update(data_dict)
    return cfg


def read_file(data_file):
    if data_file.endswith('.yaml'):
        data = read_yaml_file(data_file)
    elif data_file.endswith('.json'):
        data = read_json_file(data_file)
    elif data_file.endswith('.txt'):
        data = read_txt_file(data_file)
    else:
        raise TypeError
    return data


def create_work_dir(cfg):
    if cfg.create_time_dir:
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        cfg.work_dir = os.path.join(cfg.work_dir, time)
        cfg.repeat_work_dir = True
    if not cfg.repeat_work_dir:
        cfg.work_dir = check_repeat_dir(cfg.work_dir)
    return cfg


def read_txt_file(txt_file: str) -> list:
    assert txt_file.endswith('.txt')
    with open(txt_file, 'r', encoding='utf-8') as f:
        data_list = [line.strip() for line in f.readlines()]
    f.close()
    return data_list


def read_json_file(json_file: str) -> dict:
    assert json_file.endswith('.json')
    with open(json_file, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    f.close()
    return data_dict


def read_yaml_file(yaml_file: str) -> dict:
    assert yaml_file.endswith('.yaml')
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()
    return data_dict


def save_json_file(save_json_file: str, data_dict) -> None:
    assert save_json_file.endswith('.json') and isinstance(data_dict, dict)
    with open(save_json_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=1)
    f.close()


def save_txt_file(save_txt_file: str, data_list) -> None:
    assert save_txt_file.endswith('.txt') and isinstance(data_list, list)
    # check
    if not data_list[0].endswith('\n'):
        data_list = [data + '\n' for data in data_list]
    with open(save_txt_file, 'w', encoding='utf-8') as f:
        f.writelines(data_list)
    f.close()


def save_yaml_file(save_yaml_file: str, data_dict) -> None:
    assert save_yaml_file.endswith('.yaml') and isinstance(data_dict, dict)
    with open(save_yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(data_dict, f, sort_keys=False)
    f.close()