import os
import sys
import argparse
import json
import math
import random


def parse_axis(axis):
    if 'combine' in axis:
        return combine_axes(*axis['combine'])
    elif 'join' in axis:
        return join_axes(*axis['join'])
    else:
        return axis


def combine_axes(*axes):
    combination = {}
    for axis in axes:
        current_length = 1
        if combination.values():
            current_length = len(list(combination.values())[0])
        # combine axis
        added_length = len(list(axis.values())[0])
        for key in combination:
            combination[key] = combination[key]*added_length
        for key in axis:
            combination[key] = [value for value in axis[key] for i in range(current_length)]
    return combination


def join_axes(*axes):
    joined = {}
    for axis in axes:
        current_length = 0
        if joined.values():
            current_length = len(list(joined.values())[0])
        axis = parse_axis(axis)
        # join axis
        added_length = len(list(axis.values())[0])
        for key in joined:
            if key not in axis:
                joined[key] += ['N/a']*added_length
        for key in axis:
            if key not in joined:
                joined[key] = ['N/a']*current_length
            joined[key] += axis[key]
    return joined


def parse_args(config):
    command = ""
    for key, value in config.items():
        if key == '--weight_args':
            command += f'{key} {" ".join(map(str, value))} '
        elif key == '--data_gen_args':
            command += f'{key} {" ".join(map(str, value))} ' 
        elif value != 'N/a':
            command += f'{key} {value} '
    return command


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help='input JSON', type=str)
    parser.add_argument('--out_path', help='path to runner script', default='../runner.sh', type=str)
    args = parser.parse_args()

    # parse JSON
    with open(args.in_path) as json_file:
        json_obj = json.load(json_file)
        grid = parse_axis(json_obj["grid"])
    
    # make runner
    n_configs = len(next(iter(grid.values())))

    # open files
    command_file = open(args.out_path, 'w')

    # write to files
    for i in range(n_configs):
        # get config for this experiment (i)
        config = {}
        for key, value in grid.items():
            config[key] = value[i]
        # write command
        command_file.write(f'python gosdt/cleaned_main.py {parse_args(config)}\n')

    # close file
    command_file.close()