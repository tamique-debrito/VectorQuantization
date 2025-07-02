from __future__ import annotations
import copy
import random
from typing import Any

from Evolutionary.abstract_unit import AbstractBaseUnit

def _simple_mutate_table(table, probability, n_output):
    mutated_table = copy.deepcopy(table)
    for r_idx, row in enumerate(mutated_table):
        for c_idx, _ in enumerate(row):
            if random.random() < probability:
                mutated_table[r_idx][c_idx] = random.randrange(0, n_output)
    return mutated_table

def simple_mutate(base_unit: AbstractBaseUnit, probability: float) -> AbstractBaseUnit:
    mutated_unit = copy.deepcopy(base_unit)
    table_outputs = {
        "FORWARD_TABLE": base_unit.NUM_DATA_OBJ,
        "FIND_BACK_INFO_TABLE": base_unit.NUM_DATA_OBJ,
        "BACKWARD_TABLE": base_unit.NUM_DATA_OBJ,
        "FIND_UPDATE_TABLE": base_unit.NUM_UPDATE_OBJ,
        "APPLY_UPDATE_TABLE": base_unit.NUM_OPERATOR_OBJ,
    }

    table_names = list(table_outputs.keys())

    for table_name in table_names:
        current_table = getattr(mutated_unit, table_name)
        n_output = table_outputs[table_name]
        mutated_table = _simple_mutate_table(current_table, probability, n_output)
        setattr(mutated_unit, table_name, mutated_table)

    return mutated_unit

def _splice_table(table1, table2):
    assert len(table1) == len(table2), "Tables must have the same number of rows"
    assert len(table1[0]) == len(table2[0]), "Tables must have the same number of columns"
    
    spliced_table = []

    for r_idx in range(len(table1)):
        spliced_row = []
        for c_idx in range(len(table1[0])):
            spliced_row.append(random.choice([table1[r_idx][c_idx], table2[r_idx][c_idx]]))
        spliced_table.append(spliced_row)
    return spliced_table

def splice_units(base_unit1: AbstractBaseUnit, base_unit2: AbstractBaseUnit) -> AbstractBaseUnit:
    spliced_unit = copy.deepcopy(base_unit1) # Start with a copy of one unit
    # List of known lookup tables from AbstractBaseUnit
    table_names = [
        "FORWARD_TABLE",
        "FIND_BACK_INFO_TABLE",
        "BACKWARD_TABLE",
        "FIND_UPDATE_TABLE",
        "APPLY_UPDATE_TABLE",
    ]
    for table_name in table_names:
        table1 = getattr(spliced_unit, table_name)
        table2 = getattr(base_unit2, table_name)
        spliced_table = _splice_table(table1, table2)
        setattr(spliced_unit, table_name, spliced_table)
    return spliced_unit 