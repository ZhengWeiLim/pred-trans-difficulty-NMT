filtered_studies = {"AR19": "all", "ZHPT12": "all", "HLR13": "all",
                    "KTHJ08": ["P05_T3"], # sentence ids inconsistent
                    "XIANG19": ["P05_T6"]} # all rows filtered due to boundary crosses


table_correction = {
    "TPR-DB/translog/tables/diverse/HLR13-tables/P01_T3.sg": {"STseg": [("4+5", "4"), ("6", "5")]},
    "TPR-DB/translog/tables/diverse/HLR13-tables/P01_T3.st": {"STseg": [("5", "4"), ("6", "5")]}
}

def correct_table(table, table_path):
    for cpath, corrections in table_correction.items():
        if cpath in table_path:
            for col, replace_vals in corrections.items():
                for replace_val in replace_vals:
                    old_val, new_val = replace_val
                    table[col] = table[col].replace(old_val, new_val)
            return table

    return table



