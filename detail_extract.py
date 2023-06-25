from src.utils.utils import detailed_direct_eval
import ast
import csv

with open("unsimplified_1686828189_result.txt", mode="r") as f:
    lines = f.readlines()

status = "find_start"
current_index = -1
current_output = []
current_target = []

header = ["index", "output", "target", "predicate_match_count", "predicate_output_count", "predicate_target_count"]
available_roles = set()
data_list = []
for line in lines:
    line = line.strip()
    if status == "find_start":
        if line.startswith("---"):
            status = "expect_index"

    elif status == "expect_index":
        assert line.startswith("Index: ")
        current_index = int(line[len("Index: "):])
        status = "expect_output"

    elif status == "expect_output":
        assert line.startswith("Output: ")
        current_output = ast.literal_eval(line[len("Output: "):])
        status = "expect_target"
    
    elif status == "expect_target":
        assert line.startswith("Target: ")
        current_target = ast.literal_eval(line[len("Target: "):])
        status = "find_end"

    elif status == "find_end":
        if line.startswith("---"):
            detail = detailed_direct_eval(current_output, current_target)
            data = {
                "index": current_index,
                "output": current_output,
                "target": current_target,
                "predicate_match_count": detail["match"]["predicate"],
                "predicate_output_count": detail["output"]["predicate"],
                "predicate_target_count": detail["target"]["predicate"],
            }

            for scope in ["match", "output", "target"]:
                for key, value in detail[scope]["argument"].items():
                    if key != "all":
                        available_roles.add(key)
                    data[f"argument_{key}_{scope}_count"] = value

            data_list.append(data)

            current_index = -1
            current_output = []
            current_target = []
            status = "find_start"

assert status == "find_start"

available_roles = list(available_roles)
available_roles.sort()
available_roles.insert(0, "all")

for role in available_roles:
    for scope in ["match", "output", "target"]:
        header.append(f"argument_{role}_{scope}_count")

with open("unsimplified_result.csv", mode="w", newline="") as f:
    csv_writer = csv.DictWriter(f, fieldnames=header)
    csv_writer.writeheader()
    csv_writer.writerows(data_list)

print("Done!")
