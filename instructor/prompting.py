from datasets.shapes import DatasetEntry, Circle, Shape, Triangle
from typing import ClassVar, Type


def are_all_solution(entry: DatasetEntry, target_shape: str, target_color: str):
    shapes = entry.shapes
    output_commands = ["<NEXT>: True"]
    current_index = 0
    while output_commands[-1] != "<ACCEPT>" and output_commands[-1] != "<REJECT>" and current_index < 20:
        if output_commands[-1] == "<NEXT>: True":
            if shapes[current_index].name == target_shape:
                output_commands.append("<CHECK SHAPE>: True")
            else:
                output_commands.append("<CHECK SHAPE>: False")
        elif output_commands[-1] == "<NEXT>: False":
            output_commands.append("<ACCEPT>")
        elif output_commands[-2] == "<CHECK SHAPE>: True" and output_commands[-1] == "<CHECK COLOR>: False":
            output_commands.append("<REJECT>")
        elif output_commands[-1] == "<CHECK SHAPE>: True":
            if shapes[current_index].color == target_color:
                output_commands.append("<CHECK COLOR>: True")
            else:
                output_commands.append("<CHECK COLOR>: False")
        elif output_commands[-1] == "<CHECK SHAPE>: False" or output_commands[-1] == "<CHECK COLOR>: True":
            current_index += 1
            if current_index < len(shapes):
                output_commands.append("<NEXT>: True")
            else:
                output_commands.append("<NEXT>: False")
        # print(output_commands)
        # current_index += 1
    return f"Input: [{', '.join(map(str, shapes))}]\nOutput: [{', '.join(output_commands)}]"
