import random
import sys

def generate_dag_data(num_nodes, max_children=4):
    children_count = {i: 0 for i in range(1, num_nodes + 1)}
    data, dot_data = [], ["digraph G {"]

    for node_id in range(1, num_nodes + 1):
        possible_parents = [p for p in range(1, node_id) if children_count[p] < max_children]
        
        if not possible_parents:
            parents = [1] * 4  # Default to node 1 as the parent
        else:
            num_parents = random.randint(1, min(4, len(possible_parents)))
            parents = random.choices(possible_parents, k=num_parents)
            for parent in parents:
                children_count[parent] += 1
            while len(parents) < 4:
                parents.append(parents[-1])

        timestamp = random.randint(1, 10000)
        parents_str = ' '.join(map(str, parents))
        data.append(f"{parents_str} {timestamp}")

        if node_id != 1:
            for parent in set(parents):
                dot_data.append(f"    {parent} -> {node_id};")

    dot_data.append("}")
    return [f"{num_nodes}"] + data, dot_data

def write_to_file(filename, content):
    with open(filename, "w") as file:
        file.write("\n".join(content))

if len(sys.argv) > 1:
    num_nodes = int(sys.argv[1])
else:
    print("Please provide the number of nodes as an argument.")
    sys.exit(1)

dag_data, dag_dot_data = generate_dag_data(num_nodes)

# Write the DAG dataset
write_to_file("dag_dataset.txt", dag_data)

# Write the DOT file for Graphviz
write_to_file("dag_graph.dot", dag_dot_data)

print("Dataset and DOT file generated successfully.")
