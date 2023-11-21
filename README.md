# DAG Statistics Calculator

## Overview
This project is a Rust-based program designed to calculate various statistics of Directed Acyclic Graphs (DAGs). It computes the following metrics:
 - Average depth per node (AVG DAG DEPTH).
 - Average number of nodes per depth (AVG TXS PER DEPTH).
 - Average in-degree per node (AVG REF).
 - Shortest path between two specific nodes within the DAG. These nodes are specified by command line parameters --from / --to (SHORTEST PATH).
 - DAG's longest path (LONGEST PATH). 
 - DAG's maximum depth (MAX DAG DEPTH).

## Running the Project
Compile the project using Cargo:
```bash
cargo build --release
```

Run the compiled program, specifying nodes for calculating the shortest path and optionally displaying the constructed DAG:
```bash
./target/release/dag_statistics --from <FROM_NODE> --to <TO_NODE> [--display-dag]
```

- FROM_NODE: Starting point for the shortest path in the DAG.
- TO_NODE: Destination for the shortest path in the DAG.

## DAG Generation
The project includes a Python script `gen_dag.py` which dynamically generates a DAG. This script produces two files:
- `dag_graph.dot`: A DOT file representing the graph structure.
- `dag_dataset.txt`: Corresponds to the graph dataset as per the challenge specifications.

Initially, the challenge requested a binary tree DAG with a maximum of two parents per node. To add complexity and versatility, this was expanded to allow up to four parents per node.

### Usage
To generate a DAG with a specific number of nodes:
```bash
python gen_dag.py <NUMBER_NODES>
```

### Visualizing the DAG
The generated DAG can be visualized using Graphviz. To create a PNG image of the DAG:
```bash
dot -Tpng dag_graph.dot -o dag_graph.png
```

## Algorithms Used
The program employs both Breadth-First Search (BFS) and Depth-First Search (DFS) algorithms. The use of both algorithms showcases a diverse approach to solving graph-related problems. 

### Breadth-First Search (BFS):

Used in shortest_path function to find the shortest path between two nodes. BFS is particularly effective for this purpose in unweighted graphs like DAGs, as it ensures the shortest path is found in terms of the number of edges.
Also employed in iterative_bfs_for_depth to calculate the average depth of the nodes in the DAG. BFS guarantees that the shortest path to each node is considered for the depth calculation.

### Depth-First Search (DFS):

Applied in iterative_dfs_for_nodes_per_depth for counting the number of nodes at each depth level. DFS is adept at exhaustively traversing each branch of the graph, making it ideal for depth-related calculations.
Utilized with memoization in longest_path to find the longest path in the DAG. DFS, combined with memoization, efficiently explores all possible paths and remembers the longest ones, reducing redundant calculations.
Also used in max_depth to determine the maximum depth of the DAG, ensuring each node is visited at its deepest level in the graph.

## Concurrency
The calculations are executed concurrently to optimize performance, especially for large graphs. This is achieved using the Rayon library in Rust, which enables parallel processing. As a result, the output may not always be in the same order, highlighting the concurrent execution of tasks.

## Asynchronous File Reading with Tokio
The project utilizes Tokio, a Rust asynchronous runtime, for non-blocking file reading. This approach is applied in reading the DAG dataset from dag_dataset.txt, particularly useful for processing large files.

## Testing
The program has been tested with DAGs containing up to 10,000 nodes, ensuring robustness and efficiency even with large datasets.

## TODO:

- Create unit testing for a known set of values/results.


