use clap::{App, Arg};
use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use tokio::fs::File;
use tokio::io;
use tokio::io::{AsyncBufReadExt, BufReader};

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Node {
    id: u32,
    parents: [u32; 4],
    timestamp: u64,
}

// Function to read the input file and parse the nodes, in this case ./dag_dataset.txt.
// The function relies on the asyncronous tokio library to read the file an increase the performance.
async fn read_input(file_path: &str) -> io::Result<Vec<Node>> {
    let file = File::open(file_path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut nodes = Vec::new();
    let mut index = 0;
    while let Some(line) = lines.next_line().await? {
        if index != 0 {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let parents = [
                parts[0].parse().unwrap(),
                parts[1].parse().unwrap(),
                parts[2].parse().unwrap(),
                parts[3].parse().unwrap(),
            ];
            nodes.push(Node {
                id: index as u32,
                parents,
                timestamp: parts[4].parse().unwrap(),
            });
        }
        index += 1;
    }
    Ok(nodes)
}

/// Function to build the DAG from the nodes read from the function read_input.
/// The function relies on the concurrent hashmap library DashMap to build the DAG in parallel manner.
/// https://github.com/xacrimon/dashmap
fn build_dag(nodes: &[Node]) -> HashMap<u32, Vec<u32>> {
    let graph: DashMap<u32, Vec<u32>> = DashMap::new();

    // Initialize an entry for every node in the graph
    for node in nodes {
        graph.entry(node.id).or_insert_with(Vec::new);
    }

    // Parallel iteration over nodes to build the DAG
    nodes.par_iter().for_each(|node| {
        // Add this node to the children of each of its parents
        for &parent in &node.parents {
            if parent != node.id && parent != 0 {
                graph
                    .entry(parent)
                    .and_modify(|children| {
                        if !children.contains(&node.id) {
                            children.push(node.id);
                        }
                    })
                    .or_insert_with(|| vec![node.id]);
            }
        }
    });

    // Convert DashMap to HashMap and deduplicate children lists
    graph
        .iter()
        .map(|entry| {
            let mut children = entry.value().clone();
            children.sort_unstable(); // Sorting before deduplication is efficient
            children.dedup();
            (*entry.key(), children)
        })
        .collect()
}

/// Display the DAG in a human-readable format (can be used for debugging by using the parameter --display-dag)
fn display_dag(dag: &HashMap<u32, Vec<u32>>) {
    let mut sorted_nodes: Vec<_> = dag.iter().collect();
    sorted_nodes.sort_by_key(|&(node, _)| node);

    for (node, children) in sorted_nodes {
        let children_str = if children.is_empty() {
            String::from("[]")
        } else {
            format!(
                "[{}]",
                children
                    .iter()
                    .map(|child| child.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        println!("{} -> {}", node, children_str);
    }
}

/// Calculates the average depth of nodes in a Directed Acyclic Graph (DAG).
///
/// This function performs an iterative breadth-first search (BFS) starting from the root node (assumed to be node 1)
/// to calculate the average depth of all nodes in the graph. The average depth is the sum of the depths of all nodes
/// divided by the total number of nodes.
///
/// Note: The function assumes that the DAG has a single root node with an id of 1. It does not handle graphs
/// with multiple roots or disconnected nodes.
fn iterative_bfs_for_depth(graph: &HashMap<u32, Vec<u32>>) {
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    let mut total_depth = 0;
    let mut total_nodes = 0;

    // Start with the root node
    queue.push_back((1, 0)); // Node 1 with depth 0

    while let Some((node, depth)) = queue.pop_front() {
        if visited.insert(node) {
            total_depth += depth;
            total_nodes += 1;

            if let Some(children) = graph.get(&node) {
                for &child in children {
                    if !visited.contains(&child) {
                        queue.push_back((child, depth + 1));
                    }
                }
            }
        }
    }

    let avg_depth = if total_nodes > 0 {
        total_depth as f64 / total_nodes as f64
    } else {
        0.0
    };

    println!("> AVG DAG DEPTH: {:.2}", avg_depth);
}

/// This function performs an iterative depth-first search (DFS) starting from the root node (assumed to be node 1)
/// to count the number of nodes at each depth level in the graph. The average is calculated as the total number of
/// nodes divided by the number of depth levels.
///
/// Note: The function assumes that the DAG has a single root node with an id of 1. It does not handle graphs
/// with multiple roots or disconnected nodes.
fn iterative_dfs_for_nodes_per_depth(graph: &HashMap<u32, Vec<u32>>) {
    let mut stack = Vec::with_capacity(graph.len());
    stack.push((1, 0));

    let mut depth_count = HashMap::new();
    let mut visited = HashSet::new();

    while let Some((node, depth)) = stack.pop() {
        if visited.insert(node) {
            if depth > 0 {
                *depth_count.entry(depth).or_insert(0) += 1;
            }
            if let Some(children) = graph.get(&node) {
                for &child in children {
                    if !visited.contains(&child) {
                        stack.push((child, depth + 1));
                    }
                }
            }
        }
    }

    let total_nodes: usize = depth_count.values().sum();
    let total_depths = depth_count.len();

    let avg_nodes_per_depth = (total_depths > 0)
        .then(|| total_nodes as f64 / total_depths as f64)
        .unwrap_or(0.0);
    println!("> AVG TXS PER DEPTH: {:.2}", avg_nodes_per_depth);
}

/// Calculates the average number of incoming edges (indegree) per node in a Directed Acyclic Graph (DAG).
///
/// This function iterates through the graph to count the number of incoming edges for each node.
/// The average is calculated as the total sum of indegrees divided by the number of nodes in the graph.
fn average_indegree(graph: &HashMap<u32, Vec<u32>>) {
    let mut indegrees = HashMap::new();

    // Initialize indegrees for all nodes to 0
    for &node in graph.keys() {
        indegrees.entry(node).or_insert(0);
    }

    // Count the indegrees
    for children in graph.values() {
        for &child in children {
            *indegrees.entry(child).or_insert(0) += 1;
        }
    }

    // Calculate the sum of indegrees
    let total_indegrees: u32 = indegrees.values().sum();
    let total_nodes = indegrees.len() as u32;

    // Calculate the average indegree
    let avg_indegree = if total_nodes > 0 {
        total_indegrees as f64 / total_nodes as f64
    } else {
        0.0
    };

    println!("> AVG REF: {:.3}", avg_indegree);
}

/// Calculates the shortest path from a starting node to a target node in a Directed Acyclic Graph (DAG).
///
/// This function uses a Breadth-First Search (BFS) algorithm to find the shortest path. BFS is chosen because
/// it efficiently finds the shortest path in terms of the number of edges in unweighted graphs like DAGs.
fn shortest_path(graph: &HashMap<u32, Vec<u32>>, from: u32, to: u32) {
    if from == to {
        println!("> SHORTEST PATH: [{:?}]", from);
    }

    let mut queue = VecDeque::new();
    let mut visited = HashMap::new();
    queue.push_back(from);

    while let Some(node) = queue.pop_front() {
        if node == to {
            // Reconstruct the path from 'to' to 'from'
            let mut path = Vec::new();
            let mut current = to;
            while current != from {
                path.push(current);
                current = *visited.get(&current).unwrap();
            }
            path.push(from);
            path.reverse();
            println!("> SHORTEST PATH: {:?}", path);
            return;
        }

        // Visit the children of the current node
        if let Some(children) = graph.get(&node) {
            for &child in children {
                if !visited.contains_key(&child) {
                    visited.insert(child, node);
                    queue.push_back(child);
                }
            }
        }
    }

    // No path found
    println!("> SHORTEST PATH: Path not found");
}

/// Calculates the longest path in a Directed Acyclic Graph (DAG).
///
/// This function leverages Depth-First Search (DFS) with memoization to find the longest path in the DAG. DFS is suitable
/// for exploring all possible paths in the graph deeply, while memoization helps in avoiding recomputation of the longest
/// path for already visited nodes.
fn longest_path(graph: &HashMap<u32, Vec<u32>>) {
    fn dfs(
        node: u32,
        graph: &HashMap<u32, Vec<u32>>,
        memo: &mut HashMap<u32, Vec<u32>>,
    ) -> Vec<u32> {
        if let Some(path) = memo.get(&node) {
            return path.clone();
        }

        let mut max_path = vec![node];
        if let Some(children) = graph.get(&node) {
            for &child in children {
                let mut path = dfs(child, graph, memo);
                if path.len() + 1 > max_path.len() {
                    path.insert(0, node);
                    max_path = path;
                }
            }
        }
        memo.insert(node, max_path.clone());
        max_path
    }

    let mut memo = HashMap::new();
    let mut longest_path = Vec::new();
    for &node in graph.keys() {
        let path = dfs(node, graph, &mut memo);
        if path.len() > longest_path.len() {
            longest_path = path;
        }
    }
    println!("> LONGEST PATH: {:?}", longest_path);
}

/// Calculates the maximum depth of a Directed Acyclic Graph (DAG).
///
/// This function uses an iterative Depth-First Search (DFS) algorithm to traverse the graph and find its maximum depth.
/// It iterates through each node, keeping track of the depth of each node relative to the root node (node 1). The function
/// ensures that each node is visited only at its deepest occurrence in the graph, which is crucial in DAGs where a node
/// may be reached through multiple paths.
fn max_depth(graph: &HashMap<u32, Vec<u32>>) {
    let mut max_depth = 0;
    let mut stack: Vec<(u32, usize)> = vec![(1, 0)];
    let mut visited = HashMap::new();

    while let Some((node, depth)) = stack.pop() {
        if let Some(&v_depth) = visited.get(&node) {
            if v_depth >= depth {
                continue; // Already visited with equal or greater depth
            }
        }

        visited.insert(node, depth);
        max_depth = max_depth.max(depth);

        if let Some(children) = graph.get(&node) {
            for &child in children {
                stack.push((child, depth + 1));
            }
        }
    }

    println!("> MAX DAG DEPTH: {}", max_depth);
}

// Function to calculate the total number of leaf nodes in the DAG
fn total_leaf_nodes(graph: &HashMap<u32, Vec<u32>>) {
    let total_leaf = graph
        .values()
        .filter(|children| children.is_empty())
        .count();
    println!("> TOTAL LEAFS: {}", total_leaf);
}

// Function to parse command line arguments, validate node IDs and decide whether to display the DAG.
fn parse_and_validate_args(nodes_len: usize) -> (u32, u32, bool) {
    let matches = App::new("DAG Statistics Calculator")
        .version("1.0")
        .author("Your Name")
        .about("Calculates statistics of a DAG")
        .arg(
            Arg::with_name("from")
                .short("f")
                .long("from")
                .value_name("FROM")
                .help("Starting node for shortest path calculation")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("to")
                .short("t")
                .long("to")
                .value_name("TO")
                .help("Ending node for shortest path calculation")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("display-dag")
                .short("d")
                .long("display-dag")
                .help("Show the constructed DAG")
                .takes_value(false),
        )
        .get_matches();

    let from_node: u32 = matches
        .value_of("from")
        .unwrap_or("1")
        .parse()
        .expect("Invalid FROM node ID");
    let to_node: u32 = matches
        .value_of("to")
        .unwrap_or("1")
        .parse()
        .expect("Invalid TO node ID");
    let display_dag: bool = matches.is_present("display-dag");

    // Validate node IDs
    if from_node as usize > nodes_len || to_node as usize > nodes_len {
        panic!("FROM or TO node ID is out of range");
    }

    (from_node, to_node, display_dag)
}

#[tokio::main]
async fn main() {
    let file_path = "./dag_dataset.txt";
    match read_input(file_path).await {
        Ok(nodes) => {
            let dag = build_dag(&nodes);
            //display_dag(&dag);
            let (from_node, to_node, should_display_dag) = parse_and_validate_args(nodes.len());
            if should_display_dag {
                display_dag(&dag);
            }
            // # Rayon Usage
            // Rayon was chosen for this implementation primarily for its ability to handle computationally intensive tasks efficiently.
            // Given that each calculation on large DAGs can be resource-intensive, Rayon's parallel processing significantly improves performance.
            // By spawning each computation in a separate thread, Rayon allows these calculations to be executed concurrently.
            rayon::scope(|s| {
                s.spawn(|_| iterative_bfs_for_depth(&dag));
                s.spawn(|_| iterative_dfs_for_nodes_per_depth(&dag));
                s.spawn(|_| average_indegree(&dag));
                s.spawn(|_| shortest_path(&dag, from_node, to_node));
                s.spawn(|_| longest_path(&dag));
                s.spawn(|_| max_depth(&dag));
                s.spawn(|_| total_leaf_nodes(&dag));
            });
        }
        Err(e) => println!("Error reading file: {}", e),
    }
}
