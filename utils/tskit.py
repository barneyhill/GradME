import numpy as np
import jax.numpy as jnp
import tskit, msprime
import matplotlib.pyplot as plt

def simulate_ts(
    sample_size: int,
    length: int = 100,
    population_size: int = 1000,
    sim_duration = 1E10,
    mutation_rate: float = 1e-6,
    random_seed: int = 1,
) -> tskit.TreeSequence:
    """
    Simulate some data using msprime with recombination and mutation and
    return the resulting tskit TreeSequence.
    Note this method currently simulates with ploidy=1 to minimise the
    update from an older version. We should update to simulate data under
    a range of ploidy values.
    """
    ancestry_ts = msprime.sim_ancestry(
        sample_size,
        population_size=population_size,
        ploidy=1,
        recombination_rate=1E-8,
        sequence_length=length,
        random_seed=random_seed,
        model=msprime.StandardCoalescent(duration=sim_duration)
    )

    # Make sure we generate some data that's not all from the same tree
    assert ancestry_ts.num_trees > 1
    return msprime.sim_mutations(
        ancestry_ts, rate=mutation_rate, random_seed=random_seed, model="binary",
    )

def build_ts_from_matrix(M, samples):
    # Step 1: Initialize a TableCollection
    tables = tskit.TableCollection(sequence_length=1.0) # Assuming sequence length is 1.0

    # assuming M is your numpy array
    # merging children columns
    children = np.concatenate((M[:, 1], M[:, 2]))

    # finding root node mask
    root = np.isin(M[:, 0], children, invert=True)

    t = M.shape[0] - len(samples)

    node_map = {node_id: None for node_id in np.unique(M)}

    # Step 2: Create a node map and set time as 0 for terminal nodes
    for sample in samples:
        node_map[sample] = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)

    # Step 3: Starting at the root BFS down adding times so that parents are always older than children

    # initialize a queue for BFS
    queue = deque()

    # starting node (or nodes) is/are the root(s), i.e., nodes that are not children of any other nodes

    node_map[M[root, 0][0]] = tables.nodes.add_row(time=t)
    queue.append((M[root, 0][0], t))

    # BFS traversal
    while queue:
        # Get the parent node and its time
        parent, parent_time = queue.popleft()

        # Decrease the time for each new generation
        t -= 1

        # Get the children of the parent node
        children = M[M[:, 0] == parent, 1:]

        # Go through each child
        for child in children[0]:              

            # Add a new node for each child and set its time
            node_map[int(child)] = tables.nodes.add_row(time=t)
            
            # Add an edge from the parent to the child
            tables.edges.add_row(left=0, right=1, parent=node_map[parent], child=node_map[int(child)])

            # Enqueue the child node for further processing
            if child not in samples:
              queue.append((child, t))


    # Step 4: Sort and simplify the tables
    tables.sort()
    tables.simplify(samples=[node_map[sample] for sample in samples])

    # Step 5: Generate the tree sequence
    ts = tables.tree_sequence()
    return ts