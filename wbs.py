import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "gantt chart.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Ensure WBS numbers are strings
df["WBS #"] = df["WBS #"].astype(str)

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
for _, row in df.iterrows():
    wbs_id = row["WBS #"]
    name = row["Name / Title"]
    wbs_type = row["Type"]
    
    # Add node with label
    G.add_node(wbs_id, label=f"{wbs_id}: {name} ({wbs_type})")
    
    # Add edge to parent if it exists
    parent_id = ".".join(wbs_id.split(".")[:-1])
    if parent_id:
        G.add_edge(parent_id, wbs_id)

# Create a layout for the nodes (hierarchical top-down)
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(
    G,
    pos,
    with_labels=True,
    labels=nx.get_node_attributes(G, 'label'),
    node_size=3000,
    node_color="lightblue",
    font_size=8,
    font_color="black",
)
plt.title("Work Breakdown Structure")
plt.savefig("wbs_structure.png", dpi=300)
plt.show()
