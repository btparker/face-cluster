def chinese_whispers(encoding_list, threshold=0.6, iterations=10):
    import networkx as nx
    from face_recognition.api import _face_distance
    from random import shuffle
    import json

    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print("Not enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx+1
        node = (node_id, {'class': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        # Facial encodings to compare
        compare_encodings = encodings[idx+1:]
        distances = _face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance < threshold:
                # Add edge if facial match
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    for z in range(0,iterations):
        gn = G.nodes()
        # I randomize the nodes to give me an arbitrary start point
        shuffle(gn)
        for node in gn:
            neighs = G[node]
            classes = {}
            # do an inventory of the given nodes neighbours and edge weights
            for ne in neighs:
                if isinstance(ne, int) :
                    if G.node[ne]['class'] in classes:
                        classes[G.node[ne]['class']] += G[node][ne]['weight']
                    else:
                        classes[G.node[ne]['class']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            max = 0
            maxclass = 0
            for c in classes:
                if classes[c] > max:
                    max = classes[c]
                    maxclass = c

            # set the class of target node to the winning local class
            G.node[node]['class'] = maxclass

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():

        node_class = data['class']
        path = data['path']

        if node_class:
            if node_class not in clusters:
                clusters[node_class] = []

            clusters[node_class].append(path)
    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters

def main(args):
    import face_recognition
    from glob import glob
    from os.path import join, basename, exists
    from os import makedirs
    import itertools
    import json
    import numpy as np
    import shutil

    # Facial encodings
    facial_encodings = {}
    for idx, image_path in enumerate(glob(join(args.input, '*.jpg'))):
        print("Encoding '{}' ...".format(image_path))
        picture = face_recognition.load_image_file(image_path)
        results = face_recognition.face_encodings(picture)
        if len(results) == 1:
            facial_encodings[image_path] = results[0]
            print("... stored")
        else:
            print("... image does not have just one face, skipping")

    with open(join(args.output, 'facial_encodings.npy'), 'w') as outfile:
        np.save(outfile, facial_encodings)

    encoding_list = facial_encodings.items()
    sorted_clusters = chinese_whispers(encoding_list)

    with open(join(args.output, 'facial_clusters.npy'), 'w') as outfile:
        np.save(outfile, sorted_clusters)

    for idx, cluster in enumerate(sorted_clusters):
        cluster_name = str(idx).zfill(4)
        cluster_dir = join(args.output, cluster_name)
        if not exists(cluster_dir):
            makedirs(cluster_dir)
        for path in cluster:
            name = basename(path)
            shutil.copy(path, join(cluster_dir, name))

def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--input', type=str, help='Input dir of images', required=True)
    parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
