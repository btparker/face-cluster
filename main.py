""" Face Cluster """

def _create_graph_from_encodings(encoding_list, threshold=0.6):
    """ Create Graph from Facial Encodings

    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold

    Outputs:
        G: graph of facial encodings as nodes and distances as edges
    """
    import networkx as nx
    from face_recognition.api import _face_distance

    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print "Not enough encodings to cluster!"
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx+1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break

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

    return G

def _chinese_whispers(G, iterations=10):
    """ Chinese Whispers Algorithm

    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/

    Inputs:
        G: graph of facial encodings as nodes and distances as edges
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate

    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """
    from random import shuffle

    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            # do an inventory of the given nodes neighbors and edge weights
            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters

def cluster_facial_encodings(facial_encodings):
    """ Cluster facial encodings

        Intended to be an optional switch for different clustering algorithms, as of right now
        only chinese whispers is available.

        Input:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest

    """

    # Only use the chinese whispers algorithm for now
    facial_graph = _create_graph_from_encodings(facial_encodings.items())
    sorted_clusters = _chinese_whispers(facial_graph)
    return sorted_clusters

def compute_facial_encodings(image_paths):
    """ Compute Facial Encodings

        Given a set of images, compute the facial encodings of each face detected in the images and
        return them. If no faces, or more than one face found, return nothing for that image.

        Inputs:
            image_paths: a list of image paths

        Outputs:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

    """
    import face_recognition

    facial_encodings = {}
    for image_path in image_paths:
        print "Encoding '{}' ...".format(image_path)
        picture = face_recognition.load_image_file(image_path)
        results = face_recognition.face_encodings(picture)
        if len(results) == 1:
            facial_encodings[image_path] = results[0]
            print "... stored"
        else:
            print "... image does not have just one face, skipping"

    return facial_encodings

def main(args):
    """ Main

    Given a list of images, save out facial encoding data files and copy
    images into folders of face clusters.

    """
    from glob import glob
    from os.path import join, basename, exists
    from os import makedirs
    import numpy as np
    import shutil
    import sys

    if not exists(args.output):
        makedirs(args.output)

    # Facial encodings
    image_paths = glob(join(args.input, '*.jpg'))

    if len(image_paths) == 0:
        print "No jpg images found in {}, exiting...".format(args.input)
        sys.exit(0)

    facial_encodings = compute_facial_encodings(image_paths)

    # Save facial encodings
    with open(join(args.output, 'facial_encodings.npy'), 'w') as outfile:
        np.save(outfile, facial_encodings)

    # Compute facial clusters, return as sorted
    sorted_clusters = cluster_facial_encodings(facial_encodings)

    # Save clusters
    with open(join(args.output, 'facial_clusters.npy'), 'w') as outfile:
        np.save(outfile, sorted_clusters)

    # Copy image files to cluster folders
    for idx, cluster in enumerate(sorted_clusters):
        cluster_dir = join(args.output, str(idx).zfill(4))
        if not exists(cluster_dir):
            makedirs(cluster_dir)
        for path in cluster:
            shutil.copy(path, join(cluster_dir, basename(path)))

def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--input', type=str, help='Input dir of images', required=True)
    parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    main(parse_args())
