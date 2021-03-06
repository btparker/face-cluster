""" Face Cluster """

def _chinese_whispers(encoding_list, threshold=0.6, iterations=10):
    """ Chinese Whispers Algorithm

    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/

    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate

    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """
    from random import shuffle
    import networkx as nx
    from face_recognition.api import _face_distance

    # Create graph
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

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

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

    if len(facial_encodings) <= 1:
        print "Number of facial encodings must be greater than one, can't cluster"
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings.items())
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
    for idx, image_path in enumerate(image_paths):
        print "Encoding '{}' ...".format(image_path)
        picture = face_recognition.load_image_file(image_path)

        # Find all the faces in the image
        face_locations = face_recognition.face_locations(picture)

        if len(face_locations) == 1:
            picture_face_encodings = face_recognition.face_encodings(picture)

            if len(picture_face_encodings) == 1:
                facial_encodings[image_path] = picture_face_encodings[0]
                print "... stored"

            else:
                print "... did not encode face, skipping"

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

    print "Created {} clusters:".format(len(sorted_clusters))
    for idx, cluster in enumerate(sorted_clusters):
        print "   - cluster {} size {}".format(idx, len(cluster))

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

    print "Saved results to {}".format(args.output)

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
