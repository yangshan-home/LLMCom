
def main(edges_path, attributes_path):
    with open(edges_path, 'r', encoding='utf-8') as f_edges:
        edges_lines = f_edges.readlines()
        edges_list = []
        for edge in edges_lines:
            edge = edge.strip()

    pass


if __name__ == '__main__':
    edges_path = './real_network/non-overlapping/citeseer-doc-classification/citeseer.cites'
    attributes_path = './real_network/non-overlapping/citeseer-doc-classification/citeseer.content'
    main(edges_path, attributes_path)