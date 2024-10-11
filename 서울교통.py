import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt


# 특정 좌표 범위 또는 주소 값을 이용해 도로망 데이터를 추출하는 함수
def extract_road_graph_by_bbox(north, south, east, west):
    # 좌표 범위를 지정하여 도로망 그래프 추출
    G = ox.graph_from_bbox(north, south, east, west, network_type='drive')

    # 그래프를 시각화
    ox.plot_graph(G, bgcolor='white', node_color='red', edge_color='blue', node_size=10, edge_linewidth=1)

    return G


# 특정 주소를 이용해 도로망 데이터를 추출하는 함수
def extract_road_graph_by_address(address, dist=1000):
    # 주소와 반경을 지정하여 도로망 그래프 추출
    G = ox.graph_from_address(address, dist=dist, network_type='drive')

    # 그래프를 시각화
    ox.plot_graph(G, bgcolor='white', node_color='red', edge_color='blue', node_size=10, edge_linewidth=1)

    return G


# 버텍스와 엣지를 정의하여 그래프 생성하는 방법
def define_vertices_edges(graph):
    # 노드(버텍스)와 엣지 정보 추출
    nodes, edges = ox.graph_to_gdfs(graph)

    # 노드와 엣지의 데이터 출력
    print("Vertices (Nodes):")
    print(nodes.head())
    print("\nEdges:")
    print(edges.head())


# 예시: 서울특별시 도로망 추출 및 시각화 (주소 기반)
place_name = "Seoul, South Korea"
north, south, east, west = 37.57, 37.55, 127.03, 126.97

# 좌표 범위를 이용해 도로망 추출
road_graph_bbox = extract_road_graph_by_bbox(north, south, east, west)

# 주소를 이용해 도로망 추출
road_graph_address = extract_road_graph_by_address(place_name)

# 버텍스와 엣지 정의 및 정보 출력
define_vertices_edges(road_graph_bbox)

# 그래프 정보를 NetworkX를 이용해 추가 분석 가능
print(nx.info(road_graph_bbox))