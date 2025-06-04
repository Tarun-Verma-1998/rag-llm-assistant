from pymilvus import connections, list_collections

connections.connect("default", host="127.0.0.1", port="19530")
print(list_collections())