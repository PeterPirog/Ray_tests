"""
import time
import ray

print(dir(ray))
#ray.init()
ray.init(address='192.168.1.16:6379', _redis_password='5241590000000000')
@ray.remote
def f():
    time.sleep(0.01)
    #return ray.services.get_node_ip_address()
    return ray.nodes()

# Get a list of the IP addresses of the nodes that have joined the cluster.

print(set(ray.get([f.remote() for _ in range(1000)])))
print(ray.global_state.cluster_resources())

"""


import ray
import time
#ray.init(address='192.168.1.16:6379', _redis_password='5241590000000000')
ray.init()
@ray.remote
def f(i):
    time.sleep(1)
    return i

futures = [print(ray.get(f.remote(i))) for i in range(200)]
#print(ray.get(futures))
#print(ray.global_state.cluster_resources())