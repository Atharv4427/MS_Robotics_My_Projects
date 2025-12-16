# 3D 2-link arm with 3 DOF (base yaw, base pitch, elbow/pitch) and 3D obstacles
# This code plots a 2-link manipulator in 3D:
# - Link 1 is oriented by (yaw, pitch) from the base
# - Link 2 is attached to the end of link 1 and rotates by `joint_angle` (pitch in local frame)
# Obstacles: spheres and axis-aligned boxes. Simple collision check is performed by sampling points along each link.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from collections import namedtuple

def rodrigues_rotation_matrix(axis, theta):
    """Return rotation matrix that rotates by theta around given axis (3-vector)."""
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)  ],
                     [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)  ],
                     [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc]])

def compute_arm_points(l1, l2, yaw, pitch, joint_angle):
    # Link 1 direction in world frame (spherical coordinates)
    # yaw = rotation around z (azimuth), pitch = elevation from XY plane
    dx1 = math.cos(pitch) * math.cos(yaw)
    dy1 = math.cos(pitch) * math.sin(yaw)
    dz1 = math.sin(pitch)
    dir1 = np.array([dx1, dy1, dz1])
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = p0 + l1 * dir1

    # Create rotation matrix that maps local x-axis (1,0,0) to dir1
    # Find rotation axis = cross([1,0,0], dir1)
    x_axis = np.array([1.0, 0.0, 0.0])
    v = np.cross(x_axis, dir1)
    if np.linalg.norm(v) < 1e-8:
        # dir1 is (nearly) aligned with x-axis (no rotation or 180 deg)
        if np.dot(x_axis, dir1) > 0:
            R = np.eye(3)
        else:
            # 180 deg rotation around any perpendicular axis, choose z
            R = rodrigues_rotation_matrix(np.array([0,0,1]), np.pi)
    else:
        angle = math.acos(np.dot(x_axis, dir1) / (np.linalg.norm(x_axis)*np.linalg.norm(dir1)))
        R = rodrigues_rotation_matrix(v, angle)

    # In the local frame attached to link1, place link2 oriented by joint_angle in x-z plane:
    # local vector for link2 (pitch): rotate from +x toward +z by joint_angle
    local_link2 = np.array([math.cos(joint_angle)*l2, 0.0, math.sin(joint_angle)*l2])
    dir2_world = R.dot(local_link2)
    p2 = p1 + dir2_world

    return p0, p1, p2

def sample_segment(p_start, p_end, n=50):
    t = np.linspace(0, 1, n)
    return np.vstack([p_start + (p_end - p_start) * ti for ti in t])

def point_in_box(pt, box_min, box_max):
    return np.all(pt >= box_min - 1e-9) and np.all(pt <= box_max + 1e-9)

def segment_collides_obstacles(p_start, p_end, obstacles, n_samples=100):
    pts = sample_segment(p_start, p_end, n_samples)
    for i, obs in enumerate(obstacles):
        if obs['type'] == 'sphere':
            center = np.array(obs['center'])
            r = obs['radius']
            d2 = np.sum((pts - center)**2, axis=1)
            if np.any(d2 <= r**2):
                return True, i+1
        elif obs['type'] == 'box':
            mn = np.array(obs['min'])
            mx = np.array(obs['max'])
            inside = np.all((pts >= mn) & (pts <= mx), axis=1)
            if np.any(inside):
                return True, i+1
    return False, 0

def plot_arm_and_obstacles(link_len, yaw, pitch, joint_angle, obstacles):
    l1, l2 = link_len
    p0, p1, p2 = compute_arm_points(l1, l2, yaw, pitch, joint_angle)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    # Plot links as lines (default color cycle will be used)
    xs = [p0[0], p1[0], p2[0]]
    ys = [p0[1], p1[1], p2[1]]
    zs = [p0[2], p1[2], p2[2]]
    ax.plot(xs[:2], ys[:2], zs[:2], linewidth=3)
    ax.plot(xs[1:], ys[1:], zs[1:], linewidth=3)

    # Plot joint markers
    ax.scatter([p0[0]], [p0[1]], [p0[2]], s=40)
    ax.scatter([p1[0]], [p1[1]], [p1[2]], s=40)
    ax.scatter([p2[0]], [p2[1]], [p2[2]], s=40)

    # Plot obstacles (spheres as wireframe, boxes as edges)
    for obs in obstacles:
        if obs['type'] == 'sphere' or obs['type'] == 'goal':
            center = np.array(obs['center'])
            r = obs['radius']
            u = np.linspace(0, 2*np.pi, 24)
            v = np.linspace(0, np.pi, 12)
            x = center[0] + r * np.outer(np.cos(u), np.sin(v))
            y = center[1] + r * np.outer(np.sin(u), np.sin(v))
            z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
            if obs['type'] == 'sphere':
                ax.plot_wireframe(x, y, z, linewidth=0.5)
            else:
                ax.plot_surface(x, y, z, linewidth=0.5, color = 'g')
        elif obs['type'] == 'box':
            mn = np.array(obs['min'])
            mx = np.array(obs['max'])
            # 8 corners
            corners = np.array([[mn[0], mn[1], mn[2]],
                                [mx[0], mn[1], mn[2]],
                                [mx[0], mx[1], mn[2]],
                                [mn[0], mx[1], mn[2]],
                                [mn[0], mn[1], mx[2]],
                                [mx[0], mn[1], mx[2]],
                                [mx[0], mx[1], mx[2]],
                                [mn[0], mx[1], mx[2]]])
            # edges to draw
            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)
            ]
            lines = [(corners[a], corners[b]) for (a,b) in edges]
            lc = Line3DCollection(lines, linewidths=1)
            ax.add_collection3d(lc)

    # Simple collision check
    coll1, idx1 = segment_collides_obstacles(p0, p1, obstacles)
    coll2, idx2 = segment_collides_obstacles(p1, p2, obstacles)
    text_lines = [
        f"Link lengths: {l1}, {l2}",
        f"Yaw (deg): {np.degrees(yaw):.1f}, Pitch (deg): {np.degrees(pitch):.1f}, Elbow pitch (deg): {np.degrees(joint_angle):.1f}",
        f"Link1 collision: {'YES (obs '+str(idx1)+')' if coll1 else 'NO'}",
        f"Link2 collision: {'YES (obs '+str(idx2)+')' if coll2 else 'NO'}"
    ]
    ax.text2D(0.02, 0.95, "\n".join(text_lines), transform=ax.transAxes)

    # Axis labels and view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = max(np.ptp(xs+ys+zs), 1.0)

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_title('2-link Arm (3 DOF) in 3D with Obstacles')
    plt.show()


# Example usage
link_len = [3.0, 3.0]
yaw = np.radians(0)       # base yaw
pitch = np.radians(0)     # base pitch (elevation)
joint_angle = np.radians(0)  # elbow/pitch relative to link1

goal = {'center':[2.0, 4.0, 0.5], 'radius':1}
# Define obstacles: sphere and axis-aligned box
obstacles = [
    {'type':'sphere', 'center':[2.0, 1.0, 0.8], 'radius':0.5},
    {'type':'box', 'min':[4.0, -1.0, 0.5], 'max':[5.0, 1.0, 2.0]},
    {'type': 'goal', 'center':goal['center'], 'radius':goal['radius']}
]
plot_arm_and_obstacles(link_len, yaw, pitch, joint_angle, obstacles)

# You can change yaw, pitch, joint_angle or obstacles and re-run to visualize different configurations.
res_pts = 40
# Define the range of angles
angles = np.linspace(0, 2*np.pi, num=res_pts)

# Initialize array to store collision information
collision_array = np.zeros((len(angles), len(angles), len(angles)), dtype=int)



# Iterate over angles and check for collisions
l1, l2 = link_len
count = 0
for i, theta1 in enumerate(angles):
    for j, theta2 in enumerate(angles):
        for k, theta3 in enumerate(angles):
            p0, p1, p2 = compute_arm_points(l1, l2, theta1, theta2, theta3)
            temp, a = segment_collides_obstacles(p0, p1, obstacles)
            if a > 0:
                collision_array[i, j, k] = a
                continue
            temp, b = segment_collides_obstacles(p1, p2, obstacles)
            if b > 0:
                collision_array[i, j, k] = b
                continue
            if np.linalg.norm(p2 - np.array(goal['center'])) <= int(goal['radius']) :
                count += 1
                collision_array[i, j, k] = len(obstacles)
            # Check if the end effector is inside the goal sphere

obstacle_colors = ['red', 'purple', 'green']

def plot_collision_3d(ax, collision_array, angles, obstacle_colors):
    # theta1, theta2, theta3 = np.meshgrid(angles, angles, angles, indexing='ij')

    # Plot each point with color based on collision information
    for i in range(len(angles)):
        for j in range(len(angles)):
            for k in range(len(angles)):
                color_index = collision_array[i, j, k] - 1  # Subtract 1 to match the index of colors list
                if (color_index == -1 ):
                    continue
                ax.scatter(angles[i], angles[j], angles[k], color=obstacle_colors[color_index], s=(2 + 3*(collision_array[i, j, k]/len(obstacle_colors)) ))

    ax.set_xlabel('Theta1')
    ax.set_ylabel('Theta2')
    ax.set_zlabel('Theta3')
    ax.set_title('Collision Information in 3D')
    return ax



Node = namedtuple('Node', ['pt', 'parent', 'cost'])  # pt is np.array([x,y,z])

def find_goal(collision_array):
    inds = np.argwhere(collision_array == 3)
    if inds.shape[0] == 0:
        return None
    return inds[0].astype(float)

def is_in_bounds(pt, shape):
    return all(0 <= pt[i] < shape[i] for i in range(3))

def collision_check_segment(p1, p2, collision_array, step=0.01):
    """Return True if the segment collides with obstacles (values 1 or 2) or leaves bounds."""
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length == 0:
        idx = tuple(np.round(p1).astype(int))
        if not is_in_bounds(idx, collision_array.shape):
            return True
        return collision_array[idx] in (1,2)
    n_samples = max(2, int(math.ceil(length / step)))
    for t in np.linspace(0, 1, n_samples):
        pt = p1 + t * vec
        idx = tuple(np.round(pt).astype(int))
        if not is_in_bounds(idx, collision_array.shape):
            return True  # out-of-bounds = collision
        val = collision_array[idx]
        if val in (1,2):
            return True
    return False

def nearest_node_index(tree, sample_pt):
    dists = [np.linalg.norm(n.pt - sample_pt) for n in tree]
    return int(np.argmin(dists))

def steer(from_pt, to_pt, step_size):
    vec = to_pt - from_pt
    dist = np.linalg.norm(vec)
    if dist <= step_size or dist == 0.0:
        return to_pt.copy()
    return from_pt + (vec / dist) * step_size

def get_near_indices(tree, new_pt, radius):
    return [i for i,n in enumerate(tree) if np.linalg.norm(n.pt - new_pt) <= radius]

def rrt_star(collision_array, start=(0,0,0), max_iters=2000, step_size=2.0, search_radius=3.0, goal_sample_rate=0.05):
    """
    Returns: tree (list of Nodes), path (list of pts) or None, goal_pt
    """
    shape = collision_array.shape
    goal_pt = find_goal(collision_array)
    if goal_pt is None:
        raise ValueError("No goal (value 3) found in collision_array")
    start = np.array(start, dtype=float)
    if collision_array[tuple(np.round(start).astype(int))] in (1,2):
        raise ValueError("Start is in collision")
    tree = [Node(pt=start, parent=None, cost=0.0)]
    goal_idx_in_tree = None

    for it in range(max_iters):
        # biased sampling towards goal by goal_sample_rate
        if random.random() < goal_sample_rate:
            sample = goal_pt.copy()
        else:
            sample = np.array([random.uniform(0, shape[0]-1),
                               random.uniform(0, shape[1]-1),
                               random.uniform(0, shape[2]-1)])
        nn_idx = nearest_node_index(tree, sample)
        nn = tree[nn_idx]
        new_pt = steer(nn.pt, sample, step_size)
        # collision test from nearest node to new_pt
        if collision_check_segment(nn.pt, new_pt, collision_array):
            continue
        # find near nodes
        near_idxs = get_near_indices(tree, new_pt, search_radius)
        # Choose best parent among near nodes (including nn)
        min_cost = nn.cost + np.linalg.norm(new_pt - nn.pt)
        min_parent = nn_idx
        for i in near_idxs:
            if not collision_check_segment(tree[i].pt, new_pt, collision_array):
                cost = tree[i].cost + np.linalg.norm(tree[i].pt - new_pt)
                if cost < min_cost:
                    min_cost = cost
                    min_parent = i
        new_node = Node(pt=new_pt, parent=min_parent, cost=min_cost)
        tree.append(new_node)
        new_idx = len(tree) - 1
        # Rewiring: try to improve near nodes' cost through new_node
        for i in near_idxs:
            if i == min_parent:
                continue
            if collision_check_segment(tree[i].pt, new_pt, collision_array):
                continue
            new_cost = new_node.cost + np.linalg.norm(tree[i].pt - new_node.pt)
            if new_cost < tree[i].cost:
                tree[i] = Node(pt=tree[i].pt, parent=new_idx, cost=new_cost)
        # Try to connect to goal
        if np.linalg.norm(new_pt - goal_pt) <= step_size and not collision_check_segment(new_pt, goal_pt, collision_array):
            goal_node = Node(pt=goal_pt.copy(), parent=new_idx, cost=new_node.cost + np.linalg.norm(goal_pt - new_pt))
            tree.append(goal_node)
            goal_idx_in_tree = len(tree) - 1
            break

    # extract path
    path = None
    if goal_idx_in_tree is not None:
        path = []
        cur = goal_idx_in_tree
        while cur is not None:
            path.append(tree[cur].pt)
            cur = tree[cur].parent
        path.reverse()
    return tree, path, goal_pt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = plot_collision_3d(ax, collision_array, angles, obstacle_colors)

start = (0, 0, 0)
tree, path, goal_pt = rrt_star(
    collision_array, start=start,
    max_iters=500000, step_size=2, search_radius=8.0, goal_sample_rate=0.05
)

# Convert goal_pt to integer indices in a single line
i, j, k = goal_pt.astype(int)
print(f"Goal point indices (i, j, k): {i}, {j}, {k}")

# Use the angle values corresponding to the indices in goal_pt
ax.scatter(angles[i], angles[j], angles[k], color='green', s=80, marker='*', label='Goal')
ax.scatter([angles[int(start[0])]], [angles[int(start[1])]], [angles[int(start[2])]], color='blue', s=50, marker='o', label='Start')

for node in tree:
    if node.parent is None:
        continue
    p = node.pt
    q = tree[node.parent].pt
    # Use angle values for plotting the tree
    ax.plot([angles[int(p[0])]], [angles[int(q[0])]], [angles[int(p[1])]], [angles[int(q[1])]], [angles[int(p[2])]], [angles[int(q[2])]], linewidth=0.5, alpha=0.6)

if path is not None:
    path = np.array(path)
    # Use angle values for plotting the path
    ax.plot(angles[path[:,0].astype(int)], angles[path[:,1].astype(int)], angles[path[:,2].astype(int)], linewidth=3, color='yellow', label='Planned path')
    ax.scatter(angles[path[:,0].astype(int)], angles[path[:,1].astype(int)], angles[path[:,2].astype(int)], color='blue', s=20)
    print(f"Path found with {len(path)} waypoints. Path length: {sum(np.linalg.norm(path[i+1]-path[i]) for i in range(len(path)-1)):.2f}")
else:
    print("No path found within iteration limit.")

# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.title('RRT* on 3D occupancy grid (tree in gray, path in red)')
plt.show()

for i, waypoint in enumerate(path):
    # Extract the angle indices from the waypoint
    yaw_idx, pitch_idx, joint_angle_idx = waypoint.astype(int)

    # Convert indices to actual angle values using the 'angles' array
    current_yaw = angles[yaw_idx]
    current_pitch = angles[pitch_idx]
    current_joint_angle = angles[joint_angle_idx]

    print(f"Plotting waypoint {i+1}/{len(path)}: Yaw={np.degrees(current_yaw):.1f} deg, Pitch={np.degrees(current_pitch):.1f} deg, Joint Angle={np.degrees(current_joint_angle):.1f} deg")
    # Plot the arm configuration for the current waypoint
    plot_arm_and_obstacles(link_len, current_yaw, current_pitch, current_joint_angle, obstacles)