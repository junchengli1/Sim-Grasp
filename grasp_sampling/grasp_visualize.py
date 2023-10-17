import open3d as o3d
import glob
import pickle
import numpy as np
import sys
import copy

def create_mesh_cylinder_detection(R, t, collision, radius=0.5, height=5):
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
    vertices[:, 0] += height / 2
    vertices = np.dot(R, vertices.T).T + t
    
    cylinder.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    if collision:
        colors = np.array([1, 0, 0])
    else:
        colors = np.array([0, 1, 0])
    colors = np.expand_dims(colors, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)

    return cylinder

display=[]

list=glob.glob("/media/juncheng/jason-msral/iros/"+ f"stage_{sys.argv[1]}"+"/**/*.pcd", recursive=True)

list.remove("/media/juncheng/jason-msral/iros/"+ f"stage_{sys.argv[1]}"+"/ground.pcd")
#if sys.argv[2]=="simulation":
candidate1="/media/juncheng/jason-msral/iros/"+f"stage_{sys.argv[1]}"+ f"/stage_{sys.argv[1]}_grasp_candidates_after_overlap_roq.pkl"
#if sys.argv[2]=="sampling":
#print("This is the name of the script:", sys.argv[1])
#print("Number of arguments:", len(sys.argv))
#print("The arguments are:" , str(sys.argv))

with open(candidate1, 'rb') as f:
    candidates1= pickle.load(f)
#print(candidates.keys())
#print(candidates)
count_good=0
count_bad=0
hand_mesh=o3d.io.read_triangle_mesh("/home/juncheng/Documents/symbol.ply")

print(candidates1.keys())
for object_index in candidates1.keys():
    print(candidates1[object_index].keys())
    translation_candidates1=candidates1[object_index]["translation_after_overlap_pass"]
    rotation_candidates1=candidates1[object_index]["rotation_after_overlap_pass"]

    #count_good+=len(translation_candidates1)
    #print(len(translation_candidates))
    translation_candidates_bad1=candidates1[object_index]["translation_after_overlap_fail"]
    rotation_candidates_bad1=candidates1[object_index]["rotation_after_overlap_fail"]
    #print(len(translation_candidates_bad))
    count_bad+=len(translation_candidates_bad1)
    count_good+=len(translation_candidates1)
   
    for i in range(len(translation_candidates_bad1)):

        t_bad=translation_candidates_bad1[i]
        R_bad=rotation_candidates_bad1[i]
        collision=True
        T = np.eye(4)
        T[:3, :3] = R_bad
        T[0, 3] = t_bad[0]
        T[1, 3] = t_bad[1]
        T[2, 3] = t_bad[2]

        hand_mesh.paint_uniform_color([0, 0, 0.2])

        mesh_t = copy.deepcopy(hand_mesh).transform(T)
        #line_t=o3d.geometry.LineSet.create_from_triangle_mesh(mesh_t)
        #display.append(mesh_t)

    for i in range(len(translation_candidates1)):
            t=translation_candidates1[i]
            R=rotation_candidates1[i]
            #print(R,t)
            collision=False

            T = np.eye(4)
            T[:3, :3] = R
            T[0, 3] = t[0]
            T[1, 3] = t[1]
            T[2, 3] = t[2]
            new=copy.deepcopy(hand_mesh)


            #new.paint_uniform_color([0, 0.7, 0])
            mesh_t = new.transform(T)
            new.paint_uniform_color([0, 1, 0])


            #line_t=o3d.geometry.LineSet.create_from_triangle_mesh(mesh_t)
            #display.append(line_t)
            #display=[]
        #mesh=create_mesh_cylinder_detection(R_bad,t_bad,collision)
            display.append(mesh_t)
            
            
            print(object_index,i)
            #o3d.visualization.draw_geometries_with_custom_animation(display,width=720,height=720)


            #mesh=create_mesh_cylinder_detection(R_bad,t_bad,collision)
            #display.append(mesh)
    
for i_1 in list:

                pcd = o3d.io.read_point_cloud(i_1)
                display.append(pcd)    
print(count_good,count_bad)

o3d.visualization.draw_geometries_with_custom_animation(display,width=720,height=720)










