from omni.isaac.kit import SimulationApp
CONFIG = {"headless":True}
simulation_app=SimulationApp(launch_config=CONFIG)
import omni.usd
import omni.client

from pxr import UsdGeom, Sdf
import os 
import glob
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim,delete_prim
from omni.physx.scripts import utils
from pxr import Gf, Sdf, UsdGeom, UsdShade, Semantics, UsdPhysics
from pxr import UsdGeom, Usd
from omni.usd.commands.usd_commands import MovePrimCommand
from pathlib import Path
from pxr import Usd, UsdGeom, Gf, Vt,Sdf,UsdUtils
import argparse
import re

base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='convert instanceable')
parser.add_argument('--headless', type=bool, default=False, help='headless')#make it True when gerneate dataset 
parser.add_argument('--data_path', type=str, default=(base_dir.parent / "synthetic_data_grasp_augment").as_posix(), help='data path')
parser.add_argument('--asset_path2', type=str, default=(base_dir.parent / 'google_usd/*/*.usd').as_posix(), help='shapenet (subset)')

def ensure_valid_object_path(s):
    # Check if the string matches the pattern with an extra segment
    match = re.match(r'^(/World/objects/object_\d+)/.*$', s)
    if match:
        # If matched, return only the desired segment
        return match.group(1)
    return s
def has_extra_segment(s):
    # Check if the string matches the pattern with an extra segment
    match = re.match(r'^/World/objects/object_\d+/.+$', s)
    return bool(match)  # Returns True if there's an extra segment, False otherwise
def hasSchema(prim, schemaName):
    schemas = prim.GetAppliedSchemas()
    for s in schemas:
        if s == schemaName:
            return True
    return False

def create_parent_xforms(asset_usd_path, source_prim_path, save_as_path=None):
    """ Adds a new UsdGeom.Xform prim for each Mesh/Geometry prim under source_prim_path.
        Moves material assignment to new parent prim if any exists on the Mesh/Geometry prim.

        Args:
            asset_usd_path (str): USD file path for asset
            source_prim_path (str): USD path of root prim
            save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
    """
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()

    prims = [stage.GetPrimAtPath(source_prim_path)]
    edits = Sdf.BatchNamespaceEdit()
    while len(prims) > 0:
        prim = prims.pop(0)
        #print(prim)
        if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
            new_xform = UsdGeom.Xform.Define(stage, str(prim.GetPath()) + "_xform")
            print(prim, new_xform)
            edits.Add(Sdf.NamespaceEdit.Reparent(prim.GetPath(), new_xform.GetPath(), 0))
            MovePrimCommand(prim.GetPath(), new_xform.GetPath())
            continue

        children_prims = prim.GetChildren()
        prims = prims + children_prims

    stage.GetRootLayer().Apply(edits)

    if save_as_path is None:
        omni.usd.get_context().save_stage()
    else:
        omni.usd.get_context().save_as_stage(save_as_path)

def convert_asset_instanceable(asset_usd_path, source_prim_path, save_as_path=None, create_xforms=False):

    """ Makes all mesh/geometry prims instanceable.
        Can optionally add UsdGeom.Xform prim as parent for all mesh/geometry prims.
        Makes a copy of the asset USD file, which will be used for referencing.
        Updates asset file to convert all parent prims of mesh/geometry prims to reference cloned USD file.

        Args:
            asset_usd_path (str): USD file path for asset
            source_prim_path (str): USD path of root prim
            save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
            create_xforms (bool): Whether to add new UsdGeom.Xform prims to mesh/geometry prims.
    """

    if create_xforms:
        create_parent_xforms(asset_usd_path, source_prim_path, save_as_path)
        asset_usd_path = save_as_path

    #instance_usd_path = ".".join(asset_usd_path.split(".")[:-1]) + "_meshes.usd"
    #if os.path.exists(instance_usd_path):
        #os.remove(instance_usd_path)
    #omni.client.copy(asset_usd_path, instance_usd_path)
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(root_prim)
    delete_prim("/World/groundPlane")

    prims = [stage.GetPrimAtPath(source_prim_path)]
    curr_prim = [stage.GetPrimAtPath("/World")]
    
    obj_prim = stage.GetPrimAtPath("/World/objects")



    #for prim in Usd.PrimRange(obj_prim):
    #          if (prim.IsA(UsdGeom.Xform)):
                
    #              utils.setRigidBody(prim, "convexDecomposition", True)
     
    #              mass_api=UsdPhysics.MassAPI.Apply(prim)
    #              mass_api.CreateDensityAttr(0.0001)
      
    for prim in Usd.PrimRange(obj_prim):
            if (prim.IsA(UsdGeom.Mesh)):
                utils.setCollider(prim, approximationShape="convexDecomposition")
   
     
    #omni.usd.get_context().save_as_stage(instance_usd_path)
    (all_layers, all_assets, unresolved_paths) = UsdUtils.ExtractExternalReferences(stage.GetRootLayer().identifier)
        #print(stage.GetRootLayer().identifier)
        #(all_layers, all_assets, unresolved_paths) = UsdUtils.ExtractExternalReferences(stage.GetRootLayer().identifier)
    unresolved_paths = unresolved_paths[::-1]

    #print(all_assets)

    path_number_dict = {}

    # Iterate over each file in the folder
    folder_path=args.data_path+f"/stage_{stage_ind}"
    for filename in os.listdir(folder_path):
        if filename.endswith(".pcd"):
            # Use regex to extract the integer number
            match = re.search(r'_(\d+).pcd$', filename)
            if match:
                number = int(match.group(1))
                # Extract the base name (without extension and number) to match with paths
                base_name = filename.split(f"_{number}")[0]
                # Find the corresponding path
                for path in all_assets:
                    if base_name.lower() in path.lower():
                        path_key = f"/World/objects/object_{number - 1}"

                        path_number_dict[path_key] = path
                        break

    #print(path_number_dict)

    if len(curr_prim)!=0:
        prims = curr_prim

    while len(prims) > 0:
        prim = prims.pop(0)
        if prim:
            if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
                parent_prim = prim.GetParent()
                obj_parent_prim = parent_prim.GetParent()
                #print(obj_parent_prim.GetPrimStack()[0].referenceList.prependedItems)
                #path=(obj_parent_prim.GetPrimStack()[0].referenceList.prependedItems[0])
                ref=obj_parent_prim.GetReferences()
                #stage.ExportToString(prim.GetPath())
                string_obj=str(ref.GetPrim().GetPath())
                #print(ref.GetPrim().GetPath())
                flag=has_extra_segment(string_obj)
                if flag:
                    parent_prim = parent_prim.GetParent()
                string_obj=ensure_valid_object_path(string_obj)

                #ref.RemoveReference(path)
                ref.ClearReferences()
                
                obj_prim_path=stage.GetPrimAtPath(string_obj)
                mass_api=UsdPhysics.MassAPI.Apply(obj_prim_path)
                mass_api.CreateDensityAttr(0.00005)

                #obj_prim_path.GetChildren()
                obj_prim_path.GetReferences().ClearReferences()
                utils.removeRigidBody(parent_prim)
                if parent_prim and not parent_prim.IsInstance():
                    if string_obj in path_number_dict:
                        assetpath=path_number_dict[string_obj]
                        
                        parent_prim.GetReferences().AddReference(assetPath=assetpath)
                        parent_prim.SetInstanceable(True)
                    else:
                        #print(string_obj)
                        #path=(obj_parent_prim.GetPrimStack()[0].referenceList.prependedItems[0])
                        #parent_prim.GetReferences().AddReference(assetPath=path.assetpath)
                        parent_prim.SetInstanceable(True)
    
                    continue
                
                    
            children_prims = prim.GetChildren()
            prims = prims + children_prims

    if save_as_path is None:
        omni.usd.get_context().save_stage()
    else:
        omni.usd.get_context().save_as_stage(save_as_path)
if __name__ == "__main__":
    # Run the main function
    args = parser.parse_args()
    objects_asset_paths2=glob.glob(args.asset_path2)
    #print(objects_asset_paths2)
    for stage_ind in range(500,1000):
        print(stage_ind)
        asset_usd_path=args.data_path+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}.usd"
        save_as_path=args.data_path+f"/stage_{stage_ind}"+f"/stage_{stage_ind}_instanceable.usd"
        source_prim_path="/World"
        convert_asset_instanceable(asset_usd_path,source_prim_path,save_as_path)
      
      

