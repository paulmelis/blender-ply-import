# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8-80 compliant>

bl_info = {
    "name": "Stanford PLY format (alternative)",
    "author": "Paul Melis",
    "blender": (2, 75, 0),
    "location": "File > Import-Export",
    "description": "Import-Export PLY mesh data, including vertex colors",
    "warning": "",
    "wiki_url": "",                
    "support": 'OFFICIAL',
    "category": "Import-Export"}

import os, time
import bpy

from bpy.props import (
        CollectionProperty,
        StringProperty,
        BoolProperty,
        EnumProperty,
        FloatProperty,
        )
from bpy_extras.io_utils import (
        ImportHelper,
        orientation_helper_factory,
        axis_conversion,
        )


#IOPLYOrientationHelper = orientation_helper_factory("IOPLYOrientationHelper", axis_forward='Y', axis_up='Z')

from readply import readply

def load_ply_mesh(filepath, ply_name):    
    
    # XXX call needs update for new API!
    num_vertices, num_faces, varray, farray, vnarray, vcolarray = readply(filepath)

    # Create a mesh + object using the binary vertex and face data

    mesh = bpy.data.meshes.new(name=ply_name)

    mesh.vertices.add(num_vertices)
    mesh.vertices.foreach_set('co', varray)

    mesh.tessfaces.add(num_faces)
    mesh.tessfaces.foreach_set('vertices_raw', farray)

    mesh.validate()
    mesh.update()

    if vcolarray is not None:
        
        """    
        # For each face, set the vertex colors of the vertices making up that face
        for fi in range(num_faces):
            
            # Get vertex indices for this triangle/quad
            i, j, k, l = farray[4*fi:4*fi+4]
            
            face_col = vcol_data[fi]
            face_col.color1 = vcolarray[3*i:3*i+3]
            face_col.color2 = vcolarray[3*j:3*j+3]
            face_col.color3 = vcolarray[3*k:3*k+3]
            if l != 0:
                face_col.color4 = vcolarray[3*l:3*l+3]
        """
        
        vcol_layer = mesh.vertex_colors.new()
        vcol_data = vcol_layer.data
        vcol_data.foreach_set('color', vcolarray)

    if vnarray is not None:
        print('Warning: NOT applying vertex normals (yet)')
        
    mesh.validate()
    mesh.update()
    
    return mesh


def load_ply(context, filepath):
    
    t = time.time()
    ply_name = bpy.path.display_name_from_filepath(filepath)

    mesh = load_ply_mesh(filepath, ply_name)

    scn = bpy.context.scene

    obj = bpy.data.objects.new(ply_name, mesh)
    scn.objects.link(obj)
    scn.objects.active = obj
    obj.select = True

    print('\nSuccessfully imported %r in %.3f sec' % (filepath, time.time() - t))
    

class ImportPLY(bpy.types.Operator, ImportHelper):
    """Load a PLY geometry file"""
    bl_idname = "import_mesh.ply2"
    bl_label = "Import PLY (alternative)"
    bl_description = 'Alternative importer for PLY files'
    bl_options = {'UNDO'}

    files = CollectionProperty(
        name="File Path",
        description="File path used for importing the PLY file",
        type=bpy.types.OperatorFileListElement
        )

    directory = StringProperty()

    filename_ext = ".ply"
    filter_glob = StringProperty(default="*.ply", options={'HIDDEN'})

    def execute(self, context):
        paths = [os.path.join(self.directory, name.name)
                 for name in self.files]
                 
        if not paths:
            paths.append(self.filepath)

        for path in paths:
            load_ply(context, path)
            
        bpy.context.scene.update()

        return {'FINISHED'}

def menu_func_import(self, context):
    self.layout.operator(ImportPLY.bl_idname, text="Stanford PLY [ALTERNATIVE] (.ply)")

def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()
