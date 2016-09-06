/*
TODO:
- check why we return triangle/quad indices as int32 arrays, and not uint32
- need to handle faces with more than 4 vertices
- we assume property order in the file is always x,y,z. Need good way to handle other orders
- add parameter to specify if returned vertex color array is
  blender-style (color per vertex per face) or plain per-vertex
- comment out printf()s
- double-check that we are really ignoring faces with >4 vertices. 
  It seems indices of such a face *are* added to the index list...
*/

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <rply.h>
#include <cstdio>
#include <cassert>

//#define DEBUG

//
// Custom object type to handle correct deallocation
// After http://blog.enthought.com/general/numpy-arrays-with-pre-allocated-memory/
//

typedef struct
{
    PyObject_HEAD
    void *memory;
#ifdef DEBUG
    const char *name;
#endif
}
_MyDeallocObject;

static void
_mydealloc_dealloc(_MyDeallocObject *self)
{
#ifdef DEBUG
    fprintf(stderr, "_mydealloc_dealloc() on '%s'\n", self->name);
    if (self->memory == NULL)
        fprintf(stderr, "self->memory == NULL!\n");
#endif
    
    free(self->memory);

#ifdef DEBUG
    fprintf(stderr, "Calling python type free()\n");
#endif    
    
#if PY_MAJOR_VERSION == 2
    self->ob_type->tp_free((PyObject*)self);
#elif PY_MAJOR_VERSION == 3
    Py_TYPE(self)->tp_free((PyObject*)self);
#endif
}

static PyTypeObject _MyDeallocType =
{
#if PY_MAJOR_VERSION == 2
    PyObject_HEAD_INIT(NULL)
    0,                          /*ob_size*/
#elif PY_MAJOR_VERSION == 3
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "mydeallocator",            /*tp_name*/
    sizeof(_MyDeallocObject),   /*tp_basicsize*/
    0,                          /*tp_itemsize*/
    (destructor)_mydealloc_dealloc,         /*tp_dealloc*/
    0,                          /*tp_print*/
    0,                          /*tp_getattr*/
    0,                          /*tp_setattr*/
    0,                          /*tp_compare*/
    0,                          /*tp_repr*/
    0,                          /*tp_as_number*/
    0,                          /*tp_as_sequence*/
    0,                          /*tp_as_mapping*/
    0,                          /*tp_hash */
    0,                          /*tp_call*/
    0,                          /*tp_str*/
    0,                          /*tp_getattro*/
    0,                          /*tp_setattro*/
    0,                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,             /*tp_flags*/
    "Internal deallocator object",  /* tp_doc */
};

static void 
_set_base_object(PyArrayObject *arrobj, void *memory, const char *name)
{
    // XXX check for NULL in return of PyObject_New()
    PyObject *newobj = (PyObject*)PyObject_New(_MyDeallocObject, &_MyDeallocType);
    
    ((_MyDeallocObject *)newobj)->memory = memory;
#ifdef DEBUG
    ((_MyDeallocObject *)newobj)->name = strdup(name); 
#endif
    
#if NPY_API_VERSION >= 0x00000007
    PyArray_SetBaseObject(arrobj, newobj);
#else
    PyArray_BASE(arrobj) = newobj;
#endif
}

//
// rply stuff (arrays and callbacks)
//

static float    *vertices = NULL;
static int      next_vertex_element_offset;

static int      *faces = NULL;
static int      *triangles = NULL;
static int      *quads = NULL;
static int      next_face_element_offset;
static int      num_triangles, num_quads;

static float    *vertex_normals = NULL;
static int      next_vertex_normal_element_offset;

static float    *vertex_colors = NULL;
static int      next_vertex_color_element_offset;
static float    vertex_color_scale_factor;

static float    *vertex_texcoords = NULL;
static int      next_vertex_texcoord_element_offset;

// Vertex callbacks

static int
vertex_cb(p_ply_argument argument)
{
    vertices[next_vertex_element_offset] = ply_get_argument_value(argument);
    next_vertex_element_offset++;

    return 1;
}

static int
vertex_color_cb(p_ply_argument argument)
{
    vertex_colors[next_vertex_color_element_offset] = ply_get_argument_value(argument) * vertex_color_scale_factor;
    next_vertex_color_element_offset++;

    return 1;
}

static int
vertex_normal_cb(p_ply_argument argument)
{
    vertex_normals[next_vertex_normal_element_offset] = ply_get_argument_value(argument);
    next_vertex_normal_element_offset++;

    return 1;
}

static int
vertex_texcoord_cb(p_ply_argument argument)
{
    vertex_texcoords[next_vertex_texcoord_element_offset] = ply_get_argument_value(argument);
    next_vertex_texcoord_element_offset++;

    return 1;
}

// Face callback

static int
face_cb(p_ply_argument argument)
{
    long    length, value_index;
    int     vertex_index;

    ply_get_argument_property(argument, NULL, &length, &value_index);

    if (value_index == -1)
    {
        // First value of a list property, the one that gives the number of entries
        if (length > 4)
            fprintf(stderr, "Warning: ignoring face with %ld vertices!\n", length);
        else if (length == 3)
            num_triangles++;
        else if (length == 4)
            num_quads++;

        return 1;
    }

    vertex_index = ply_get_argument_value(argument);
    faces[next_face_element_offset] = vertex_index;
    next_face_element_offset++;

    // Blender's vertices_raw array uses 4 vertex indices per face,
    // denoting either a triangle or a quad.
    // For a triangle the last (fourth) index needs to be 0.

    if (length == 3 && value_index == 2)
    {
        // Last index of triangle was just added to the index array.
        // Add extra 0 index to get to 4 indices per face
        faces[next_face_element_offset] = 0;
        next_face_element_offset++;
    }
    else if (length == 4 && value_index == 3 && vertex_index == 0)
    {
        // Handle the case when there is a quad that has indices i, j, k, 0.
        // We should cycle the indices to move the 0 out of the last place,
        // as it would otherwise get interpreted as a triangle.
        const int firstidx = next_face_element_offset-4;
        faces[firstidx+3] = faces[firstidx+2];
        faces[firstidx+2] = faces[firstidx+1];
        faces[firstidx+1] = faces[firstidx];
        faces[firstidx] = 0;
    }

    return 1;
}

// Main Python function

static PyObject*
readply(PyObject* self, PyObject* args, PyObject *kwds)
{
    char    *fname;
    int     blender_face_indices = 1;
    
    static char *kwlist[] = {"plyfile", "blender_face_indices", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|i", kwlist, &fname, &blender_face_indices))
        return NULL;
    
    // Open PLY file

    p_ply ply = ply_open(fname, NULL, 0, NULL);
    if (!ply)
    {
        char s[1024];
        sprintf(s, "Could not open PLY file %s", fname);
        PyErr_SetString(PyExc_IOError, s);
        return NULL;
    }

    if (!ply_read_header(ply))
    {
        PyErr_SetString(PyExc_IOError, "Could not read PLY header");
        ply_close(ply);
        return NULL;
    }

    // Check elements

    p_ply_element   vertex_element=NULL, face_element=NULL;
    const char      *name;

    p_ply_element element = ply_get_next_element(ply, NULL);
    while (element)
    {
        ply_get_element_info(element, &name, NULL);

        if (strcmp(name, "vertex") == 0)
            vertex_element = element;
        else if (strcmp(name, "face") == 0)
            face_element = element;

        element = ply_get_next_element(ply, element);
    }

    assert(vertex_element && "Don't have a vertex element");
    assert(face_element && "Don't have a face element");

    // Set vertex and face property callbacks
    
    // 3D coordinates

    long nvertices, nfaces;

    nvertices = ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 1);

    nfaces = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);

    printf("%ld vertices\n%ld faces\n", nvertices, nfaces);

    // Set optional per-vertex callbacks

    bool            have_vertex_colors = false;
    bool            have_vertex_normals = false;
    bool            have_vertex_texcoords = false;      // Either s,t or u,v sets will be used, but not both

    p_ply_property  prop;
    e_ply_type      ptype, plength_type, pvalue_type;

    prop = ply_get_next_property(vertex_element, NULL);
    while (prop)
    {
        ply_get_property_info(prop, &name, &ptype, &plength_type, &pvalue_type);

        printf("property '%s'\n", name);

        if (strcmp(name, "red") == 0)
        {
            // Assumes green and blue properties are also available
            have_vertex_colors = true;

            if (ptype == PLY_UCHAR)
                vertex_color_scale_factor = 1.0 / 255;
            else if (ptype == PLY_FLOAT)
                vertex_color_scale_factor = 1.0;
            else
                printf("Warning: vertex color value type is %d, don't know how to handle!\n", ptype);

            ply_set_read_cb(ply, "vertex", "red", vertex_color_cb, NULL, 0);
            ply_set_read_cb(ply, "vertex", "green", vertex_color_cb, NULL, 0);
            ply_set_read_cb(ply, "vertex", "blue", vertex_color_cb, NULL, 1);
        }
        else if (strcmp(name, "nx") == 0)
        {
            // Assumes ny and nz properties are also available
            have_vertex_normals = true;

            ply_set_read_cb(ply, "vertex", "nx", vertex_normal_cb, NULL, 0);
            ply_set_read_cb(ply, "vertex", "ny", vertex_normal_cb, NULL, 0);
            ply_set_read_cb(ply, "vertex", "nz", vertex_normal_cb, NULL, 1);
        }
        else if (strcmp(name, "s") == 0 && !have_vertex_texcoords)
        {
            // Assumes t property is also available
            have_vertex_texcoords = true;

            ply_set_read_cb(ply, "vertex", "s", vertex_texcoord_cb, NULL, 0);
            ply_set_read_cb(ply, "vertex", "t", vertex_texcoord_cb, NULL, 1);
        }
        else if (strcmp(name, "u") == 0 && !have_vertex_texcoords)
        {
            // Assumes v property is also available
            have_vertex_texcoords = true;

            ply_set_read_cb(ply, "vertex", "u", vertex_texcoord_cb, NULL, 0);
            ply_set_read_cb(ply, "vertex", "v", vertex_texcoord_cb, NULL, 1);
        }

        prop = ply_get_next_property(vertex_element, prop);
    }

    // Allocate memory and initialize

    vertices = (float*) malloc(sizeof(float)*nvertices*3);
    next_vertex_element_offset = 0;

    faces = (int*) malloc(sizeof(int)*nfaces*4);
    next_face_element_offset = 0;

    if (have_vertex_normals)
    {
        vertex_normals = (float*) malloc(sizeof(float)*nvertices*3);
        next_vertex_normal_element_offset = 0;
    }

    if (have_vertex_colors)
    {
        vertex_colors = (float*) malloc(sizeof(float)*nvertices*3);
        next_vertex_color_element_offset = 0;
    }

    if (have_vertex_texcoords)
    {
        vertex_texcoords = (float*) malloc(sizeof(float)*nvertices*2);
        next_vertex_texcoord_element_offset = 0;
    }

    // Let rply process the file using the callbacks we set

    num_triangles = num_quads = 0;

    if (!ply_read(ply))
    {
        // Failed!
        
        PyErr_SetString(PyExc_IOError, "Could not read PLY data");

        ply_close(ply);

        free(vertices);
        free(faces);

        if (have_vertex_normals)
            free(vertex_normals);
        if (have_vertex_colors)
            free(vertex_colors);
        if (have_vertex_texcoords)
            free(vertex_texcoords);

        return NULL;
    }

    printf("%d triangles, %d quads\n", num_triangles, num_quads);

    // Clean up PLY reader

    ply_close(ply);

    //
    // Create return value objects
    //

    // Vertices
    
    npy_intp np_vertices_dims[1] = { nvertices*3 };
    // XXX check for NULL in return of PyArray_SimpleNewFromData()
    PyArrayObject *np_vertices = (PyArrayObject*) PyArray_SimpleNewFromData(1, np_vertices_dims, NPY_FLOAT, vertices);    
    _set_base_object(np_vertices, vertices, "vertices");

    // Faces
    
    PyObject *np_faces;
    
    if (blender_face_indices)
    {
        // Single array holding both triangles and quads.
        // 4 indices per face, triangles always have fourth index of 0
        npy_intp np_faces_dims[1] = { nfaces*4 };
        np_faces = PyArray_SimpleNewFromData(1, np_faces_dims, NPY_INT, faces);
        _set_base_object((PyArrayObject*)np_faces, faces, "faces");
    }
    else
    {
        // Separate arrays of vertices and triangles
        
        triangles = (int*) malloc(sizeof(int)*num_triangles*3);
        quads = (int*) malloc(sizeof(int)*num_quads*4);
        
        const int *face = faces;
        int *triangle = triangles;
        int *quad = quads;
        for (int f = 0; f < nfaces; f++)
        {
            if (face[3] == 0)
            {
                // Triangle
                triangle[0] = face[0];
                triangle[1] = face[1];
                triangle[2] = face[2];
                triangle += 3;
            }
            else
            {
                // Quad
                quad[0] = face[0];
                quad[1] = face[1];
                quad[2] = face[2];
                quad[3] = face[3];
                quad += 4;
            }
            face += 4;
        }
        free(faces);
        
        npy_intp np_triangles_dims[1] = { num_triangles*3 };
        PyArrayObject *np_triangles = (PyArrayObject*) PyArray_SimpleNewFromData(1, np_triangles_dims, NPY_INT, triangles);
        _set_base_object(np_triangles, triangles, "triangles");
        
        npy_intp np_quads_dims[1] = { num_quads*4 };
        PyArrayObject *np_quads = (PyArrayObject*) PyArray_SimpleNewFromData(1, np_quads_dims, NPY_INT, quads);
        _set_base_object(np_quads, quads, "quads");
        
        np_faces = Py_BuildValue("(NN)", (PyObject*)np_triangles, (PyObject*)np_quads);
    }

    // Optional per-vertex arrays

    PyObject *np_vcolors, *np_vnormals, *np_vtexcoords;

    if (have_vertex_normals)
    {
        PyArrayObject *arr = (PyArrayObject*) PyArray_SimpleNewFromData(1, np_vertices_dims, NPY_FLOAT, vertex_normals);
        _set_base_object(arr, vertex_normals, "vertex_normals");
        np_vnormals = (PyObject*) arr;
    }
    else
    {
        // No normals
        np_vnormals = Py_None;
        Py_XINCREF(np_vnormals);
    }

    if (have_vertex_colors)
    {
        /* Direct return of per-vertex colors
        PyArrayObject *arr = (PyArrayObject*) PyArray_SimpleNewFromData(1, np_vertices_dims, NPY_FLOAT, vertex_colors);
        */

        // Colors per vertex
        // ->
        // Colors per vertex per face

        const int n = 3*((num_triangles*3)+(num_quads*4));

        float   *vcol2 = (float*) malloc (n*sizeof(float));
        float   *vcol2color = vcol2;
        float   *col;
        int     vi;

        for (int fi = 0; fi < nfaces; fi++)
        {
            const int *face = faces + 4*fi;

            for (int i = 0; i < 4; i++)
            {
                vi = face[i];

                if (i == 3 && vi == 0)
                {
                    // Triangle
                    break;
                }

                col = vertex_colors + 3*vi;

                vcol2color[0] = col[0];
                vcol2color[1] = col[1];
                vcol2color[2] = col[2];
                vcol2color += 3;
            }
        }

        free(vertex_colors);

        npy_intp    dims[1] = { n };
        PyArrayObject *arr = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, vcol2);        
        _set_base_object(arr, vcol2, "vertex_colors");
        np_vcolors = (PyObject*) arr;
    }
    else
    {
        // No vertex colors
        np_vcolors = Py_None;
        Py_XINCREF(np_vcolors);
    }

    if (have_vertex_texcoords)
    {
        npy_intp    np_vertex_texcoords_dims[1] = { nvertices*2 };

        PyArrayObject *arr = (PyArrayObject*) PyArray_SimpleNewFromData(1, np_vertex_texcoords_dims, NPY_FLOAT, vertex_texcoords);
        _set_base_object(arr, vertex_texcoords, "vertex_texcoords");
        np_vtexcoords = (PyObject*) arr;
    }
    else
    {
        // No texture coords
        np_vtexcoords = Py_None;
        Py_XINCREF(np_vtexcoords);
    }

    // Return the stuff!

    return Py_BuildValue("(iiNNNNN)",  nvertices, nfaces, np_vertices, np_faces, np_vnormals, np_vcolors, np_vtexcoords);
}

// Python module stuff

static char readply_func_doc[] = 
"readply_func_doc(plyfile, blender_face_indices=True)\n\
\n\
Reads a .PLY file. Returns a tuple:\n\
(num_vertices, num_faces, vertices, faces, vertex_normals, vertex_colors, vertex_tex_coords)\n\
\n\
The first two values will be integers, the remaining ones will be 1-dimensional Numpy arrays,\n\
except faces, which may also be a tuple of two 1-dimensional Numpy arrays.\n\
Any of vertex_normals, vertex_colors and vertex_tex_coords will be None if the respective\n\
model element was not present in the PLY file.\n\
\n\
If blender_face_indices is True (the default), the faces array uses the Blender vertices_raw\n\
convention of using *four indices per face*, regardless of whether the face is a triangle or quad.\n\
In case of a triangle the last index will be 0.\n\
\n\
If blender_face_indices is False, faces will be a 2-tuple of arrays, one with\n\
indices for triangles and one for quads.\n\
\n\
Note: Faces with more than 4 vertices are currently not supported.";

static PyMethodDef ModuleMethods[] =
{    
     {"readply", (PyCFunction)readply, METH_VARARGS|METH_KEYWORDS, readply_func_doc},
     {NULL, NULL, 0, NULL}
};

/* module initialization */

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC
initreadply(void)
{
    if (PyType_Ready(&_MyDeallocType) < 0)
        return;
    
    (void) Py_InitModule("readply", ModuleMethods);
    import_array();
}
#elif PY_MAJOR_VERSION == 3
static struct PyModuleDef module =
{
   PyModuleDef_HEAD_INIT,
   "readply",               /* name of module */
   NULL,                    /* module documentation, may be NULL */
   -1,                      /* size of per-interpreter state of the module,
                            or -1 if the module keeps state in global variables. */
   ModuleMethods
};

PyMODINIT_FUNC
PyInit_readply(void)
{
    PyObject *m;

    _MyDeallocType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&_MyDeallocType) < 0)
        return NULL;

    m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;

    import_array();

    return m;
}
#endif
