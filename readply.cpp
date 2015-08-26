/* 
TODO:
- need to handle faces with more than 4 vertices
*/

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <rply.h>
#include <cstdio>

static float    *vertices = NULL;
static int      next_vertex_element_offset = 0;

static int      *faces = NULL;                      
static int      next_face_element_offset = 0;

static int 
vertex_cb(p_ply_argument argument) 
{
    vertices[next_vertex_element_offset] = ply_get_argument_value(argument);
    next_vertex_element_offset++;

    return 1;
}

static int 
face_cb(p_ply_argument argument) 
{
    long    length, value_index;
    int     vertex_index;
    
    ply_get_argument_property(argument, NULL, &length, &value_index);
    
    if (value_index == -1)
    {
        if (length > 4)
            printf("Warning: ignoring face with %ld vertices!\n", length);
        return 1;
    }
    
    vertex_index = ply_get_argument_value(argument);    
    faces[next_face_element_offset] = vertex_index;
    next_face_element_offset++;
    
    // Blender's vertices_raw array uses 4 vertex indices per face,
    // for denoting either a triangle or a quad. 
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

static PyObject* 
readply(PyObject* self, PyObject* args)
{
    const char *fname;  
    
    if (!PyArg_ParseTuple(args, "s", &fname))
        return NULL;
        
    p_ply ply = ply_open(fname, NULL, 0, NULL);
    if (!ply)     
    {
        PyErr_SetString(PyExc_IOError, "Could not open PLY file");
        return NULL;        
    }
    
    if (!ply_read_header(ply)) 
    {
        PyErr_SetString(PyExc_IOError, "Could not read PLY header");
        ply_close(ply);
        return NULL;            
    }    
    
    // Set callbacks
    // normals: nx, ny, nz
    // texcoords: s, t
    // vcolors: ???
    
    long nvertices, nfaces;
    
    nvertices = ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 1);
    
    nfaces = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);
    
    // Allocate memory, initialize    
    
    vertices = (float*) malloc(sizeof(float)*nvertices*3);
    next_vertex_element_offset = 0;    
        
    faces = (int*) malloc(sizeof(int)*nfaces*4);
    next_face_element_offset = 0;
    
    printf("%ld vertices\n%ld faces\n", nvertices, nfaces);
    
    // Let rply process the file using the callbacks
        
    if (!ply_read(ply)) 
    {
        PyErr_SetString(PyExc_IOError, "Could not read PLY data");
        ply_close(ply);
        free(vertices);
        free(faces);
        return NULL;            
    }
    
    // Clean up
    
    ply_close(ply);
    
    // Return stuff
    
    npy_intp    np_vertices_dims[1] = { nvertices*3 };
    npy_intp    np_faces_dims[1] = { nfaces*4 };
    
    // http://stackoverflow.com/questions/27912483/memory-leak-in-python-extension-when-array-is-created-with-pyarray-simplenewfrom
    PyArrayObject *np_vertices = (PyArrayObject*) PyArray_SimpleNewFromData(1, np_vertices_dims, NPY_FLOAT, vertices);  
    PyArrayObject *np_faces = (PyArrayObject*) PyArray_SimpleNewFromData(1, np_faces_dims, NPY_INT, faces);  
    
    PyArray_ENABLEFLAGS(np_vertices, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(np_faces, NPY_ARRAY_OWNDATA);
    
    return Py_BuildValue("(iiOO)",  nvertices, nfaces, np_vertices, np_faces);
}

static PyMethodDef ModuleMethods[] =
{
     {"readply", readply, METH_VARARGS, ""},
     {NULL, NULL, 0, NULL}
};

/* module initialization */

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC 
initmodule(void)
{
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

    m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;
        
    import_array();
    
    return m;
}
#endif
