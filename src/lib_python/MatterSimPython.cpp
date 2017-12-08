#include "MatterSim.hpp"
#include <iostream>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mattersim {

class ViewPointPython {
public:
  ViewPointPython(ViewpointPtr locptr) {
    viewpointId = locptr->viewpointId;
    ix = locptr->ix;
    point.append(locptr->point.x);
    point.append(locptr->point.y);
    point.append(locptr->point.z);
    rel_heading = locptr->rel_heading;
    rel_elevation = locptr->rel_elevation;
    rel_distance = locptr->rel_distance;
  }
  std::string viewpointId;
  unsigned int ix;
  py::list point;
  double rel_heading;
  double rel_elevation;
  double rel_distance;
};

class SimStatePython {
public:
  SimStatePython(SimStatePtr state, bool renderingEnabled)
      : step{state->step}, viewIndex{state->viewIndex},
        location{state->location}, heading{state->heading},
        elevation{state->elevation} {
    if (renderingEnabled) {
      npy_intp colorShape[3]{state->rgb.rows, state->rgb.cols, 3};
      rgb = matToNumpyArray(3, colorShape, NPY_UBYTE, (void *)state->rgb.data);
    }
    scanId = state->scanId;
    for (auto viewpoint : state->navigableLocations) {
      navigableLocations.append(ViewPointPython{viewpoint});
    }
  }
  std::string scanId;
  unsigned int step;
  unsigned int viewIndex;
  py::object rgb;
  ViewPointPython location;
  double heading;
  double elevation;
  py::list navigableLocations;

private:
  py::object matToNumpyArray(int dims, npy_intp *shape, int type, void *data) {
    // colorDims, this->colorShape, NPY_UBYTE,
    // this->state->screenBuffer->data());
    PyObject *pyArray = PyArray_SimpleNewFromData(dims, shape, type, data);
    /* This line makes a copy: */
    PyObject *pyArrayCopied = PyArray_FROM_OTF(
        pyArray, type, NPY_ARRAY_ENSURECOPY | NPY_ARRAY_ENSUREARRAY);
    /* And this line gets rid of the old object which caused a memory leak:
     */
    Py_DECREF(pyArray);

    py::handle numpyArrayHandle = py::handle(pyArrayCopied);
    py::object numpyArray = py::reinterpret_steal<py::object>(numpyArrayHandle);

    return numpyArray;
  }
};

class RegionPython {
public:
  RegionPython(RegionPtr r)
      : id{r->id}, level{r->level}, type{r->type}, r_pos{r->r_pos},
        bbox{r->bbox} {
    for (auto &p : r->objects) {
      objects.append(p.second);
    }
  }

  int id;
  int level;
  std::string type;
  Eigen::Vector3d r_pos;
  BoundingBox bbox;
  py::list objects;
};

#if PY_MAJOR_VERSION >= 3
void *init_numpy() {
  import_array();
  return nullptr;
}
#else
void init_numpy() { import_array(); }
#endif
class SimulatorPython {
public:
  SimulatorPython() { init_numpy(); }
  void setDatasetPath(std::string path) { sim.setDatasetPath(path); }
  void setNavGraphPath(std::string path) { sim.setNavGraphPath(path); }
  void setCameraResolution(int width, int height) {
    sim.setCameraResolution(width, height);
  }
  void setCameraVFOV(double vfov) { sim.setCameraVFOV(vfov); }
  void setRenderingEnabled(bool value) { sim.setRenderingEnabled(value); }
  void setDiscretizedViewingAngles(bool value) {
    sim.setDiscretizedViewingAngles(value);
  }
  void init() { sim.init(); }
  void setSeed(int seed) { sim.setSeed(seed); }
  bool setElevationLimits(double min, double max) {
    return sim.setElevationLimits(min, max);
  }
  void newEpisode(const std::string &scanId,
                  const std::string &viewpointId = std::string(),
                  double heading = 0, double elevation = 0) {
    sim.newEpisode(scanId, viewpointId, heading, elevation);
  }
  SimStatePython *getState() {
    return new SimStatePython(sim.getState(), sim.renderingEnabled);
  }
  void makeAction(int index, double heading, double elevation) {
    sim.makeAction(index, heading, elevation);
  }
  const std::unordered_map<int, ObjectPtr> &get_objects(void) {
    return sim.get_objects();
  }
  const std::unordered_map<int, ObjectPtr> &get_objects_by_id(int region_id) {
    return sim.get_objects(region_id);
  }
  const std::unordered_map<int, RegionPtr> get_regions(void) {
    return sim.get_regions();
  }

  void set_location_by_object(ObjectPtr obj) {
    sim.set_location_by_object(obj);
  }

  void close() { sim.close(); }

private:
  Simulator sim;
};
}

using namespace mattersim;

PYBIND11_MODULE(MatterSim, m) {
  py::class_<ViewPointPython>(m, "ViewPoint")
      .def_readonly("viewpointId", &ViewPointPython::viewpointId)
      .def_readonly("ix", &ViewPointPython::ix)
      .def_readonly("point", &ViewPointPython::point)
      .def_readonly("rel_heading", &ViewPointPython::rel_heading)
      .def_readonly("rel_elevation", &ViewPointPython::rel_elevation)
      .def_readonly("rel_distance", &ViewPointPython::rel_distance);
  py::class_<SimStatePython>(m, "SimState")
      .def_readonly("scanId", &SimStatePython::scanId)
      .def_readonly("step", &SimStatePython::step)
      .def_readonly("rgb", &SimStatePython::rgb)
      .def_readonly("location", &SimStatePython::location)
      .def_readonly("heading", &SimStatePython::heading)
      .def_readonly("elevation", &SimStatePython::elevation)
      .def_readonly("viewIndex", &SimStatePython::viewIndex)
      .def_readonly("navigableLocations", &SimStatePython::navigableLocations);
  py::class_<BoundingBox>(m, "BoundingBox")
      .def_readonly("centroid", &BoundingBox::centroid)
      .def_readonly("a0", &BoundingBox::a0)
      .def_readonly("a1", &BoundingBox::a1)
      .def_readonly("a2", &BoundingBox::a2)
      .def_readonly("radii", &BoundingBox::radii)
      .def("is_in", &BoundingBox::is_in);
  py::class_<RGBHolder>(m, "RGBHolder")
      .def_readonly("r", &RGBHolder::r)
      .def_readonly("b", &RGBHolder::b)
      .def_readonly("g", &RGBHolder::g);
  py::class_<Object, ObjectPtr>(m, "SimObject")
      .def_readonly("id", &Object::id)
      .def_readonly("region_id", &Object::region_id)
      .def_readonly("coarse_class", &Object::coarse_class)
      .def_readonly("fine_class", &Object::fine_class)
      .def_readonly("color", &Object::color)
      .def_readonly("centroid", &Object::centroid)
      .def_readonly("bbox", &Object::bbox);
  py::class_<Region, RegionPtr>(m, "SimRegion")
      .def_readonly("id", &Region::id)
      .def_readonly("level", &Region::level)
      .def_readonly("type", &Region::type)
      .def_readonly("r_pos", &Region::r_pos)
      .def_readonly("bbox", &Region::bbox)
      .def_readonly("objects", &Region::objects)
      .def_readonly("viewpoints", &Region::viewpoints);
  py::class_<SimulatorPython>(m, "Simulator")
      .def(py::init<>())
      .def("setDatasetPath", &SimulatorPython::setDatasetPath)
      .def("setNavGraphPath", &SimulatorPython::setNavGraphPath)
      .def("setCameraResolution", &SimulatorPython::setCameraResolution)
      .def("setCameraVFOV", &SimulatorPython::setCameraVFOV)
      .def("setRenderingEnabled", &SimulatorPython::setRenderingEnabled)
      .def("setDiscretizedViewingAngles",
           &SimulatorPython::setDiscretizedViewingAngles)
      .def("init", &SimulatorPython::init)
      .def("setSeed", &SimulatorPython::setSeed)
      .def("setElevationLimits", &SimulatorPython::setElevationLimits)
      .def("newEpisode", &SimulatorPython::newEpisode)
      .def("getState", &SimulatorPython::getState,
           py::return_value_policy::take_ownership)
      .def("makeAction", &SimulatorPython::makeAction)
      .def("get_objects", &SimulatorPython::get_objects)
      .def("get_objects_by_id", &SimulatorPython::get_objects_by_id)
      .def("get_regions", &SimulatorPython::get_regions)
      .def("set_location_by_object", &SimulatorPython::set_location_by_object)
      .def("close", &SimulatorPython::close);
}
