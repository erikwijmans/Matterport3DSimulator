#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "Benchmark.hpp"
#include "MatterSim.hpp"
#include <json/json.h>
#include <tinyply.h>

mattersim::RGBHolder
average_color(const std::vector<mattersim::RGBHolder> &colors) {
    Eigen::Matrix3d rgb2ycrcb;
    rgb2ycrcb << 0.299, 0.587, 0.144, -0.169, -0.331, 0.5, 0.5, -0.419, -0.081;
    Eigen::Vector3d offset(0, 128, 128);
    Eigen::Vector3d accum = Eigen::Vector3d::Zero();
    for (auto &c : colors) {
        accum += offset + rgb2ycrcb * Eigen::Vector3d(c.r, c.g, c.b);
    }
    accum /= colors.size();
    // accum = rgb2ycrcb.inverse() * (accum - offset);
    accum.unaryExpr([](double d) { return std::min(255.0, std::max(d, 0.0)); });
    return mattersim::RGBHolder(accum[0], accum[1], accum[2]);
}
mattersim::RGBHolder
mode_color(const std::vector<mattersim::RGBHolder> &colors) {
    constexpr double bin_size = 5;
    auto rgb_count = std::map<
        Eigen::Vector3i, int,
        std::function<bool(const Eigen::Vector3i &, const Eigen::Vector3i &)>>{
        [](const Eigen::Vector3i &a, const Eigen::Vector3i &b) {
            return a[0] < b[0] ||
                   (a[0] == b[0] &&
                    (a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])));
        }};
    for (auto &c : colors) {
        auto key = Eigen::Vector3i(c.r, c.b, c.g);
        key.unaryExpr([](int i) { return static_cast<int>(i / bin_size); });
        auto it = rgb_count.find(key);
        if (it == rgb_count.end()) {
            rgb_count.emplace(key, 1);
        } else {
            it->second += 1;
        }
    }
    Eigen::Vector3i best_key;
    int best_c = 0;
    for (auto &p : rgb_count) {
        if (p.second > best_c) {
            best_key = p.first;
            best_c = p.second;
        }
    }

    best_key.unaryExpr([](int i) {
        return std::max(0, std::min(static_cast<int>(i * bin_size), 255));
    });
    return mattersim::RGBHolder(best_key[0], best_key[1], best_key[2]);
}

namespace mattersim {

// cube indices for index buffer object
GLushort cube_indices[] = {
    0, 1, 2, 3, 3, 2, 6, 7, 7, 6, 5, 4, 4, 5, 1, 0, 0, 3, 7, 4, 1, 2, 6, 5,
};

char *loadFile(const char *filename) {
    char *data;
    int len;
    std::ifstream ifs(filename, std::ifstream::in);

    ifs.seekg(0, std::ios::end);
    len = ifs.tellg();

    ifs.seekg(0, std::ios::beg);
    data = new char[len + 1];

    ifs.read(data, len);
    data[len] = 0;

    ifs.close();

    return data;
}

void setupCubeMap(GLuint &texture) {
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_CUBE_MAP);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

void setupCubeMap(GLuint &texture, cv::Mat &xpos, cv::Mat &xneg, cv::Mat &ypos,
                  cv::Mat &yneg, cv::Mat &zpos, cv::Mat &zneg) {
    setupCubeMap(texture);
    // use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_UNPACK_ALIGNMENT, (xneg.step & 3) ? 1 : 4);
    // set length of one complete row in data (doesn't need to equal image.cols)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, xneg.step / xneg.elemSize());
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB, xpos.rows,
                 xpos.cols, 0, GL_BGR, GL_UNSIGNED_BYTE, xpos.ptr());
    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB, xneg.rows,
                 xneg.cols, 0, GL_BGR, GL_UNSIGNED_BYTE, xneg.ptr());
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB, ypos.rows,
                 ypos.cols, 0, GL_BGR, GL_UNSIGNED_BYTE, ypos.ptr());
    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB, yneg.rows,
                 yneg.cols, 0, GL_BGR, GL_UNSIGNED_BYTE, yneg.ptr());
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB, zpos.rows,
                 zpos.cols, 0, GL_BGR, GL_UNSIGNED_BYTE, zpos.ptr());
    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB, zneg.rows,
                 zneg.cols, 0, GL_BGR, GL_UNSIGNED_BYTE, zneg.ptr());
}

Simulator::Simulator()
    : state{new SimState()}, width(320), height(240), vfov(0.8),
      minElevation(-0.94), maxElevation(0.94), navGraphPath("./connectivity"),
      datasetPath("./data"),
#ifdef OSMESA_RENDERING
      buffer(NULL),
#endif
      initialized(false), renderingEnabled(true), discretizeViews(false) {
    generator.seed(time(NULL));
};

Simulator::~Simulator() { close(); }

void Simulator::setCameraResolution(int width, int height) {
    this->width = width;
    this->height = height;
}

void Simulator::setCameraVFOV(double vfov) { this->vfov = vfov; }

void Simulator::setRenderingEnabled(bool value) {
    if (!initialized) {
        renderingEnabled = value;
    }
}

void Simulator::setDiscretizedViewingAngles(bool value) {
    if (!initialized) {
        discretizeViews = value;
    }
}

void Simulator::setDatasetPath(const std::string &path) { datasetPath = path; }

void Simulator::setNavGraphPath(const std::string &path) {
    navGraphPath = path;
}

void Simulator::init() {
    state->rgb.create(height, width, CV_8UC3);
    if (renderingEnabled) {
#ifdef OSMESA_RENDERING
        ctx = OSMesaCreateContext(OSMESA_RGBA, NULL);
        buffer = malloc(width * height * 4 * sizeof(GLubyte));
        if (!buffer) {
            throw std::runtime_error("MatterSim: Malloc image buffer failed");
        }
        if (!OSMesaMakeCurrent(ctx, buffer, GL_UNSIGNED_BYTE, width, height)) {
            throw std::runtime_error("MatterSim: OSMesaMakeCurrent failed");
        }
#else
        cv::namedWindow("renderwin", cv::WINDOW_OPENGL);
        cv::setOpenGlContext("renderwin");
        // initialize the extension wrangler
        glewInit();

        FramebufferName = 0;
        glGenFramebuffers(1, &FramebufferName);
        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

        // The texture we're going to render to
        GLuint renderedTexture;
        glGenTextures(1, &renderedTexture);

        // "Bind" the newly created texture : all future texture functions will
        // modify this texture
        glBindTexture(GL_TEXTURE_2D, renderedTexture);

        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, 0);

        // Poor filtering. Needed !
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // Set "renderedTexture" as our colour attachement #0
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, renderedTexture, 0);

        // Set the list of draw buffers.
        GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

        // Always check that our framebuffer is ok
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
            GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("MatterSim: GL_FRAMEBUFFER failure");
        }
#endif

        // set our viewport, clear color and depth, and enable depth testing
        glViewport(0, 0, this->width, this->height);
        glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
        glClearDepth(1.0f);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        // load our shaders and compile them.. create a program and link it
        glShaderV = glCreateShader(GL_VERTEX_SHADER);
        glShaderF = glCreateShader(GL_FRAGMENT_SHADER);
        const GLchar *vShaderSource = loadFile("src/lib/vertex.sh");
        const GLchar *fShaderSource = loadFile("src/lib/fragment.sh");
        glShaderSource(glShaderV, 1, &vShaderSource, NULL);
        glShaderSource(glShaderF, 1, &fShaderSource, NULL);
        delete[] vShaderSource;
        delete[] fShaderSource;
        glCompileShader(glShaderV);
        glCompileShader(glShaderF);
        glProgram = glCreateProgram();
        glAttachShader(glProgram, glShaderV);
        glAttachShader(glProgram, glShaderF);
        glLinkProgram(glProgram);
        glUseProgram(glProgram);

        // shader logs
        int vlength, flength;
        char vlog[2048], flog[2048];
        glGetShaderInfoLog(glShaderV, 2048, &vlength, vlog);
        glGetShaderInfoLog(glShaderF, 2048, &flength, flog);

        // grab the pvm matrix and vertex location from our shader program
        PVM = glGetUniformLocation(glProgram, "PVM");
        vertex = glGetAttribLocation(glProgram, "vertex");

        // these won't change
        Projection = glm::perspective((float)vfov, (float)width / (float)height,
                                      0.1f, 100.0f);
        Scale = glm::scale(glm::mat4(1.0f),
                           glm::vec3(10, 10, 10)); // Scale cube to 10m

        // cube vertices for vertex buffer object
        GLfloat cube_vertices[] = {
            -1.0, 1.0, 1.0,  -1.0, -1.0, 1.0,  1.0, -1.0, 1.0,  1.0, 1.0, 1.0,
            -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
        };
        glGenBuffers(1, &vbo_cube_vertices);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_vertices);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices,
                     GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertex);
        glVertexAttribPointer(vertex, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glGenBuffers(1, &ibo_cube_indices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_cube_indices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices),
                     cube_indices, GL_STATIC_DRAW);
    } else {
        // no rendering, e.g. for unit testing
        state->rgb.setTo(cv::Scalar(0, 0, 0));
    }
    initialized = true;
}

void Simulator::clearLocationGraph() {
    if (renderingEnabled) {
        for (auto loc : scanLocations[state->scanId]) {
            glDeleteTextures(1, &loc->cubemap_texture);
        }
    }
}

void Simulator::loadLocationGraph() {
    if (scanLocations.count(state->scanId) != 0) {
        return;
    }

    Json::Value root;
    auto navGraphFile =
        navGraphPath + "/" + state->scanId + "_connectivity.json";
    std::ifstream ifs(navGraphFile, std::ifstream::in);
    if (ifs.fail()) {
        throw std::invalid_argument(
            "MatterSim: Could not open navigation graph file: " + navGraphFile +
            ", is scan id valid?");
    }
    ifs >> root;
    for (auto viewpoint : root) {
        float posearr[16];
        int i = 0;
        for (auto f : viewpoint["pose"]) {
            posearr[i++] = f.asFloat();
        }
        // glm uses column-major order. Inputs are in row-major order.
        glm::mat4 mattPose = glm::transpose(glm::make_mat4(posearr));
        // glm access is col,row
        glm::vec3 pos{mattPose[3][0], mattPose[3][1], mattPose[3][2]};
        mattPose[3] = {0, 0, 0, 1}; // remove translation component
        // Matterport camera looks down z axis. Opengl camera looks down -z
        // axis. Rotate around x by 180 deg.
        glm::mat4 openglPose =
            glm::rotate(mattPose, (float)M_PI, glm::vec3(1.0f, 0.0f, 0.0f));
        std::vector<bool> unobstructed;
        for (auto u : viewpoint["unobstructed"]) {
            unobstructed.push_back(u.asBool());
        }
        auto viewpointId = viewpoint["image_id"].asString();
        GLuint cubemap_texture = 0;
        Location l{viewpoint["included"].asBool(),
                   viewpointId,
                   openglPose,
                   pos,
                   unobstructed,
                   cubemap_texture};
        scanLocations[state->scanId].push_back(std::make_shared<Location>(l));
    }
}

void Simulator::loadHouse(void) {
    const auto datafolder =
        datasetPath + "/v1/scans/" + state->scanId + "/house_segmentations/";
    const auto house_filename = datafolder + state->scanId + ".house";
    const auto ply_filename = datafolder + state->scanId + ".ply";

    std::ifstream ifs(house_filename, std::ios::in);
    if (ifs.fail()) {
        throw std::invalid_argument("MatterSim: Could not open house file: \"" +
                                    house_filename + "\"");
    }

    int nimages, npanoramas, nvertices, nsurfaces, nsegments, nobjects,
        ncategories, nregions, nportals, nlevels;

    std::string type, version;
    ifs >> type >> version;
    if (type != "ASCII") {
        throw std::invalid_argument("MatterSim: Could not read file \"" +
                                    house_filename + "\" of type \"" + type +
                                    "\"");
    }

    if (version == "1.0") {
        nsegments = 0;
        nobjects = 0;
        ncategories = 0;
        nportals = 0;
        std::string tmp;
        ifs >> tmp >> tmp >> tmp;
        ifs >> nimages >> npanoramas >> nvertices >> nsurfaces >> nregions >>
            nlevels;

        double d;
        for (int i = 0; i < 8 + 6; ++i)
            ifs >> d;
    } else {
        std::string tmp;
        ifs >> tmp >> tmp >> tmp;
        ifs >> nimages >> npanoramas >> nvertices >> nsurfaces >> nsegments >>
            nobjects >> ncategories >> nregions >> nportals >> nlevels;
        double d;
        for (int i = 0; i < 5 + 6 + 5; ++i)
            ifs >> d;
    }

    {
        std::string line;
        // Eat the hanging "\n"
        std::getline(ifs, line);
        // Trash the levels for now
        for (int i = 0; i < nlevels; ++i)
            std::getline(ifs, line);
    }

    this->regions.clear();
    for (int i = 0; i < nregions; ++i) {
        std::string tmp, cmd;
        ifs >> cmd;
        if (cmd != "R") {
            throw std::invalid_argument(
                "MatterSim: Tried to read a region, but got an \"" + cmd +
                "\"");
        }

        int region_idx, level_idx;
        ifs >> region_idx >> level_idx;

        // Trash the dummy values
        double d;
        ifs >> d >> d;

        std::string region_label;
        ifs >> region_label;

        Eigen::Vector3d pos, lo, hi;
        /* clang-format off */
        ifs >> pos[0] >> pos[1] >> pos[2]
            >> lo[0] >> lo[1] >> lo[2]
            >> hi[0] >> hi[1] >> hi[2];
        /* clang-format on */

        for (int j = 0; j < 4 + 1; ++j)
            ifs >> d;

        if (region_label != "Z" and region_label != "-") {
            this->regions.emplace(region_idx,
                                  std::make_shared<Region>(
                                      region_idx, level_idx, region_label, pos,
                                      BoundingBox::AxisAligned(lo, hi)));
        }
    }

    {
        std::string line;
        // Eat the hanging "\n"
        std::getline(ifs, line);
        // Trash the portals, surfaces, vertices
        for (int i = 0; i < nportals + nsurfaces + nvertices; ++i)
            std::getline(ifs, line);
    }

    for (int i = 0; i < npanoramas; ++i) {
        std::string cmd, tmp;
        double d;
        ifs >> cmd;
        if (cmd != "P") {
            throw std::invalid_argument(
                "MatterSim: Tried to read a panorama, but got an \"" + cmd +
                "\"");
        }

        std::string panorama_name;
        int region_idx;
        ifs >> panorama_name >> d >> region_idx;

        for (int j = 0; j < 9; ++j)
            ifs >> d;

        if (this->regions.find(region_idx) != this->regions.end()) {
            this->regions[region_idx]->viewpoints.emplace(panorama_name);
        }
    }

    {
        std::string line;
        // Eat the hanging "\n"
        std::getline(ifs, line);
        // Trash the images
        for (int i = 0; i < nimages; ++i)
            std::getline(ifs, line);
    }

    std::unordered_map<int, std::string> cat_idx_to_mpcat40;
    std::unordered_map<int, std::string> cat_idx_to_name;
    for (int i = 0; i < ncategories; ++i) {
        std::string tmp, cmd;

        ifs >> cmd;
        if (cmd != "C") {
            throw std::invalid_argument(
                "MatterSim: Tried to read a category, but got an \"" + cmd +
                "\"");
        }
        int cat_idx;
        ifs >> cat_idx;

        int mapping_idx, mpcat40_id;
        std::string label_name, mpcat40_name;
        // Only keep id's, not string
        ifs >> mapping_idx >> label_name;
        ifs >> mpcat40_id >> mpcat40_name;

        double d;
        for (int j = 0; j < 5; ++j)
            ifs >> d;

        cat_idx_to_mpcat40.emplace(cat_idx, mpcat40_name);
        cat_idx_to_name.emplace(cat_idx, label_name);
    }

    this->objects.clear();
    for (int i = 0; i < nobjects; ++i) {
        std::string tmp, cmd;
        ifs >> cmd;
        if (cmd != "O") {
            throw std::invalid_argument(
                "MatterSim: Tried to read a object, but got an \"" + cmd +
                "\"");
        }

        int object_idx, region_idx, category_idx;
        ifs >> object_idx >> region_idx >> category_idx;

        Eigen::Vector3d centroid, a0, a1, radii;
        /* clang-format off */
        ifs >> centroid[0] >> centroid[1] >> centroid[2]
            >> a0[0] >> a0[1] >> a0[2]
            >> a1[0] >> a1[1] >> a1[2]
            >> radii[0] >> radii[1] >> radii[2];
        /* clang-format on */

        double d;
        for (int j = 0; j < 8; ++j)
            ifs >> d;

        if (this->regions.find(region_idx) != this->regions.end()) {
            BoundingBox bbox(centroid, a0, a1, radii);

            ObjectPtr o = std::make_shared<Object>(
                object_idx, region_idx, cat_idx_to_name[category_idx],
                cat_idx_to_mpcat40[category_idx], centroid, bbox);

            this->objects.emplace(object_idx, o);
            this->regions.at(region_idx)->objects.emplace(object_idx, o);
        }
    }

#if 1
    std::unordered_map<int, std::vector<int>> obj_id_to_segments;
    for (int i = 0; i < nsegments; ++i) {
        std::string cmd, tmp;
        ifs >> cmd;
        if (cmd != "E") {
            throw std::invalid_argument(
                "MatterSim: Tried to read a segment, but got an \"" + cmd +
                "\"");
        }
        int segment_idx, object_idx;
        ifs >> segment_idx >> object_idx;
        obj_id_to_segments[object_idx].push_back(segment_idx);

        ifs >> tmp >> tmp;
        double d;
        for (int j = 0; j < 14; ++j)
            ifs >> d;
    }

    std::ifstream ply_in(ply_filename, std::ios::in | std::ios::binary);
    if (ply_in.fail()) {
        throw std::invalid_argument("MatterSim: Could not open ply file: \"" +
                                    ply_filename + "\"");
    }
    tinyply::PlyFile ply_file;
    ply_file.parse_header(ply_in);

    std::shared_ptr<tinyply::PlyData> ply_colors =
        ply_file.request_properties_from_element("vertex",
                                                 {"red", "green", "blue"});
    ply_file.read(ply_in);

    std::vector<RGBHolder> seg_colors(ply_colors->count);
    std::memcpy(seg_colors.data(), ply_colors->buffer.get(),
                ply_colors->buffer.size_bytes());

    std::unordered_map<int, std::vector<RGBHolder>> obj_id_to_segment_colors;
    for (const auto &p : obj_id_to_segments) {
        auto &k = p.first;
        auto &segs = p.second;

        auto it =
            obj_id_to_segment_colors.emplace(k, std::vector<RGBHolder>()).first;
        for (auto &s : segs) {
            it->second.emplace_back(seg_colors[s]);
        }
    }

    for (const auto &p : obj_id_to_segment_colors) {
        auto &k = p.first;
        auto &colors = p.second;
        this->objects[k]->color = mode_color(colors);
    }
#endif
}

const std::unordered_map<int, ObjectPtr> &Simulator::get_objects(void) {
    return this->objects;
}

const std::unordered_map<int, ObjectPtr> &
Simulator::get_objects(int region_idx) {
    if (this->regions.find(region_idx) == this->regions.end()) {
        throw std::invalid_argument("MatterSim: Region index is invalid");
    }
    return this->regions.at(region_idx)->objects;
}

const std::unordered_map<int, RegionPtr> &Simulator::get_regions(void) {
    return this->regions;
}

void Simulator::set_location_by_object(const ObjectPtr obj) {
    if (!initialized) {
        std::stringstream msg;
        msg << "MatterSim: Simulator is not initialized!";
        throw std::domain_error(msg.str());
    }
    auto &centroid = obj->centroid;
    auto &region = this->regions[obj->region_id];
    LocationPtr closest_location = nullptr;
    double best_dist = 1e38;
    constexpr double min_dist = 1.0;
    for (auto &location : scanLocations[state->scanId]) {
        if (location->included &&
            region->viewpoints.find(location->viewpointId) !=
                region->viewpoints.end()) {
            Eigen::Vector3d pos(location->pos[0], location->pos[1],
                                location->pos[2]);

            double dist = (pos - centroid).norm();
            double xy_dist = (Eigen::Vector2d(pos[0], pos[1]) -
                              Eigen::Vector2d(centroid[0], centroid[1]))
                                 .norm();
            if (dist < best_dist && xy_dist >= min_dist) {
                best_dist = dist;
                closest_location = location;
            }
        }
    }

    if (closest_location == nullptr) {
        throw std::runtime_error(
            "MatterSim: Could not find a suitable viewport for object " +
            obj->id);
    }

    Eigen::Vector3d gaze_vector =
        (centroid - Eigen::Vector3d(closest_location->pos[0],
                                    closest_location->pos[1],
                                    closest_location->pos[2]))
            .normalized();
    double elevation = std::atan2(
        gaze_vector[2], Eigen::Vector2d(gaze_vector[0], gaze_vector[1]).norm());
    double heading = std::atan2(gaze_vector[0], gaze_vector[1]);
    current_object = obj;

    this->newEpisode(state->scanId, closest_location->viewpointId, heading,
                     elevation);
}

void Simulator::populateNavigable() {
    std::vector<ViewpointPtr> updatedNavigable;
    updatedNavigable.push_back(state->location);
    unsigned int idx = state->location->ix;
    unsigned int i = 0;
    cv::Point3f curPos = state->location->point;
    double adjustedheading = M_PI / 2.0 - state->heading;
    glm::vec3 camera_horizon_dir(cos(adjustedheading), sin(adjustedheading),
                                 0.f);
    double cos_half_hfov = cos(vfov * width / height / 2.0);
    for (unsigned int i = 0; i < scanLocations[state->scanId].size(); ++i) {
        if (i == idx) {
            // Current location is pushed first
            continue;
        }
        if (scanLocations[state->scanId][idx]->unobstructed[i] &&
            scanLocations[state->scanId][i]->included) {
            // Check if visible between camera left and camera right
            glm::vec3 target_dir = scanLocations[state->scanId][i]->pos -
                                   scanLocations[state->scanId][idx]->pos;
            double rel_distance = glm::length(target_dir);
            double tar_z = target_dir.z;
            target_dir.z = 0.f; // project to xy plane
            double rel_elevation =
                atan2(tar_z, glm::length(target_dir)) - state->elevation;
            glm::vec3 normed_target_dir = glm::normalize(target_dir);
            double cos_angle = glm::dot(normed_target_dir, camera_horizon_dir);
            if (cos_angle >= cos_half_hfov) {
                glm::vec3 pos(scanLocations[state->scanId][i]->pos);
                double rel_heading =
                    atan2(target_dir.x * camera_horizon_dir.y -
                              target_dir.y * camera_horizon_dir.x,
                          target_dir.x * camera_horizon_dir.x +
                              target_dir.y * camera_horizon_dir.y);
                Viewpoint v{scanLocations[state->scanId][i]->viewpointId,
                            i,
                            cv::Point3f(pos[0], pos[1], pos[2]),
                            rel_heading,
                            rel_elevation,
                            rel_distance};
                updatedNavigable.push_back(std::make_shared<Viewpoint>(v));
            }
        }
    }
    std::sort(updatedNavigable.begin(), updatedNavigable.end(),
              ViewpointPtrComp());
    state->navigableLocations = updatedNavigable;
}

void Simulator::loadTexture(int locationId) {
    if (glIsTexture(
            scanLocations[state->scanId][locationId]->cubemap_texture)) {
        // Check if it's already loaded
        return;
    }
    cpuLoadTimer.Start();
    auto datafolder = datasetPath + "/v1/scans/" + state->scanId +
                      "/matterport_skybox_images/";
    auto viewpointId = scanLocations[state->scanId][locationId]->viewpointId;
    auto xpos = cv::imread(datafolder + viewpointId + "_skybox2_sami.jpg");
    auto xneg = cv::imread(datafolder + viewpointId + "_skybox4_sami.jpg");
    auto ypos = cv::imread(datafolder + viewpointId + "_skybox0_sami.jpg");
    auto yneg = cv::imread(datafolder + viewpointId + "_skybox5_sami.jpg");
    auto zpos = cv::imread(datafolder + viewpointId + "_skybox1_sami.jpg");
    auto zneg = cv::imread(datafolder + viewpointId + "_skybox3_sami.jpg");
    if (xpos.empty() || xneg.empty() || ypos.empty() || yneg.empty() ||
        zpos.empty() || zneg.empty()) {
        throw std::invalid_argument(
            "MatterSim: Could not open skybox files at: " + datafolder +
            viewpointId + "_skybox*_sami.jpg");
    }
    cpuLoadTimer.Stop();
    gpuLoadTimer.Start();
    setupCubeMap(scanLocations[state->scanId][locationId]->cubemap_texture,
                 xpos, xneg, ypos, yneg, zpos, zneg);
    gpuLoadTimer.Stop();
    if (!glIsTexture(
            scanLocations[state->scanId][locationId]->cubemap_texture)) {
        throw std::runtime_error("MatterSim: loadTexture failed");
    }
}

void Simulator::setHeadingElevation(double heading, double elevation) {
    // Normalize heading to range [0, 360]
    state->heading = fmod(heading, M_PI * 2.0);
    while (state->heading < 0.0) {
        state->heading += M_PI * 2.0;
    }
    if (discretizeViews) {
        // Snap heading to nearest discrete value
        double headingIncrement = M_PI * 2.0 / headingCount;
        int heading_step = std::lround(state->heading / headingIncrement);
        if (heading_step == headingCount)
            heading_step = 0;
        state->heading = (double)heading_step * headingIncrement;
        // Snap elevation to nearest discrete value (disregarding elevation
        // limits)
        state->elevation = elevation;
        if (state->elevation < -elevationIncrement / 2.0) {
            state->elevation = -elevationIncrement;
            state->viewIndex = heading_step;
        } else if (state->elevation > elevationIncrement / 2.0) {
            state->elevation = elevationIncrement;
            state->viewIndex = heading_step + 2 * headingCount;
        } else {
            state->elevation = 0.0;
            state->viewIndex = heading_step + headingCount;
        }
    } else {
        // Set elevation with limits
        state->elevation =
            std::max(std::min(elevation, maxElevation), minElevation);
    }
}

bool Simulator::setElevationLimits(double min, double max) {
    if (min < 0.0 && min > -M_PI / 2.0 && max > 0.0 && max < M_PI / 2.0) {
        minElevation = min;
        maxElevation = max;
        return true;
    } else {
        return false;
    }
}

void Simulator::newEpisode(const std::string &scanId,
                           const std::string &viewpointId, double heading,
                           double elevation) {
    totalTimer.Start();
    if (!initialized) {
        init();
    }
    state->step = 0;
    setHeadingElevation(heading, elevation);
    if (state->scanId != scanId) {
        // Moving to a new building...
        clearLocationGraph();
        state->scanId = scanId;
        loadLocationGraph();
        loadHouse();
    }
    int ix = -1;
    if (viewpointId.empty()) {
        // Generate a random starting viewpoint
        std::uniform_int_distribution<int> distribution(
            0, scanLocations[state->scanId].size() - 1);
        int start_ix =
            distribution(generator); // generates random starting index
        ix = start_ix;
        while (!scanLocations[state->scanId][ix]
                    ->included) { // Don't start at an excluded viewpoint
            ix++;
            if (ix >= scanLocations[state->scanId].size())
                ix = 0;
            if (ix == start_ix) {
                throw std::logic_error("MatterSim: ScanId: " + scanId +
                                       " has no included viewpoints!");
            }
        }
    } else {
        // Find index of selected viewpoint
        for (int i = 0; i < scanLocations[state->scanId].size(); ++i) {
            if (scanLocations[state->scanId][i]->viewpointId == viewpointId) {
                if (!scanLocations[state->scanId][i]->included) {
                    throw std::invalid_argument(
                        "MatterSim: ViewpointId: " + viewpointId +
                        ", is excluded from the connectivity graph.");
                }
                ix = i;
                break;
            }
        }
        if (ix < 0) {
            throw std::invalid_argument(
                "MatterSim: Could not find viewpointId: " + viewpointId +
                ", is viewpoint id valid?");
        }
    }
    glm::vec3 pos(scanLocations[state->scanId][ix]->pos);
    Viewpoint v{scanLocations[state->scanId][ix]->viewpointId,
                (unsigned int)ix,
                cv::Point3f(pos[0], pos[1], pos[2]),
                0.0,
                0.0,
                0.0};
    state->location = std::make_shared<Viewpoint>(v);
    populateNavigable();
    if (renderingEnabled) {
        loadTexture(state->location->ix);
        renderScene();
    }
    totalTimer.Stop();
}

SimStatePtr Simulator::getState() { return this->state; }

void Simulator::renderScene() {
    renderTimer.Start();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Scale and move the cubemap model into position
    Model = scanLocations[state->scanId][state->location->ix]->rot * Scale;
    // Opengl camera looking down -z axis. Rotate around x by 90deg (now
    // looking
    // down +y). Keep rotating for - elevation.
    RotateX = glm::rotate(glm::mat4(1.0f),
                          -(float)M_PI / 2.0f - (float)state->elevation,
                          glm::vec3(1.0f, 0.0f, 0.0f));
    // Rotate camera for heading, positive heading will turn right.
    View = glm::rotate(RotateX, (float)state->heading,
                       glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 M = Projection * View * Model;
    glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(M));
#ifndef OSMESA_RENDERING
    // Render to our framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
#endif
    glViewport(0, 0, width, height);
    glBindTexture(
        GL_TEXTURE_CUBE_MAP,
        scanLocations[state->scanId][state->location->ix]->cubemap_texture);
    glDrawElements(GL_QUADS, sizeof(cube_indices) / sizeof(GLushort),
                   GL_UNSIGNED_SHORT, 0);

    cv::Mat img(height, width, CV_8UC3);
    // use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
    // set length of one complete row in destination data (doesn't need to
    // equal
    // img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
    glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);
    cv::flip(img, img, 0);
    this->state->rgb = img;
    renderTimer.Stop();
}

void Simulator::makeAction(int index, double heading, double elevation) {
    totalTimer.Start();
    // move
    if (!initialized || index < 0 ||
        index >= state->navigableLocations.size()) {
        std::stringstream msg;
        msg << "MatterSim: Invalid action index: " << index;
        throw std::domain_error(msg.str());
    }
    state->location = state->navigableLocations[index];
    state->location->rel_heading = 0.0;
    state->location->rel_elevation = 0.0;
    state->location->rel_distance = 0.0;
    state->step += 1;
    if (discretizeViews) {
        // Increments based on sign of input
        if (heading > 0.0)
            heading = M_PI * 2.0 / headingCount;
        if (heading < 0.0)
            heading = -M_PI * 2.0 / headingCount;
        if (elevation > 0.0)
            elevation = elevationIncrement;
        if (elevation < 0.0)
            elevation = -elevationIncrement;
    }
    setHeadingElevation(state->heading + heading, state->elevation + elevation);
    populateNavigable();
    if (renderingEnabled) {
        // loading cubemap
        if (!glIsTexture(scanLocations[state->scanId][state->location->ix]
                             ->cubemap_texture)) {
            loadTexture(state->location->ix);
        }
        renderScene();
    }
    totalTimer.Stop();
    // std::cout << "\ntotalTimer: " << totalTimer.MilliSeconds() << " ms"
    // <<
    // std::endl;
    // std::cout << "cpuLoadTimer: " << cpuLoadTimer.MilliSeconds() << " ms"
    // <<
    // std::endl;
    // std::cout << "gpuLoadTimer: " << gpuLoadTimer.MilliSeconds() << " ms"
    // <<
    // std::endl;
    // std::cout << "renderTimer: " << renderTimer.MilliSeconds() << " ms"
    // <<
    // std::endl;
    // cpuLoadTimer.Reset();
    // gpuLoadTimer.Reset();
    // renderTimer.Reset();
    // totalTimer.Reset();
}

void Simulator::close() {
    if (initialized) {
        if (renderingEnabled) {
            // delete textures
            clearLocationGraph();
            // release vertex and index buffer object
            glDeleteBuffers(1, &ibo_cube_indices);
            glDeleteBuffers(1, &vbo_cube_vertices);
            // detach shaders from program and release
            glDetachShader(glProgram, glShaderF);
            glDetachShader(glProgram, glShaderV);
            glDeleteShader(glShaderF);
            glDeleteShader(glShaderV);
            glDeleteProgram(glProgram);
#ifdef OSMESA_RENDERING
            free(buffer);
            buffer = NULL;
            OSMesaDestroyContext(ctx);
#else
            cv::destroyAllWindows();
#endif
        }
        initialized = false;
    }
}

BoundingBox::BoundingBox(const Eigen::Vector3d &c, const Eigen::Vector3d &_a0,
                         const Eigen::Vector3d &_a1, const Eigen::Vector3d &_r)
    : centroid{c}, a0{_a0}, a1{_a1}, radii{_r} {

    // Normalize a0 and a1
    a0.normalize();
    a1.normalize();

    // Orthognalize a0 and a1
    a1 = a0.cross(a1).cross(a0).normalized();

    a2 = a0.cross(a1).normalized();

    if (radii[1] > radii[0]) {
        std::swap(a0, a1);
        std::swap(radii[0], radii[1]);
    }
    if (radii[2] > radii[0]) {
        std::swap(a0, a2);
        std::swap(radii[0], radii[2]);
    }
    if (radii[2] > radii[1]) {
        std::swap(a1, a2);
        std::swap(radii[1], radii[2]);
    }

    a2 = a0.cross(a1).normalized();
}

bool BoundingBox::is_in(const Eigen::Vector3d &pt) {
    const auto to_center = pt - this->centroid;
    return std::abs(to_center.dot(this->a0)) <= this->radii[0] &&
           std::abs(to_center.dot(this->a1)) <= this->radii[1] &&
           std::abs(to_center.dot(this->a2)) <= this->radii[2];
}

std::shared_ptr<std::vector<Eigen::Vector3d>> BoundingBox::corners() {
    auto corners = std::make_shared<std::vector<Eigen::Vector3d>>();
    corners->resize(8);

    corners->at(0) = -this->a0 * this->radii[0] - this->a1 * this->radii[1] -
                     this->a2 * this->radii[2];
    corners->at(1) = -this->a0 * this->radii[0] - this->a1 * this->radii[1] +
                     this->a2 * this->radii[2];
    corners->at(2) = -this->a0 * this->radii[0] + this->a1 * this->radii[1] +
                     this->a2 * this->radii[2];
    corners->at(3) = -this->a0 * this->radii[0] + this->a1 * this->radii[1] -
                     this->a2 * this->radii[2];

    corners->at(4) = this->a0 * this->radii[0] - this->a1 * this->radii[1] -
                     this->a2 * this->radii[2];
    corners->at(5) = this->a0 * this->radii[0] - this->a1 * this->radii[1] +
                     this->a2 * this->radii[2];
    corners->at(6) = this->a0 * this->radii[0] + this->a1 * this->radii[1] +
                     this->a2 * this->radii[2];
    corners->at(7) = this->a0 * this->radii[0] + this->a1 * this->radii[1] -
                     this->a2 * this->radii[2];

    for (auto &c : *corners) {
        c += this->centroid;
    }

    return corners;
}

BoundingBox BoundingBox::AxisAligned(const Eigen::Vector3d &lo,
                                     const Eigen::Vector3d &hi) {
    auto radii = (hi - lo) / 2;
    auto centroid = (hi + lo) / 2;
    return BoundingBox(centroid, Eigen::Vector3d::UnitX(),
                       Eigen::Vector3d::UnitY(), radii);
}
}
