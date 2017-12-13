#pragma once
#ifndef _READ_PLY_HPP
#define _READ_PLY_HPP

#include <fstream>
#include <string>
#include <vector>

namespace ply {
struct Vertex {
    float x, y, z, nx, ny, nz, tx, ty;
    uint8_t r, g, b;

    void load(std::ifstream &in);

    typedef std::shared_ptr<Vertex> Ptr;
};

struct Face {
    std::vector<int> vert_inds;
    int mat_id, obj_id, cat_id;

    void load(std::ifstream &in);

    typedef std::shared_ptr<Face> Ptr;
};

void read_file(const std::string &filename, std::vector<Vertex::Ptr> &verts,
               std::vector<Face::Ptr> &faces);
}

#endif
