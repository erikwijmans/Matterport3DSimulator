#include "ReadPly.hpp"

#include <sstream>

void ply::Vertex::load(std::ifstream &in) {
	in.read(reinterpret_cast<char *>(&this->x), sizeof(x));
	in.read(reinterpret_cast<char *>(&this->y), sizeof(y));
	in.read(reinterpret_cast<char *>(&this->z), sizeof(z));

	in.read(reinterpret_cast<char *>(&this->nx), sizeof(nx));
	in.read(reinterpret_cast<char *>(&this->ny), sizeof(ny));
	in.read(reinterpret_cast<char *>(&this->nz), sizeof(nz));

	in.read(reinterpret_cast<char *>(&this->tx), sizeof(tx));
	in.read(reinterpret_cast<char *>(&this->ty), sizeof(ty));

	in.read(reinterpret_cast<char *>(&this->r), sizeof(r));
	in.read(reinterpret_cast<char *>(&this->g), sizeof(g));
	in.read(reinterpret_cast<char *>(&this->b), sizeof(b));
}

void ply::Face::load(std::ifstream &in) {
	uint8_t len;
	in.read(reinterpret_cast<char *>(&len), sizeof(len));
	this->vert_inds.resize(len);
	for (auto &ind : this->vert_inds)
		in.read(reinterpret_cast<char *>(&ind), sizeof(ind));

	in.read(reinterpret_cast<char *>(&this->mat_id), sizeof(mat_id));
	in.read(reinterpret_cast<char *>(&this->obj_id), sizeof(obj_id));
	in.read(reinterpret_cast<char *>(&this->cat_id), sizeof(cat_id));
}

void ply::read_file(const std::string &filename,
		    std::vector<ply::Vertex::Ptr> &verts,
		    std::vector<ply::Face::Ptr> &faces) {

	std::ifstream in(filename, std::ios::binary | std::ios::in);
	if (in.fail()) {
		throw std::invalid_argument("MatterSim: Could not open \"" +
					    filename + "\"");
	}

	std::string line;
	while (std::getline(in, line)) {
		std::istringstream iss(line);
		std::string token;
		iss >> token;
		if (token == "ply" || token == "PLY" || token == "")
			continue;
		else if (token == "comment")
			continue;
		else if (token == "format")
			continue;
		else if (token == "element") {
			std::string type;
			int n;
			iss >> type >> n;
			if (type == "vertex")
				verts.resize(n);
			else if (type == "face")
				faces.resize(n);
			else
				throw std::invalid_argument(
				    "MatterSim: Unreconized type: \"" + type +
				    "\"");
		} else if (token == "property")
			continue;
		else if (token == "obj_info")
			continue;
		else if (token == "end_header")
			break;
		else
			throw std::invalid_argument(
			    "MatterSim: Unreconized token: \"" + token + "\"");
	}

	for (auto &v : verts) {
		v = std::make_shared<Vertex>();
		v->load(in);
	}

	for (auto &f : faces) {
		f = std::make_shared<Face>();
		f->load(in);
	}
}
