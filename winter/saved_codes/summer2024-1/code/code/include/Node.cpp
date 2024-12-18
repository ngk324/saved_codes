#include "Node.hpp"

#include <math.h>
#include <iostream>

Node::Node(int id) : id(id), z(0), z_old(0) {}
Node::Node(int id, double zz) : id(id), z(zz), z_old(zz) {}

