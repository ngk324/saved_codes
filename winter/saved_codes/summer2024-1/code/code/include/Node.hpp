#ifndef _NODE_H_
#define _NODE_H_

#include <vector>
#include <memory>

class Node
{
private:
public:
    const int id;
    double z;     // State
    double z_old; // Previous state (for discrete time computations)
    double z_dot{0};
    double z_dot_old{0};

    Node(int id);
    Node(int id, double zz);

    double getDist(const Node &n) const;
    double isNear(const Node &n) const;

    void print(const std::string head) const;
};

#endif