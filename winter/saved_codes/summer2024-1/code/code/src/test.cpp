// normal_distribution
#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>

int main()
{
  /*
  const int numNodes = 100;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);

  std::vector<double> force_vec;

  for (int i=0; i<numNodes; ++i) {
    double number = distribution(generator);
    force_vec.push_back(number);
    printf("%f\n",number);
  }

  double norm = sqrt(std::inner_product(force_vec.begin(), force_vec.begin()+101, force_vec.begin(), 0.0));

  printf("%f\n",norm);

  for (int i=0; i<force_vec.size(); ++i) {
    force_vec[i] = force_vec[i] / norm;
  }

  norm = sqrt(std::inner_product(force_vec.begin(), force_vec.begin()+101, force_vec.begin(), 0.0));
  printf("%f\n",norm);

  return 0;
  */
  std::string filename_force = "initial/forcing_vectors/forcing_vector_" + std::to_string(1) + ".txt";
  std::vector<double> buffer;
    std::string line;
    std::ifstream myFile(filename_force);
    while(getline(myFile, line))
    {
        std::istringstream lineStream(line);
        double first;
        lineStream >> first;
        buffer.push_back(first);
        printf("%f\n",first);
    }
    for(int i = 0; i < buffer.size(); i++){
      printf("%f\n",buffer[i]);
    }
}