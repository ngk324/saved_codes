
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <string>



 

int main(int argc, char *argv[]){

    //std::string filename = argv[1];
    std::string filename = "AAA-TOTAL-TwoNormResults.txt";

    std::string line;
    FILE *my_file = fopen(filename.c_str(), "r");
    
    fseek(my_file, 0, SEEK_END);
    int file_size = ftell(my_file);
    rewind(my_file);


    char c[50];
    std::vector<std::string> freqs;

    while(fgets(c,50,my_file) != NULL){
        char buf[10];
        std::string result;
        //std::cout<< "c: " << c << " line: " << string_line<<'\n';
        if(c[0] == 'F' && c[5] == 'f'){
            buf[0] = c[20];
            buf[1] = c[21];
            buf[2] = c[22];
            buf[3] = c[23];
            buf[4] = c[24];
            buf[5] = c[25];
            buf[6] = c[26];
            buf[7] = c[27];
            if(c[28] != '\n'){
                buf[8] = c[28];
            }
            result = std::string(buf);
            freqs.push_back(result);
        }
    }
    std::cout << "size: " << freqs.size() << std::endl;

    std::string fileName = "frequencies.txt";
    std::ofstream log;
    log.open(fileName, std::ofstream::app);
    for(int i = 0; i < freqs.size(); i++){
        if(i < freqs.size() - 1){
            //std::cout << freqs[i] << std::endl;
            log << freqs[i] << std::endl;
        }
        else{
            //std::cout << freqs[i];
            log << freqs[i];
        }
    }
    log.close();


    
   
    //std::cout << "Total # of values: " << file_size;// << twoNormsR.size() << "  AVG: " << avg  << std::endl;
    fclose(my_file);

    return 0;
}


