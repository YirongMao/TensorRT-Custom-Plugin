#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "NvUtils.h"
#include "flattenConcatCustom.h"

using namespace std;
using namespace nvinfer1;

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kINFO)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:  break;
        case Severity::kERROR: break;
        case Severity::kWARNING:  break;
        case Severity::kINFO:  break;
        default:break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

static Logger gLogger;

int main(int argc, char ** argv){
    std::cout<<"To read engine" << endl;
    std::ifstream ins; 
    ins.open(argv[1], std::ofstream::binary);
    ins.seekg(0, std::ios::end);
    int slength = ins.tellg();
    ins.seekg(0, std::ios::beg);
    char* sbuffer = new char[slength];
    ins.read(sbuffer, sizeof(char)*slength);
    ins.close();
    std::cout << "Read engine file " << argv[1] << ", size " << slength << endl;
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(sbuffer, slength, nullptr);
    std::cout<< "engine pointer " << engine << "batch size " << engine->getMaxBatchSize()<<endl;

    vector<vector<int>> inputs_shape;
    vector<vector<int>> outputs_shape;
    int nb_bindings = engine->getNbBindings();
    std::cout<<"nb bindings" << nb_bindings << endl;
    for (int i = 0; i < nb_bindings; i++) {
        std::cout<<"index binding " << i << endl;
        auto dims = engine->getBindingDimensions(i);
        if (engine->bindingIsInput(i)) {
            vector<int> shapes;
            for (int j = 0; j < dims.nbDims; ++j) {
                shapes.push_back(dims.d[j]);
                //std::cout<<dims.d[j]<<endl;
            }
            inputs_shape.push_back(shapes);
        } else {
            vector<int> shapes;
            for (int j = 0; j < dims.nbDims; ++j) {
                shapes.push_back(dims.d[j]);
            }
            outputs_shape.push_back(shapes);
        }
    }
    std::cout << "input shapes" << endl;
    for(auto vec: inputs_shape){
        for(auto v: vec){
            std::cout << v << " ";
        }
        std::cout << endl;
    }
    std::cout << "output shapes" << endl;
    for(auto vec: outputs_shape){
        for(auto v: vec){
            std::cout << v << " ";
        }
        std::cout << endl;
    }
    return 0;

}
