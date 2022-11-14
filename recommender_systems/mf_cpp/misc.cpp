#include <iostream>
#include <unordered_map>
#include <numeric>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>


using namespace std;

int main()
{

    vector<double*> v;
    double arr[3]={1,2,3};
    v.push_back(arr);
    v.push_back(arr);
    //vector<double> p = &v[3];

    printf("%f\n", v[0][1]);

}
