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
  
int n_factors=10;
float alpha = 0.01;
float my_lambda = 0.01;
int n_iters = 20;

unordered_map<long, double*> u_factors;
unordered_map<long, double*> i_factors;
vector<double*> prefs;
int f_cols = 4;

void read_file(){

    ifstream fin;
    
    
    fin.open ("../../datasets/ml-latest-small/ratings.csv");
    //fin.open ("../../datasets/ml-1m/ratings.csv");
    if (! fin.is_open()) {
        cerr << "error: cannot open file\n";
    }

    string line, val;
    int i = 0;
    int k = 0;
    while (getline (fin, line)) {
        if (i==0){
            i++;
            continue;
        }
        double *f = new double[f_cols];
        stringstream s (line);
        k = 0;
        while (getline (s, val, ',')){
            f[k] = (stod (val));
            k++;
        }
        prefs.push_back (f);
    }


    // i = 0;
    // for (double* row : prefs) {
    //     for (k=0; k<f_cols; k++)
    //         cout << row[k] << "  ";
    //     cout << "\n";
    //     i++;
    //     if (i > 10)
    //         break;
    // }

}

double dot_product(double *v1, double *v2, int size){
    double product = 0;
    for(int i=0;i<size;i++){
        product += v1[i] * v2[i];
    }
    return product;
}
/*
double calc_error(vector<vector<double> > X, unordered_map<long, vector<double> > u_factors, unordered_map<long, vector<double> > i_factors){
    long rows = X.size();
    long cols = X[0].size();
    long u_idx, i_idx;
    double error = 0;
    for (long i=0; i<rows; i++){
        u_idx = X[i][0];
        i_idx = X[i][1];

        error += abs(X[i][2] - inner_product(u_factors[u_idx].begin(), u_factors[u_idx].end(), i_factors[i_idx].begin(), 0));
    }
    return error/rows;

}
*/

pair<double,double> calc_error(long start, long end){
    long u_idx, i_idx;
    double abs_error = 0;
    double error = 0;
    double s_error = 0;

    for (long i=start; i<end; i++){
        u_idx = prefs[i][0];
        i_idx = prefs[i][1];

        error = prefs[i][2] - dot_product(u_factors[u_idx], i_factors[i_idx], n_factors);
        s_error += error*error;
        abs_error += abs(error);
    }
    return make_pair(abs_error/(end-start),s_error/(end-start));

}


void sgd(){
    double MIN = -0.5;
    double MAX = 0.5;
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(MIN, MAX);

    // initialize factor matrices
    for (long r=0; r<prefs.size(); r++){
        double *arr = new double[n_factors];
        for(int i=0; i<n_factors;i++){
            arr[i]=distr(eng);
        }
        u_factors[prefs[r][0]] = arr;

        arr = new double[n_factors];
        for(int i=0; i<n_factors;i++){
            arr[i]=distr(eng);
        }
        i_factors[prefs[r][1]] = arr;
    }

    std::shuffle(std::begin(prefs), std::end(prefs), eng);
    long end = (long) prefs.size()*0.9;

    // Stochastic Gradient descent

    pair<double,double> result = calc_error(0, end);
    cout << "Initial error: " << result.first << result.second << "\n";

    for (int t=0;t<n_iters;t++){
        std::shuffle(std::begin(prefs), std::begin(prefs)+end, eng);

        for (long r=0; r < end; r++){
            long u = prefs[r][0];
            long i = prefs[r][1];

            float error = prefs[r][2] - dot_product(u_factors[u], i_factors[i], n_factors);
            for (int k=0; k<n_factors; k++) {
                u_factors[u][k] = u_factors[u][k] + alpha * (error * i_factors[i][k] - my_lambda * u_factors[u][k]);
                i_factors[i][k] = i_factors[i][k] + alpha * (error * u_factors[u][k] - my_lambda * i_factors[i][k]);
            }
        }

        printf("Iteration %d\n", t);
        result = calc_error(0, end);
        cout << "Train MAE and MSE " << result.first << " " << result.second <<"\n";
        result = calc_error(end, prefs.size());
        cout << "Test MAE and MSE " << result.first << " " << result.second <<"\n";
    }
}

void write_factors(){

    std::ofstream userFile("user_factors.txt");
    std::ofstream movieFile("movie_factors.txt");
    int i;
    for (auto& pair : u_factors) {
        userFile << pair.first << ", ";
        for (i=0 ; i < n_factors; i++){
            userFile << pair.second[i];
            if (i < n_factors-1)
                userFile << ", ";
        }
        userFile << std::endl;
    }
    for (auto& pair : i_factors) {
        movieFile << pair.first << ", ";
        for (i=0 ; i < n_factors; i++){
            movieFile << pair.second[i];
            if (i < n_factors-1)
                movieFile << ", ";
        }
        movieFile << std::endl;
    }

    userFile.close();
    movieFile.close();

}

int main(){

    read_file();
    sgd();
    write_factors();


    return 0;


}
