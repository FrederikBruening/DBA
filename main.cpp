//
//  DBA
//
//  Created by Annonymous Author based on the work of Francois Petitjean ([1],https://github.com/fpetitjean/DBA.git).
//
//  [1] Francois Petitjean, Alain Ketterlin, and Pierre Gancarski. A global averaging method for dynamic time warping, with applications to clustering. Pattern Recognition, 44(3):678–693, 2011

//This program is free software: you can redistribute it and/or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation, version 3 of the License.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License
// * along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// Overview:
// The function "run" takes the strings "name" and "initialization" as input and specifies some addition experiment dependent parameters for the runs of the DBA algorithm. Here "name" specifies which data to take as an input and the initialization method "initialization" specifies how to find the initial center (the initialization method in the moment also specifies, if time series from "m5" or "UCR_TS_Archive_2015" are used). The function "run" also sets the seeds of the random number generators and specifies the maximum number of iterations as a criterium for stoping the algorithm early.
//
// Be sure to contain the data from "m5" or "UCR_TS_Archive_2015" in the corresponding directories and to create the necessary directories for the output text files.
//
// There are 4 different initialization methods:
// For the  first 3 methods "medoid", "random" and "rand+sample" the variable "name" specifies which data set from "UCR_TS_Archive_2015" to use.
// Each data set has multiple classes. In all the 3 initalization method the function "run" runs the DBA algorithm on each class seperately and then average over the resulting information regarding number of iterations and cost. The initalization methods differ in the following way:
//
// 1. "medoid":
// Starts with the input curve as a center that minimizes the cost among all input curves. No additional parameter are needed.
//
// 2. "random":
// Starts each run of the DBA algorithm with a random assignment (random walk on the DTW matrix, choosing diagonal, down, right with probability 1/3 each). The parameter "random_repetitions" specifies how many repetition of the DBA algorithm should be executed for each class. The average is taken over all repetitions of all clusters.

// 3. "rand+sample":
// Does the same as "random" but additionaly applies the following modifications: It chooses input curves as random samples from one given input curve (first curve of the class). In total "n_series" many time series of length "s_length" are sampled. Each from a random walk on the center curve "center" with pertubation N(0,"sigma"). This experiment is then repeated "length_multiplier" times, where in each repetition the length "s_length" of the sampled curves increases by a factor of 2.

// 4. "m5":
// For the "m5" initilization method, time series from the "m5" data set are used. Each saved time series represent the daily sale numbers of one item over all 10 stores. Only time series of items from the department named "name" are used. Since no other initialization method was wanted for the data set, we use here always a random assignment (like in initialization="random") to construct the starting center. The average is also computed "random_repetitions" many times. Similar to method "rand+sample", the experiment gets repeated for different length of input curves determined by "s_length" and "length_multiplier". In each run of a fixed input length, the algorithm draws a starting day uniformly at random. The time series then correspond to the sales at the "s_length" consecutive days starting from that day.

// 5. "m5_num_series":
// Same as "m5" with the only difference beeing, that the input sequences are chosen randomly and the number of selected sequences changes. The starting number of selected sequences is determined by "n_series". The experiment gets repeated for different number of input curves determined by "n_series" and "num_series_multiplier".

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include <numeric>
#include <set>
#include <map>

using namespace std;

//function for calculating the mean of an vector
double mean(vector<double>& vect){
    double sum = accumulate(vect.begin(), vect.end(), 0.0);
    return sum / vect.size();
}

//function for calculating the variance of an vector
double variance(vector<double>& vect){
    vector<double> diff(vect.size());
    double avg = mean(vect);
    transform(vect.begin(), vect.end(), diff.begin(), [avg](double x) { return x - avg; });
    double sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    return sqrt(sq_sum / vect.size());
}



//function for printing out all entries of a matrix
void print_mat(string name, const vector<vector<double>>& mat){
    cout << name << ": " << endl;
    for (const auto& i: mat){
        for (const auto& j: i){
            cout << j << " ";
        }
        cout << endl;
    }
}

//function for printing out all entries of a time series (double)
void print_time_series(string name, const vector<double>& mat){
    cout << name << ": " << endl;
    for (const auto& j: mat){
        cout << j << " ";
    }
    cout << endl;
}

//function for saving vector in file
void save_time_series(const vector<double> & vec, string file_name){
    fstream output;
    output.open("center_curves/" + file_name + ".txt", fstream::app | fstream::out);
    if (output.is_open()){
        for (const auto& j: vec){
            output << j << ", ";
        }
    }
    else {
        throw invalid_argument( "Output-file could not be opened.");
    }
}

//function for saving matrix in file
void save_matrix(const vector<vector<double>>& mat, string file_name){
    fstream output;
    output.open("selected_series/" + file_name + ".txt", fstream::app | fstream::out);
    if (output.is_open()){
        for (const auto & i: mat){
            for (const auto & j: i){
                output << j << ", ";
            }
            output << endl;
        }

    }
    else {
        throw invalid_argument( "Output-file could not be opened.");
    }
}


// function fills delta_mat with the squared distances between the respective center vertices and vertices of the time series
void fill_delta_mat(const vector<double> & center, const vector<double> &  time_series,vector<vector<double>> & delta_mat){
    int center_len = center.size();
    int s_len= time_series.size();
    
    for (int i = 0; i < center_len; i++){
        for (int j = 0; j < s_len; j++){
            delta_mat[i][j] = pow(center[i]-time_series[j],2);
        }
    }
}


// function calculates the sum of the squared DTW distances between the center and the input curves (series)
double sum_of_squares(const vector<double> & center, const vector<vector<double>> & series, vector<vector<double>> & cost_mat, vector<vector<double>> & delta_mat){
    double cost = 0;
    int center_len = center.size();
    int number_series = series.size();
    vector<double> n_elements(center_len,0);
    for (int s = 0; s < number_series; s++){
        int s_len = series[s].size();
        fill_delta_mat(center, series[s], delta_mat);
        cost_mat[0][0] = delta_mat[0][0];

        for (int i = 1; i < center_len; i++){
            cost_mat[i][0] = cost_mat[i-1][0]+delta_mat[i][0];
        }
        
        for (int j = 1; j < s_len; j++){
            cost_mat[0][j] = cost_mat[0][j-1]+delta_mat[0][j];
        }
        
        double res = 0;
        for (int i = 1; i < center_len; i++){
            for (int j = 1; j < s_len; j++){
                double diag = cost_mat[i-1][j-1];
                double left = cost_mat[i][j-1];
                double top = cost_mat[i-1][j];
                if(diag <= left){
                    if(diag <= top){
                        res = diag;
                    }
                    else {
                        res = top;
                    }
                }
                else{
                    if(left <= top){
                        res = left;
                    }
                    else {
                        res = top;
                    }
                }
                cost_mat[i][j]= res + delta_mat[i][j];
            }
        }
        
        int i = center_len-1;
        int j = s_len-1;
        cost += cost_mat[i][j];
    }
    return cost;
}


// function finds the index of the medoid (input curve that chosen as a center minimizes the cost among all curves in series)
int approximate_medoid_index(const vector<vector<double>> & series, vector<vector<double>> & cost_mat, vector<vector<double>> & delta_mat) {

    int medoid_index = 0;
    double best_cost = sum_of_squares(series[0], series, cost_mat, delta_mat);
    double cost = 0;
    int number_series = series.size();
    for (int s = 1; s < number_series; s++){
        cost = sum_of_squares(series[s], series, cost_mat, delta_mat);
        if (cost < best_cost){
            best_cost = cost;
            medoid_index = s;
        }
    }
    return medoid_index;
}


// function returns a center curve that results from a random assignment. Vertices of the center are the means of the assigned vertices.
void random_initiation(vector<double> & center_new, const vector<vector<double>> & series, vector<vector<double>> & path_mat ){
    vector<vector<int>> options_argmin = {{-1,-1},{0,-1},{-1,0}};
    int center_len = center_new.size();
    int number_series = series.size();
    vector<double> n_elements(center_len,0);
    
    for (int s = 0; s < number_series; s++){
        int s_len = series[s].size();
        path_mat[0][0] = -1;
        int i = 0;
        int j = 0;
        int rand_num;
        while (!((i == center_len-1) && (j == s_len-1))) {
            if (i == center_len-1) {
                rand_num = 1;
            }
            else if (j == s_len-1){
                rand_num = 2;
            }
            else {
                rand_num = rand() % 3;
            }
            vector<int> move = options_argmin[rand_num];
            i -= move[0];
            j -= move[1];
            path_mat[i][j]=rand_num;
        }
        
        while(path_mat[i][j] != -1){
            center_new[i] += series[s][j];
            n_elements[i] += 1;
            vector<int> move = options_argmin[path_mat[i][j]];
            i += move[0];
            j += move[1];
        }
        center_new[i] += series[s][j];
        n_elements[i] += 1;
    }

    for (int i = 0; i < center_len; i++){
        center_new[i] = center_new[i]/n_elements[i];
    }
}


//function does one step of the DBA algorithm. Based on old center (center) and the input curves (series), it finds the new center (center_new).
void dba_update(const vector<double> & center, vector<double> & center_new, const vector<vector<double>> & series, vector<vector<double>> & cost_mat, vector<vector<double>> & path_mat, vector<vector<double>> & delta_mat, double & cost){
    
    fill(center_new.begin(), center_new.end(), 0);
    cost = 0;
    
    vector<vector<int>> options_argmin = {{-1,-1},{0,-1},{-1,0}};
    int center_len = center.size();
    int number_series = series.size();
    vector<double> n_elements(center_len,0);
    for (int s = 0; s < number_series; s++){
        int s_len = series[s].size();
        fill_delta_mat(center, series[s], delta_mat);
        cost_mat[0][0] = delta_mat[0][0];
        path_mat[0][0] = -1;

        for (int i = 1; i < center_len; i++){
            cost_mat[i][0] = cost_mat[i-1][0]+delta_mat[i][0];
            path_mat[i][0] = 2;
        }
        
        for (int j = 1; j < s_len; j++){
            cost_mat[0][j] = cost_mat[0][j-1]+delta_mat[0][j];
            path_mat[0][j] = 1;
        }
        
        double res = 0;
        for (int i = 1; i < center_len; i++){
            for (int j = 1; j < s_len; j++){
                double diag = cost_mat[i-1][j-1];
                double left = cost_mat[i][j-1];
                double top = cost_mat[i-1][j];
                if(diag <= left){
                    if(diag <= top){
                        res = diag;
                        path_mat[i][j]=0;
                    }
                    else {
                        res = top;
                        path_mat[i][j]=2;
                    }
                }
                else{
                    if(left <= top){
                        res = left;
                        path_mat[i][j]=1;
                    }
                    else {
                        res = top;
                        path_mat[i][j]=2;
                    }
                }
                cost_mat[i][j]= res + delta_mat[i][j];
            }
        }
        
        int i = center_len-1;
        int j = s_len-1;
        cost += cost_mat[i][j];
        
        while(path_mat[i][j] != -1){
            center_new[i] += series[s][j];
            n_elements[i] += 1;
            vector<int> move = options_argmin[path_mat[i][j]];
            i += move[0];
            j += move[1];
        }
        center_new[i] += series[s][j];
        n_elements[i] += 1;
    }

    for (int i = 0; i < center_len; i++){
        center_new[i] = center_new[i]/n_elements[i];
    }
}


// dbaOutput combines all the relevent information output from the DBA algorithm: the final center curve, the number of iterations needed, the cost of the solution and the information, if the algorithm converged.
struct dbaOutput{
    vector<double> center;
    int iteration_counter=0;
    double cost;
    bool converged = false;
    vector<vector<double>> assignment;
};



// function runs the DBA algorithm on the input curves (series) for a maximum of n_iterations. The initialization of the first center is based on the string initialization. The output is a dbaOutput (see above)
dbaOutput performDBA(const vector<vector<double>> & series, int n_iterations, string initialization, int center_len){
    int max_len = series[0].size();
    int number_series = series.size();
    for (int s = 1; s < number_series; s++){
        int s_len=series[s].size();
        max_len = max(max_len,s_len);
    }
    
    vector<vector<double>> cost_mat(max_len,vector<double>(max_len,0));
    vector<vector<double>> delta_mat(max_len,vector<double>(max_len,0));
    vector<vector<double>> path_mat(max_len,vector<double>(max_len,0));
    if(center_len<=0){
        center_len = series[0].size();
    }
    vector<double> center(center_len,0);
    double cost;
    
    
    if(initialization == "medoid") {
        int medoid_ind = approximate_medoid_index(series, cost_mat, delta_mat);
        center = series[medoid_ind];
        cout << "Medoid found" << endl;
    }
    else {
        random_initiation(center, series, path_mat);
        cout << "Random starting center found" << endl;
    }
    
    vector<double> center_new(center.size(),0);
    
    int iteration_counter = 0;
    for (int i = 1; i <= n_iterations; i++){
        iteration_counter++;
        dba_update(center, center_new, series, cost_mat, path_mat, delta_mat, cost);
        
        
        if(center == center_new) {
            dbaOutput out;
            out.iteration_counter = iteration_counter;
            out.center = center;
            out.converged = true;
            out.cost = cost;
            return out;
        }
        else {
            center = center_new;
        }
    }
    dbaOutput out;
    out.iteration_counter = iteration_counter;
    out.center = center;
    out.cost = cost;
    return out;
}


//the function reads time series from the data set "name" out of the UCR_TS_Archive_2015 and saves them in the vector series. Folder UCR_TS_Archive_2015 is required to be in the Build/Products folder.
void read_data(string name, vector<vector<double>> & series, vector<int> & labels, string test_or_train){
    ifstream file ("UCR_TS_Archive_2015/" + name + "/" + name + "_" + test_or_train);
    
    if(file.is_open()){
        string line;
        string label;
        string coord;
        //int index = 0;
        while( getline(file,line) ){
            vector<double> s;
            stringstream ss(line);
            getline(ss,label,',');
            labels.push_back(stoi(label));
            
            while( getline(ss,coord,',') ){
                s.push_back(stod(coord));
            }
            //print_time_series("Series", s);
            series.push_back(s);
        }
        
        file.close();
    }
    else {
        throw invalid_argument( "File could not be opened.");
    }
}


//the function reads time series from the m5 data set and saves them in the vector series. Each saved time series represent the daily sale numbers of one item over all 10 stores. Only time series of items from the department named "name" will be saved. Folder m5 is required to be in the Build/Products folder
void read_data_m5(vector<vector<double>> & series, vector<string> & ids, set<string> & item_labels, vector<string> & item_ids, vector<string> & dept_ids, vector<string> & cat_ids, vector<string> & store_ids, vector<string> & state_ids, string name){
    
    //update file name
    ifstream file ("m5/m5-forecasting-accuracy/sales_train_evaluation.csv");
    
    
    // update here....
    if(file.is_open()){
        string line;
        string id1;
        string id2;
        string id3;
        string id4;
        string id5;
        string id6;
        
        string coord;
        //int index = 0;
        getline(file,line);
        
        while( getline(file,line) ){
            stringstream ss(line);
            
            getline(ss,id1,',');
            getline(ss,id2,',');
            getline(ss,id3,',');
            getline(ss,id4,',');
            getline(ss,id5,',');
            getline(ss,id6,',');
            vector<double> s;
            while( getline(ss,coord,',') ){
                s.push_back(stod(coord));
            }
            
            if (id3 == name || name == "all"){
                ids.push_back(id1);
                item_labels.insert(id2);
                item_ids.push_back(id2);
                dept_ids.push_back(id3);
                cat_ids.push_back(id4);
                store_ids.push_back(id5);
                state_ids.push_back(id6);
                //print_time_series("Series", s);
                series.push_back(s);
            }
        }
        
        file.close();
    }
    else {
        throw invalid_argument( "File could not be opened.");
    }
}


//function samples "n_series" many time series of length "s_length". Each from a random walk on the center curve "center" with pertubation N(0,"sigma") and with random number generated from "generator".
void sample_time_series(const vector<double> & center, vector<vector<double>> & series, int n_series, int s_length, double sigma, minstd_rand0 generator){
    normal_distribution<double> distribution(0.0,sigma);
    int center_len = center.size();
    double probability_move = center_len/s_length;
    if (probability_move > 1) {
        throw invalid_argument( "Required length of samples is smaller than the center curves, they should be created from.");
    }
    else {
        for (int j=0; j<n_series ; j++){
            vector<double> s_j;
            int index = 0;
            for (int i=0; i<s_length ; i++){
                double number = distribution(generator);
                s_j.push_back(center[index]+number);
                double rand_num  = rand()/ (RAND_MAX + 1.);
                if (index != center_len-1) {
                    if (center_len-index == s_length-i) {
                        index += 1;
                    }
                    else if (rand_num<= probability_move){
                        index += 1;
                    }
                }
            }
            series.push_back(s_j);
        }
    }
    
}


// function runs DBA to collect data about the number of iterations and the cost. To do so it uses the curves in "series" as input curves and behaves differently based on the given string "initilization". The different initilization methods are explained in comments above the run function. The information about the number of iterations are saved in txt files that are named based on "name" and "initialization".
void collect_data(string name, vector<vector<double>> & series, vector<int> & labels, int n_iterations, string initialization, int random_repetitions,  int n_series, int s_length, double sigma, minstd_rand0 generator, set<string> & item_labels, vector<string> & item_ids, int center_len){
    
    if(initialization == "m5" || initialization == "m5_num_series") {labels = {1,0};}
    
    int maxElement = *max_element(labels.begin(), labels.end());
    int minElement = *min_element(labels.begin(), labels.end());
    vector<double> cost_array;
    vector<double> iteration_array;
    
    if(initialization == "m5") {
        int max_length  = series[0].size();
        if(s_length > max_length){
            s_length = max_length;
        }
        for(int r = 0; r < random_repetitions; r++) {
            // For each item belonging to the dapartment given by name, we add all 10 time series of this item together
            int rand_num = 0;
            if (s_length < max_length ){
                rand_num = rand() % (max_length  - s_length);
            }
            cout<< endl << "Starting index: "<< rand_num << endl;
        
            map<string, vector<double>> sum_ts;
            vector<vector<double>> selected_series;
            for(string label : item_labels){
                vector<double> curve(s_length, 0);
                sum_ts[label]=curve;
            }
            for(int i = 0; i < item_ids.size(); i++){
                vector<double> curve(s_length, 0);
                string label = item_ids[i];
                transform(sum_ts[label].begin( ), sum_ts[label].end( ), series[i].begin()+ rand_num, sum_ts[label].begin( ),std::plus<double>( ));
            }
            for(string label : item_labels){
                selected_series.push_back(sum_ts[label]);
            }
            
            cout << "#selected Series: "<< selected_series.size()<<endl;
            save_matrix(selected_series, name + "_" + to_string(s_length) + "_" + to_string(center_len) + "_" + to_string(r+1) + "_series");
            
            dbaOutput out = performDBA(selected_series, n_iterations, initialization, center_len);
            save_time_series(out.center, name + "_" + to_string(s_length) + "_" + to_string(center_len) + "_" + to_string(r+1) + "_center");
            iteration_array.push_back(out.iteration_counter);
            cost_array.push_back(out.cost);
            cout << "Repetition: "<< r+1 << endl;
            cout << "Length: "<< s_length << endl;
            cout << "Center-Length: "<< center_len << endl;
            cout << "Iteration counter: " << out.iteration_counter << endl;
            cout << "Cost (Sum of Squared distances): " << out.cost << endl;
            cout << "Converged: " << out.converged << endl;
            
            fstream output_iter;
            output_iter.open("iterations/" + name + "_" + to_string(s_length) +  "_new_iteration_array.txt", fstream::app | fstream::out);
            if (output_iter.is_open()){
                output_iter << out.iteration_counter << ", ";
                output_iter.close();
            }
            else {
                throw invalid_argument( "Output-file could not be opened.");
            }
            
            fstream output_cost;
            output_cost.open("cost/" + name + "_" + to_string(s_length) + "_new_cost_array.txt", fstream::app | fstream::out);
            if (output_cost.is_open()){
                output_cost << out.cost << ", ";
                output_cost.close();
            }
            else {
                throw invalid_argument( "Output-file could not be opened.");
            }
            
        }
        cout << endl;
        cout << "Length: "<< s_length << endl;
        cout << endl;
        cout << "Iteration counter (mean): " << mean(iteration_array) << endl;
        cout << "Iteration counter (variance): " << variance(iteration_array) << endl;
        cout << "Cost (mean): " << mean(cost_array) << endl;
        cout << "Cost (variance): " << variance(cost_array) << endl;
    }
    
    if(initialization == "m5_num_series") {
        int max_length  = series[0].size();
        if(s_length > max_length){
            s_length = max_length;
        }
        cout << s_length<< endl;
        for(int r = 0; r < random_repetitions; r++) {
            // For each item belonging to the dapartment given by name, we add all 10 time series of this item together
            int rand_num = 0;
            if (s_length < max_length ){
                rand_num = rand() % (max_length  - s_length);
            }
            cout<< "Starting index: "<< rand_num << endl;
        
            map<string, vector<double>> sum_ts;
            vector<vector<double>> selected_series;
            for(string label : item_labels){
                vector<double> curve(s_length, 0);
                sum_ts[label]=curve;
            }
            for(int i = 0; i < item_ids.size(); i++){
                vector<double> curve(s_length, 0);
                string label = item_ids[i];
                transform(sum_ts[label].begin( ), sum_ts[label].end( ), series[i].begin()+ rand_num, sum_ts[label].begin( ),std::plus<double>( ));
            }
            for(string label : item_labels){
                selected_series.push_back(sum_ts[label]);
            }
            
            
            int total_n_series = selected_series.size();
            int rand_start_series = 0;
            if (total_n_series > n_series){
                rand_start_series = rand() % (total_n_series - n_series-1);
            }
            auto first = selected_series.begin()+rand_start_series;
            auto last = selected_series.end();
            if (total_n_series > n_series){
                last = selected_series.begin() + rand_start_series + n_series;
            }
            vector<vector<double>> selected_series_new(first, last);
            
            cout << endl <<  "Repetition: "<< r+1 << endl;
            cout << "#selected Series: "<< selected_series_new.size()<<endl;
            save_matrix(selected_series_new, name + "_" + to_string(n_series)+ "_" + to_string(s_length) + "_" + to_string(center_len) + "_" + to_string(r+1) + "num_series_series");
            
            dbaOutput out = performDBA(selected_series_new, n_iterations, initialization, center_len);
            save_time_series(out.center, name + "_"+ to_string(n_series) + "_" + to_string(s_length) + "_" + to_string(center_len) + "_" + to_string(r+1) + "num_series_center");
            iteration_array.push_back(out.iteration_counter);
            cost_array.push_back(out.cost);
            cout << endl;
            cout << "Repetition: "<< r+1 << endl;
            cout << "Length: "<< s_length << endl;
            cout << "Center-Length: "<< center_len << endl;
            cout << "Iteration counter: " << out.iteration_counter << endl;
            cout << "Cost (Sum of Squared distances): " << out.cost << endl;
            cout << "Converged: " << out.converged << endl;
            
            fstream output_iter;
            output_iter.open("iterations/" + name + "_" + to_string(n_series) + "_" + to_string(s_length) + "num_series_iteration_array.txt", fstream::app | fstream::out);
            if (output_iter.is_open()){
                output_iter << out.iteration_counter << ", ";
                output_iter.close();
            }
            else {
                throw invalid_argument( "Output-file could not be opened.");
            }
            
            fstream output_cost;
            output_cost.open("cost/" + name + "_" + to_string(s_length) + "_" + to_string(center_len) + "num_series_cost_array.txt", fstream::app | fstream::out);
            if (output_cost.is_open()){
                output_cost << out.cost << ", ";
                output_cost.close();
            }
            else {
                throw invalid_argument( "Output-file could not be opened.");
            }
            
        }
        cout << endl;
        cout << "Length: "<< s_length << endl;
        cout << endl;
        cout << "Iteration counter (mean): " << mean(iteration_array) << endl;
        cout << "Iteration counter (variance): " << variance(iteration_array) << endl;
        cout << "Cost (mean): " << mean(cost_array) << endl;
        cout << "Cost (variance): " << variance(cost_array) << endl;
    }
    if(initialization == "medoid") {
        for(int j = minElement; j <= maxElement; j++){
            vector<vector<double>> selected_series;
            for(int i = 0; i < labels.size(); i++){
                if(labels[i] == j){
                    selected_series.push_back(series[i]);
                }
            
            }
            if (selected_series.size() == 0){
                cout << endl << "No cluster of label " << j << endl << endl;
                continue;
                
            }
            dbaOutput out = performDBA(selected_series, n_iterations, initialization, center_len);
            //print_time_series("Center", out.center);
            iteration_array.push_back(out.iteration_counter);
            cost_array.push_back(out.cost);
            cout << endl;
            cout << "Iteration counter: " << out.iteration_counter << endl;
            cout << "Cost (Sum of Squared distances): " << out.cost << endl;
            cout << "Converged: " << out.converged << endl;
            // get output for DBA on cluster with label j
        }

        cout << "Iteration counter (mean): " << mean(iteration_array) << endl;
        cout << "Iteration counter (variance): " << variance(iteration_array) << endl;
        cout << "Cost (mean): " << mean(cost_array) << endl;
        cout << "Cost (variance): " << variance(cost_array) << endl;
    }
    
    
    if (initialization == "random") {
        for(int j = minElement; j <= maxElement; j++){
            vector<vector<double>> selected_series;
            for(int i = 0; i < labels.size(); i++){
                if(labels[i] == j){
                    selected_series.push_back(series[i]);
                }
            }
            if (selected_series.size() == 0){
                cout << endl << "No cluster of label " << j << endl << endl;
                continue;
            }
            for(int r = 0; r < random_repetitions; r++) {
                dbaOutput out = performDBA(selected_series, n_iterations, initialization, center_len);
                //print_time_series("Center", out.center);
                iteration_array.push_back(out.iteration_counter);
                cost_array.push_back(out.cost);
                cout << endl;
                cout << "Iteration counter: " << out.iteration_counter << endl;
                cout << "Cost (Sum of Squared distances): " << out.cost << endl;
                cout << "Converged: " << out.converged << endl;
                // get output for DBA on cluster with label j
            }
        }
        cout << endl;
        cout << "Iteration counter (mean): " << mean(iteration_array) << endl;
        cout << "Iteration counter (variance): " << variance(iteration_array) << endl;
        cout << "Cost (mean): " << mean(cost_array) << endl;
        cout << "Cost (variance): " << variance(cost_array) << endl;
        
    }
    
    
    if (initialization == "rand+sample") {
        for(int j = minElement; j <= maxElement; j++){
            vector<double> selected_curve;
            for(int i = 0; i < labels.size(); i++){
                if(labels[i] == j){
                    selected_curve = series[i];
                    break;
                }
            }
            if (selected_curve.size() == 0){
                cout << endl << "No cluster of label " << j << endl << endl;
                continue;
                
            }
            cout << endl;
            cout << "Length: "<< s_length << endl;
            cout << "Cluster: "<< j << endl;
            vector<vector<double>> sampled_series;
            auto first = selected_curve.begin();
            auto last = selected_curve.end();
            if (selected_curve.size()>s_length){
                last = selected_curve.begin() + s_length + 1;
            }
            vector<double> center(first, last);
            for(int r = 0; r < random_repetitions; r++) {
                sample_time_series(center, sampled_series, n_series, s_length, sigma, generator);
                dbaOutput out = performDBA(sampled_series, n_iterations, initialization, center_len);
                //print_time_series("Center", out.center);
                iteration_array.push_back(out.iteration_counter);
                cost_array.push_back(out.cost);
                cout << endl;
                cout << "Repetition: "<< r+1 << endl;
                cout << "Length: "<< s_length << endl;
                cout << "Center-Length: "<< center_len << endl;
                cout << "Iteration counter: " << out.iteration_counter << endl;
                cout << "Cost (Sum of Squared distances): " << out.cost << endl;
                cout << "Converged: " << out.converged << endl;
                // get output for DBA on cluster with label j
            }
        }
    }
    fstream output_file;
    output_file.open( name + "_" + initialization + "_" + to_string(center_len) +  ".txt", fstream::app | fstream::out);
    
    if (output_file.is_open())
      {
        output_file << endl;
        //output_file << endl << name << " (" << initialization << "):"<< endl;
        if (initialization == "rand+sample") {
            output_file << "Length: "<< s_length << endl;
            output_file << "number of series: "<< n_series << endl;
            output_file << "Standart dev (Permutation): "<< sigma << endl;
        }
        if (initialization == "m5") {
            output_file << "Length: "<< s_length << endl;
        }
        if (initialization == "m5_num_series") {
            output_file << "Length: "<< s_length << endl;
            output_file << "number of series: "<< n_series << endl;
        }
        if (initialization != "medoid") {
            output_file << "Repetitions per cluster: "<< random_repetitions << endl;
        }
        output_file << "Iteration counter (mean): " << mean(iteration_array) << endl;
        output_file << "Iteration counter (variance): " << variance(iteration_array) << endl;
        output_file << "Cost (mean): " << mean(cost_array) << endl;
        output_file << "Cost (variance): " << variance(cost_array) << endl;
        output_file.close();
      }
    else {
        throw invalid_argument( "Output-file could not be opened.");
    }
    if (initialization == "rand+sample" || initialization == "m5" ) {
        fstream output_iter;
        output_iter.open( name + "_" + initialization +  "_iteration_array.txt", fstream::app | fstream::out);
        if (output_iter.is_open()){
            output_iter << mean(iteration_array) << ", ";
        }
        else {
            throw invalid_argument( "Output-file could not be opened.");
        }
    
        fstream output_iter_var;
            output_iter_var.open( name + "_" + initialization + "_" + to_string(center_len) + "_iteration_var_array.txt", fstream::app | fstream::out);
        if (output_iter_var.is_open()){
            output_iter_var << variance(iteration_array) << ", " ;
        }
        else {
            throw invalid_argument( "Output-file could not be opened.");
        }
    }
    if (initialization == "m5_num_series" ) {
        fstream output_iter;
        output_iter.open( name + "_" + initialization + "_" + to_string(s_length) + "_iteration_array.txt", fstream::app | fstream::out);
        if (output_iter.is_open()){
            output_iter << mean(iteration_array) << ", ";
        }
        else {
            throw invalid_argument( "Output-file could not be opened.");
        }
    
        fstream output_iter_var;
            output_iter_var.open( name + "_" + initialization + "_" + to_string(s_length) + "_" + to_string(center_len) + "_iteration_var_array.txt", fstream::app | fstream::out);
        if (output_iter_var.is_open()){
            output_iter_var << variance(iteration_array) << ", " ;
        }
        else {
            throw invalid_argument( "Output-file could not be opened.");
        }
    }
    if (initialization == "random") {
        fstream output_iter;
        output_iter.open( initialization + "_iteration_array.txt", fstream::app | fstream::out);
        if (output_iter.is_open()){
            output_iter << mean(iteration_array) << ", ";
        }
        else {
            throw invalid_argument( "Output-file could not be opened.");
        }
    
        fstream output_iter_var;
            output_iter_var.open( initialization + "_" + to_string(center_len) + "_iteration_var_array.txt", fstream::app | fstream::out);
        if (output_iter_var.is_open()){
            output_iter_var << variance(iteration_array) << ", ";
        }
        else {
            throw invalid_argument( "Output-file could not be opened.");
        }
    }
}



// function is explained in the overview
void run(string name, string initialization){
    auto const seed = 2022;
    srand(seed);
    minstd_rand0 generator (seed);
    
    vector<vector<double>> series;      // initialization of the vector for the input curves
    vector<int> labels;                 // initialization of the vector for the labels of the input curves (stating to which cluster a curve belongs)
    int n_iterations = 3000;            // maximum number of iterations as a criterium for stoping the algorithm early
    
                                           
    
    
    int random_repetitions = 10;         // for "random" and "rand+sample" and "m5": repitions of random experiment. The average is taken in the end.
    
    
    int center_length = 0;             // for all: complexity of the center curve, if set to a value less or equal to 0, it will take the size of the complexity curves
    int center_multiplier = 0;          // for all: DBA gets called for center curves of length 2^i * center_length for all 0<= i <= center_multiplier
    
    
    int s_length = 100;                  // for "rand+sample" and "m5": complexity of sampled input curves
    int length_multiplier = 0;         // for "rand+sample" and "m5": DBA gets called for sample curves of length 2^i * s_length for all 0<= i <= length_multiplier
    
    int n_series = 25;                  // for "rand+sample" and "m5_num_series": how many sampled input curves
    int num_series_multiplier = 7;      // for "m5_num_series": DBA gets called for 2^i * n_series many sample curves for all 0<= i <= num_series_multiplier
    
    
    double sigma = 0.2;                 // just for "rand+sample": variance of pertubation of sampled input curves
    
    
    
    
    vector<string>  ids;
    set<string> item_labels;
    vector<string>  item_ids;
    vector<string>  dept_ids;
    vector<string>  cat_ids;
    vector<string>  store_ids;
    vector<string>  state_ids;

    
    

    if (initialization != "rand+sample" && initialization != "m5")  {length_multiplier = 0;}
    if (initialization != "m5_num_series")  {num_series_multiplier = 0;}
    
    if (initialization == "rand+sample" || initialization == "m5") {
        fstream output_iter;
        output_iter.open( name + "_" + initialization + "_" + to_string(center_length) +  "_length_array.txt", fstream::app | fstream::out);
        if (output_iter.is_open()){
            for (int i=1; i<(length_multiplier+1); i++) {
                output_iter << pow( 2, i) * s_length << ", ";
            }
            
        }
        else {
            throw invalid_argument( "Output-file could not be opened.");
        }
    }
    

    if (initialization=="m5" || initialization=="m5_num_series"){
        read_data_m5(series, ids, item_labels, item_ids, dept_ids, cat_ids, store_ids, state_ids, name);
        cout << "#Labels: "<< item_labels.size()<<endl;
        cout << "#Series: "<< series.size()<<endl;
    }
    else {
        read_data(name, series, labels, "TRAIN");
        read_data(name, series, labels, "TEST");
        
    }
    for (int i=0; i<(length_multiplier+1); i++) {
        for (int j = 0; j<(num_series_multiplier+1); j++) {
            for (int c = 0; c<(center_multiplier+1); c++) {
                cout<< endl;
                collect_data(name, series, labels, n_iterations, initialization, random_repetitions, pow( 2, j) * n_series, pow( 2, i) * s_length, sigma, generator, item_labels, item_ids, pow( 2, c) * center_length);
            }
        }
    }
}





int main(int argc, const char * argv[]) {
    vector<string> names = {"50words", "Adiac", "Beef", "CBF", "ChlorineConcentration", "Coffee", "ECG200", "ECG5000", "ElectricDevices", "FaceAll", "FaceFour", "FISH", "Gun_Point", "Lighting2", "Lighting7", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "SwedishLeaf", "synthetic_control", "Trace", "Two_Patterns", "wafer", "yoga"};              // names of the data set
    
                                            // choice of initialization method
    //string initialization = "medoid";     // starts with the input curve as a center that is the best center among all input curve
    
    //string initialization = "random";     // starting with a random assignment (random walk on the DTW matrix, choosing diagonal, down, right with probability 1/3 each)
    
    //string initialization = "rand+sample";  //  initialization as random + chooses input curves as random samples from one given input curve
    
    //string initialization = "m5";

    //for (const string & name: names){
        //cout << endl << "--------------" << name << "--------------" <<endl;
        //run(name,initialization);
    //}
    

    
    //cout << endl << "--------------" << "m5" << "--------------" <<endl;
    //run("FOODS_3", "m5");             // initialization that takes data from the m5 data set. Always uses a random assignment (like in initialization="random") to construct the starting center.
    
    cout << endl << "--------------" << "m5" << "--------------" <<endl;
    run("all", "m5_num_series");        // initialization that takes data from the m5 data set. Always uses a random assignment (like in initialization="random") to construct the starting center.
    
    return 0;
}




