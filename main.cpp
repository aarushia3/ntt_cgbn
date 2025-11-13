// main.cpp
#include "include/gpu_ntt.h"
#include "include/multiply.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " [merge|4step|benchmark]\n";
        return 1;
    }

    string option = argv[1];

    vector<uint32_t> A = {1, 2, 3, 4};
    vector<uint32_t> B = {5, 6, 7, 8};
    vector<uint32_t> C;

    if (option == "merge") {
        cout << "[Host] Multiplying small polynomials using merge method...\n";
        host_multiply_merge(A, B, C);
        cout << "Result (merge): ";
        for (auto x : C)
            cout << x << " ";
        cout << endl;

    } else if (option == "4step") {
        cout << "[Host] Multiplying small polynomials using 4-step method...\n";
        // host_multiply_4step(A, B, C);
        cout << "Result (4step): ";
        for (auto x : C)
            cout << x << " ";
        cout << endl;

    } else if (option == "benchmark") {
        cout << "[Benchmark] Comparing merge vs 4step performance...\n";

        vector<uint32_t> C_merge, C_4step;

        auto start_merge = chrono::high_resolution_clock::now();
        // host_multiply_merge(A, B, C_merge);
        auto end_merge = chrono::high_resolution_clock::now();
        double time_merge = chrono::duration<double, milli>(end_merge - start_merge).count();

        auto start_4step = chrono::high_resolution_clock::now();
        // host_multiply_4step(A, B, C_4step);
        auto end_4step = chrono::high_resolution_clock::now();
        double time_4step = chrono::duration<double, milli>(end_4step - start_4step).count();

        cout << "Merge time:  " << time_merge << " ms\n";
        cout << "4Step time:  " << time_4step << " ms\n";

        cout << "Result (merge): ";
        for (auto x : C_merge)
            cout << x << " ";
        cout << "\nResult (4step): ";
        for (auto x : C_4step)
            cout << x << " ";
        cout << endl;

        if (C_merge == C_4step)
            cout << "[OK] Results match!\n";
        else
            cout << "[WARNING] Results differ!\n";

    } else {
        cerr << "Unknown option: " << option << "\n";
        cerr << "Usage: " << argv[0] << " [merge|4step|benchmark]\n";
        return 1;
    }

    cout << "Done.\n";
    return 0;
}
