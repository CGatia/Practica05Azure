#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>
#include <cstdlib>   
#include <cstring>   

#define N 100000000 // Longitud de la secuencia

const std::string PATTERN = "ATGC"; 
const int P_LEN = PATTERN.length();

using namespace std;

// Genera una secuencia aleatoria grande
void generate_sequence(vector<char>& seq) {
    for (int i = 0; i < N - P_LEN; ++i) {
        seq[i] = "ATGC"[rand() % 4];
    }
    // Se coloca el patrón cerca del final para garantizar al menos una coincidencia
    for (int i = 0; i < P_LEN; ++i) {
        seq[N - 10000 + i] = PATTERN[i];
    }
}

// Verifica si existe el patrón en la posición i
inline bool match_pattern(const vector<char>& seq, int i) {
    for (int j = 0; j < P_LEN; ++j) {
        if (seq[i + j] != PATTERN[j]) {
            return false;
        }
    }
    return true;
}

// Ejecuta la búsqueda bajo distintos tipos de schedule
void run_search(int num_threads, const char* schedule_type, int chunk_size) {
    vector<char> dna_sequence(N);
    generate_sequence(dna_sequence);

    long long first_index = -1;
    omp_set_num_threads(num_threads);

    auto start = chrono::high_resolution_clock::now();

    // Se elige la política de scheduling
    if (strcmp(schedule_type, "static") == 0) {
        #pragma omp parallel for schedule(static, chunk_size)
        for (int i = 0; i < N - P_LEN; ++i) {
            if (first_index != -1) continue;

            bool match = match_pattern(dna_sequence, i);

            if (match) {
                #pragma omp critical
                {
                    if (first_index == -1 || i < first_index) {
                        first_index = i;
                    }
                }
            }
        }
    } else if (strcmp(schedule_type, "dynamic") == 0) {
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < N - P_LEN; ++i) {
            if (first_index != -1) continue;

            bool match = match_pattern(dna_sequence, i);

            if (match) {
                #pragma omp critical
                {
                    if (first_index == -1 || i < first_index) {
                        first_index = i;
                    }
                }
            }
        }
    } else if (strcmp(schedule_type, "guided") == 0) {
        #pragma omp parallel for schedule(guided, chunk_size)
        for (int i = 0; i < N - P_LEN; ++i) {
            if (first_index != -1) continue;

            bool match = match_pattern(dna_sequence, i);

            if (match) {
                #pragma omp critical
                {
                    if (first_index == -1 || i < first_index) {
                        first_index = i;
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(auto)
        for (int i = 0; i < N - P_LEN; ++i) {
            if (first_index != -1) continue;

            bool match = match_pattern(dna_sequence, i);

            if (match) {
                #pragma omp critical
                {
                    if (first_index == -1 || i < first_index) {
                        first_index = i;
                    }
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Hilos: " << num_threads
         << ", Schedule: " << schedule_type
         << " (" << chunk_size << ")"
         << ", Tiempo: " << elapsed.count() << " s"
         << ", Posición: " << first_index << endl;
}

int main() {
    cout << "--- Búsqueda de Patrón en ADN (" << PATTERN << ") ---" << endl;

    int num_threads = 4;
    int chunk_size  = 10000;

    run_search(num_threads, "static",  chunk_size);
    run_search(num_threads, "dynamic", chunk_size);
    run_search(num_threads, "guided",  chunk_size);
    run_search(num_threads, "auto",    chunk_size);

    return 0;
}
