#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <climits>

namespace fs = std::filesystem;

double bfs_serial(const int* row_ptr, const int* col_idx, int n, int src, int* cost);
double bfs_parallel(const int* row_ptr, const int* col_idx, int n, int src, int* cost, int nthreads);

struct CSR {
    int n = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;

    static inline std::string trim(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    }

    static bool load_mtx_to_csr(const std::string& path, CSR& out) {
        std::ifstream fin(path);
        if (!fin) {
            std::fprintf(stderr, "cannot open %s\n", path.c_str());
            return false;
        }

        std::string line;
        if (!std::getline(fin, line)) {
            std::fprintf(stderr, "empty file %s\n", path.c_str());
            return false;
        }

        std::string lower = trim(line);
        for (char& c : lower) c = (char)std::tolower((unsigned char)c);
        bool symmetric = (lower.find("symmetric") != std::string::npos);

        long long M = 0, N = 0, NZ = 0;
        while (std::getline(fin, line)) {
            std::string t = trim(line);
            if (t.empty() || t[0] == '%') continue;
            std::istringstream iss(t);
            if (!(iss >> M >> N >> NZ)) return false;
            break;
        }

        std::vector<std::pair<int, int>> edges;
        edges.reserve((size_t)NZ * (symmetric ? 2 : 1));

        long long read = 0;
        while (read < NZ && std::getline(fin, line)) {
            std::string t = trim(line);
            if (t.empty() || t[0] == '%') continue;
            std::istringstream iss(t);
            long long i, j;
            if (!(iss >> i >> j)) continue;
            edges.emplace_back((int)i, (int)j);
            if (symmetric && i != j) edges.emplace_back((int)j, (int)i);
            ++read;
        }

        int min_idx = INT_MAX;
        for (auto& e : edges) {
            min_idx = std::min({min_idx, e.first, e.second});
        }
        if (min_idx != 0) {
            for (auto& e : edges) {
                e.first--;
                e.second--;
            }
        }

        out.n = (int)std::max(M, N);
        std::vector<int> deg(out.n, 0);
        for (auto& e : edges) deg[e.first]++;

        out.row_ptr.assign(out.n + 1, 0);
        for (int i = 0; i < out.n; i++) out.row_ptr[i + 1] = out.row_ptr[i] + deg[i];

        out.col_idx.assign(out.row_ptr.back(), 0);
        std::vector<int> cur = out.row_ptr;
        for (auto& e : edges) out.col_idx[cur[e.first]++] = e.second;

        return true;
    }
};

static std::vector<fs::path> find_all_graphs_under_data() {
    std::vector<fs::path> files;
    fs::path data = fs::path("..") / "data";
    if (!fs::exists(data)) return files;

    for (auto& entry : fs::directory_iterator(data)) {
        if (!entry.is_directory()) continue;
        std::string base = entry.path().filename().string();
        fs::path mtx = entry.path() / (base + ".mtx");
        if (fs::exists(mtx)) files.push_back(mtx);
    }
    std::sort(files.begin(), files.end());
    return files;
}

static void append_csv_row(const fs::path& csv, const std::string& name,
                           int n, long long m, const std::vector<double>& ms) {
    bool exists = fs::exists(csv);
    std::ofstream out(csv, std::ios::app);

    if (!exists) {
        out << "graph,num_nodes,num_edges,avg_degree,"
               "ms_t1,ms_t2,ms_t4,ms_t8,ms_t16,ms_t32\n";
    }

    double avg = (double)m / n;
    out << name << "," << n << "," << m << "," << avg;
    for (double t : ms) out << "," << t;
    out << "\n";
}

int main(int argc, char** argv) {
    bool do_c = false, do_p = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-c")) do_c = true;
        if (!strcmp(argv[i], "-p")) do_p = true;
    }

    if (do_c) {
        fs::path mtx = fs::path("..") / "data" / "roadNet-CA" / "roadNet-CA.mtx";
        CSR G;
        if (!CSR::load_mtx_to_csr(mtx.string(), G)) return 1;

        std::vector<int> ref(G.n, -1);
        double t_ref = bfs_serial(G.row_ptr.data(), G.col_idx.data(), G.n, 0, ref.data());

        std::printf("[-c] serial: %.3f ms\n", t_ref);

        for (int nt : {1, 2, 4, 8, 16}) {
            std::vector<int> out(G.n, -1);
            double t = bfs_parallel(G.row_ptr.data(), G.col_idx.data(), G.n, 0, out.data(), nt);
            bool ok = (out == ref);
            std::printf("[-c] omp %d threads: %.3f ms %s\n", nt, t, ok ? "OK" : "BAD");
        }
    }

    if (do_p) {
        fs::path csv = fs::path("..") / "output" / "bfs_profile.csv";
        fs::create_directories(csv.parent_path());

        for (auto& mtx : find_all_graphs_under_data()) {
            CSR G;
            if (!CSR::load_mtx_to_csr(mtx.string(), G)) continue;

            std::vector<double> times;
            for (int nt : {1, 2, 4, 8, 16, 32}) {
                std::vector<int> cost(G.n, -1);
                times.push_back(
                    bfs_parallel(G.row_ptr.data(), G.col_idx.data(), G.n, 0, cost.data(), nt)
                );
            }
            append_csv_row(csv, mtx.parent_path().filename().string(),
                           G.n, (long long)G.col_idx.size(), times);
        }
    }

    return 0;
}
