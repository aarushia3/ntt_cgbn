// bench_gmp_multiply.cpp
//
// Benchmark GMP's mpz_mul across a range of operand sizes and dump timing
// (import/multiply/export breakdown) to CSV. Standalone — only depends on
// GMP and the standard library
//
// Build:  g++ -O2 -std=c++17 bench_gmp_multiply.cpp -lgmp -o build/bench_gmp_multiply
//         g++ -O2 -std=c++17 -DNATIVE_HOST_LIMBS bench_gmp_multiply.cpp -lgmp -o build/bench_gmp_multiply_64
//
// Run:    ./build/bench_gmp_multiply [--warmup N] [--iters N] [--csv FILE] [--append] [L ...]
//
// L spec: values < 64 mean 1<<L limbs; values >= 64 are literal limb counts.
//         ranges are inclusive, e.g. 16-24 -> 16,17,...,24.
//         Limb width is 32 bits unless NATIVE_HOST_LIMBS is defined at
//         compile time, in which case limbs are 64 bits (uint64_t)

#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

#if defined(NATIVE_HOST_LIMBS)
using LimbType = uint64_t;
static constexpr unsigned LIMB_BITS_VAL = 64;
#else
using LimbType = uint32_t;
static constexpr unsigned LIMB_BITS_VAL = 32;
#endif

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

static constexpr int DEFAULT_WARMUP = 2;
static constexpr int DEFAULT_ITERS  = 20;

static const vector<size_t> DEFAULT_L_VALUES = {
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22,
};

// ---------------------------------------------------------------------------
// Random limb generation (mirrors test_full_pipeline.cpp conventions)
// ---------------------------------------------------------------------------

static vector<LimbType> random_limbs(size_t n, uint64_t seed)
{
    mt19937_64 rng(seed);
    vector<LimbType> v(n);

    for (size_t i = 0; i < n; i++) {
        if (i % 8 == 0) v[i] = 0;                  // edge case: zero limb
        else if (i % 8 == 1) v[i] = 1;              // edge case: one limb
#if defined(NATIVE_HOST_LIMBS)
        else if (i % 8 == 2) v[i] = UINT64_MAX;     // edge case: max limb
        else v[i] = rng();
#else
        else v[i] = (LimbType)(rng() % (1ULL << 30));
#endif
    }
    return v;
}

static size_t resolve_limb_count(size_t L_arg)
{
    if (L_arg < 64)
        return size_t(1) << L_arg;
    return L_arg;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

struct TimingStats {
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

static TimingStats compute_stats(const vector<double>& samples)
{
    TimingStats s{};
    if (samples.empty())
        return s;

    double sum = 0.0, sum_sq = 0.0;
    s.min_ms = samples[0];
    s.max_ms = samples[0];

    for (double t : samples) {
        sum += t;
        sum_sq += t * t;
        s.min_ms = min(s.min_ms, t);
        s.max_ms = max(s.max_ms, t);
    }

    const double n = static_cast<double>(samples.size());
    s.mean_ms = sum / n;
    const double var = (sum_sq / n) - (s.mean_ms * s.mean_ms);
    s.stddev_ms = sqrt(max(0.0, var));
    return s;
}

// ---------------------------------------------------------------------------
// GMP multiply with stage timing
// ---------------------------------------------------------------------------

struct GmpTiming {
    double import_a_ms = 0.0;
    double import_b_ms = 0.0;
    double multiply_ms = 0.0;
    double export_ms = 0.0;
    double total_ms = 0.0;
};

static GmpTiming gmp_mul_timed(
    const vector<LimbType>& A,
    const vector<LimbType>& B,
    vector<LimbType>& out)
{
    using clock = chrono::high_resolution_clock;

    mpz_t a, b, c;
    mpz_init(a);
    mpz_init(b);
    mpz_init(c);

    auto t0 = clock::now();
    mpz_import(a, A.size(), -1, sizeof(LimbType), 0, 0, A.data());
    auto t1 = clock::now();

    mpz_import(b, B.size(), -1, sizeof(LimbType), 0, 0, B.data());
    auto t2 = clock::now();

    mpz_mul(c, a, b);
    auto t3 = clock::now();

    const size_t expected_limbs = A.size() + B.size();
    out.assign(expected_limbs, 0);
    size_t count = 0;
    mpz_export(out.data(), &count, -1, sizeof(LimbType), 0, 0, c);
    out.resize(expected_limbs, 0);
    auto t4 = clock::now();

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);

    GmpTiming timing{};
    timing.import_a_ms = chrono::duration<double, milli>(t1 - t0).count();
    timing.import_b_ms = chrono::duration<double, milli>(t2 - t1).count();
    timing.multiply_ms = chrono::duration<double, milli>(t3 - t2).count();
    timing.export_ms   = chrono::duration<double, milli>(t4 - t3).count();
    timing.total_ms    = chrono::duration<double, milli>(t4 - t0).count();
    return timing;
}

// ---------------------------------------------------------------------------
// Bench row
// ---------------------------------------------------------------------------

struct BenchRow {
    size_t L_arg = 0;
    size_t L = 0;
    int warmup = 0;
    int iters = 0;

    TimingStats total;
    TimingStats import_a;
    TimingStats import_b;
    TimingStats multiply;
    TimingStats export_stage;
};

static BenchRow benchmark_L(size_t L_arg, int warmup, int iters, uint64_t seed)
{
    const size_t L = resolve_limb_count(L_arg);

    vector<LimbType> A = random_limbs(L, seed);
    vector<LimbType> B = random_limbs(L, seed + 1);
    vector<LimbType> C;

    for (int i = 0; i < warmup; i++)
        gmp_mul_timed(A, B, C);

    vector<double> total_s, import_a_s, import_b_s, mul_s, export_s;
    total_s.reserve(static_cast<size_t>(iters));
    import_a_s.reserve(static_cast<size_t>(iters));
    import_b_s.reserve(static_cast<size_t>(iters));
    mul_s.reserve(static_cast<size_t>(iters));
    export_s.reserve(static_cast<size_t>(iters));

    for (int i = 0; i < iters; i++) {
        GmpTiming t = gmp_mul_timed(A, B, C);
        total_s.push_back(t.total_ms);
        import_a_s.push_back(t.import_a_ms);
        import_b_s.push_back(t.import_b_ms);
        mul_s.push_back(t.multiply_ms);
        export_s.push_back(t.export_ms);
    }

    BenchRow row{};
    row.L_arg = L_arg;
    row.L = L;
    row.warmup = warmup;
    row.iters = iters;
    row.total = compute_stats(total_s);
    row.import_a = compute_stats(import_a_s);
    row.import_b = compute_stats(import_b_s);
    row.multiply = compute_stats(mul_s);
    row.export_stage = compute_stats(export_s);
    return row;
}

// ---------------------------------------------------------------------------
// CLI parsing (mirrors bench/gpu_full_multiply_benchmark.cpp)
// ---------------------------------------------------------------------------

static bool file_nonempty(const string& path)
{
    ifstream f(path);
    return f.good() && f.peek() != ifstream::traits_type::eof();
}

static bool parse_size_t(const string& s, size_t& out)
{
    char* end = nullptr;
    unsigned long long v = strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v == 0)
        return false;
    out = static_cast<size_t>(v);
    return true;
}

static bool append_l_spec(const string& spec, vector<size_t>& out)
{
    const auto dash = spec.find('-');
    if (dash == string::npos) {
        size_t L_arg = 0;
        if (!parse_size_t(spec, L_arg))
            return false;
        out.push_back(L_arg);
        return true;
    }

    if (dash == 0 || dash == spec.size() - 1)
        return false;

    size_t lo = 0, hi = 0;
    if (!parse_size_t(spec.substr(0, dash), lo) ||
        !parse_size_t(spec.substr(dash + 1), hi))
        return false;

    if (lo > hi)
        return false;

    for (size_t v = lo; v <= hi; ++v)
        out.push_back(v);
    return true;
}

static void usage(const char* prog)
{
    cerr << "Usage: " << prog
         << " [--warmup N] [--iters N] [--csv FILE] [--append] [L ...]\n"
         << "\n"
         << "  --warmup N   warmup iterations per L (default "
         << DEFAULT_WARMUP << ")\n"
         << "  --iters N    timed iterations per L (default "
         << DEFAULT_ITERS << ")\n"
         << "  --csv FILE   output CSV path (default gmp_multiply_bench.csv)\n"
         << "  --append     append rows to CSV instead of overwriting\n"
         << "  L ...        limb spec: if L < 64, use 1<<L limbs; else L limbs\n"
         << "               ranges inclusive: 16-24 -> 16,17,...,24\n"
         << "               (default sweep: log2 sizes 6..22)\n"
         << "\n"
         << "  Compile with -DNATIVE_HOST_LIMBS to benchmark 64-bit limbs\n"
         << "  instead of the default 32-bit limbs.\n";
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

static void write_csv(const string& path, const vector<BenchRow>& rows, bool append)
{
    const bool write_header = !append || !file_nonempty(path);
    ofstream csv(path, append ? ios::app : ios::trunc);
    if (!csv) {
        cerr << "Failed to open CSV: " << path << "\n";
        exit(1);
    }

    if (write_header) {
        csv << "engine,limb_bits,operand_bits,L_arg,L,warmup,iters,"
            << "mean_ms,stddev_ms,min_ms,max_ms,"
            << "import_a_mean_ms,import_b_mean_ms,multiply_mean_ms,export_mean_ms\n";
    }
    csv << fixed << setprecision(6);

    for (const BenchRow& row : rows) {
        csv << "gmp" << ","
            << LIMB_BITS_VAL << ","
            << (row.L * LIMB_BITS_VAL) << ","
            << row.L_arg << ","
            << row.L << ","
            << row.warmup << ","
            << row.iters << ","
            << row.total.mean_ms << ","
            << row.total.stddev_ms << ","
            << row.total.min_ms << ","
            << row.total.max_ms << ","
            << row.import_a.mean_ms << ","
            << row.import_b.mean_ms << ","
            << row.multiply.mean_ms << ","
            << row.export_stage.mean_ms << "\n";
    }
}

static void print_row(const BenchRow& row)
{
    cout << fixed << setprecision(3);
    cout << "L_arg=" << setw(3) << row.L_arg
         << "  L=" << setw(10) << row.L
         << "  bits=" << setw(12) << (row.L * LIMB_BITS_VAL) << "\n";
    cout << "  total:    mean=" << setw(9) << row.total.mean_ms
         << "  stddev=" << setw(8) << row.total.stddev_ms
         << "  min=" << setw(9) << row.total.min_ms
         << "  max=" << setw(9) << row.total.max_ms << " ms\n";
    cout << "  import_a=" << row.import_a.mean_ms
         << "  import_b=" << row.import_b.mean_ms
         << "  multiply=" << row.multiply.mean_ms
         << "  export=" << row.export_stage.mean_ms << " ms\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    int warmup = DEFAULT_WARMUP;
    int iters = DEFAULT_ITERS;
    string csv_path = "gmp_multiply_bench.csv";
    bool csv_append = false;
    vector<size_t> L_args;

    for (int i = 1; i < argc; i++) {
        const string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--warmup") {
            if (i + 1 >= argc) { cerr << "Missing value for --warmup\n"; return 1; }
            warmup = atoi(argv[++i]);
            continue;
        }
        if (arg == "--iters") {
            if (i + 1 >= argc) { cerr << "Missing value for --iters\n"; return 1; }
            iters = atoi(argv[++i]);
            continue;
        }
        if (arg == "--csv") {
            if (i + 1 >= argc) { cerr << "Missing value for --csv\n"; return 1; }
            csv_path = argv[++i];
            continue;
        }
        if (arg == "--append") {
            csv_append = true;
            continue;
        }

        if (!append_l_spec(arg, L_args)) {
            cerr << "Invalid limb spec: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (L_args.empty())
        L_args = DEFAULT_L_VALUES;

    if (warmup < 0 || iters <= 0) {
        cerr << "warmup must be >= 0 and iters must be > 0\n";
        return 1;
    }

    cout << "GMP multiply benchmark"
         << " (limb_bits=" << LIMB_BITS_VAL
         << ", warmup=" << warmup
         << ", iters=" << iters << ")\n";
    cout << string(72, '-') << "\n";

    vector<BenchRow> rows;
    rows.reserve(L_args.size());

    uint64_t seed = 1234;
    for (size_t L_arg : L_args) {
        const size_t L = resolve_limb_count(L_arg);
        cout << "Benchmarking L_arg=" << L_arg << " (L=" << L << ") ... " << flush;
        BenchRow row = benchmark_L(L_arg, warmup, iters, seed);
        seed += 17;
        rows.push_back(row);
        cout << "done\n";
        print_row(row);
    }

    write_csv(csv_path, rows, csv_append);
    cout << string(72, '-') << "\n";
    cout << "Wrote " << rows.size() << " rows to " << csv_path << "\n";
    return 0;
}