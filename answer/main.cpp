#include <ostream>
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef __GNUC__
#include <x86intrin.h>
#endif

#ifdef __GNUC__
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("arch=skylake"))), apply_to = function)
/* 最後に↓を貼る
#ifdef __clang__
#pragma clang attribute pop
#endif
*/
#else // defined(__clang__)
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")
#pragma GCC optimize("O3")
#pragma GCC optimize("Ofast")
//#pragma GCC optimize("unroll-loops")
#endif // defined(__clang__)
#endif // defined(__GNUC__)

// ========================== macroes ==========================

#define rep(i, n) for (auto i = 0; (i) < (n); (i)++)
#define rep1(i, n) for (auto i = 1; (i) <= (n); (i)++)
#define rep3(i, s, n) for (auto i = (s); (i) < (n); (i)++)

//#define NDEBUG

#ifndef NDEBUG
#define VISUALIZE
#endif

#ifndef NDEBUG
#define ASSERT(expr, ...)                                                                                                                            \
    do {                                                                                                                                             \
        if (!(expr)) {                                                                                                                               \
            printf("%s(%d): Assertion failed.\n", __FILE__, __LINE__);                                                                               \
            printf(__VA_ARGS__);                                                                                                                     \
            abort();                                                                                                                                 \
        }                                                                                                                                            \
    } while (false)
#else
#define ASSERT(...)
#endif

#define ASSERT_RANGE(value, left, right) ASSERT((left <= value) && (value < right), "`%s` (%d) is out of range [%d, %d)", #value, value, left, right)

#define CHECK(var)                                                                                                                                   \
    do {                                                                                                                                             \
        std::cout << #var << '=' << var << endl;                                                                                                     \
    } while (false)

// ========================== utils ==========================

using namespace std;
using ll = long long;
constexpr double PI = 3.1415926535897932;

template <class T, class S> inline bool chmin(T& m, S q) {
    if (m > q) {
        m = q;
        return true;
    } else
        return false;
}

template <class T, class S> inline bool chmax(T& m, const S q) {
    if (m < q) {
        m = q;
        return true;
    } else
        return false;
}

// クリッピング  // clamp (C++17) と等価
template <class T> inline T clipped(const T& v, const T& low, const T& high) { return min(max(v, low), high); }

// 2 次元ベクトル
template <typename T> struct Vec2 {
    /*
    y 軸正は下方向
    x 軸正は右方向
    回転は時計回りが正（y 軸正を上と考えると反時計回りになる）
    */
    T y, x;
    constexpr inline Vec2() = default;
    constexpr Vec2(const T& arg_y, const T& arg_x) : y(arg_y), x(arg_x) {}
    inline Vec2(const Vec2&) = default;            // コピー
    inline Vec2(Vec2&&) = default;                 // ムーブ
    inline Vec2& operator=(const Vec2&) = default; // 代入
    inline Vec2& operator=(Vec2&&) = default;      // ムーブ代入
    template <typename S> constexpr inline Vec2(const Vec2<S>& v) : y((T)v.y), x((T)v.x) {}
    inline Vec2 operator+(const Vec2& rhs) const { return Vec2(y + rhs.y, x + rhs.x); }
    inline Vec2 operator+(const T& rhs) const { return Vec2(y + rhs, x + rhs); }
    inline Vec2 operator-(const Vec2& rhs) const { return Vec2(y - rhs.y, x - rhs.x); }
    template <typename S> inline Vec2 operator*(const S& rhs) const { return Vec2(y * rhs, x * rhs); }
    inline Vec2 operator*(const Vec2& rhs) const { // x + yj とみなす
        return Vec2(x * rhs.y + y * rhs.x, x * rhs.x - y * rhs.y);
    }
    template <typename S> inline Vec2 operator/(const S& rhs) const {
        ASSERT(rhs != 0.0, "Zero division!");
        return Vec2(y / rhs, x / rhs);
    }
    inline Vec2 operator/(const Vec2& rhs) const { // x + yj とみなす
        return (*this) * rhs.inv();
    }
    inline Vec2& operator+=(const Vec2& rhs) {
        y += rhs.y;
        x += rhs.x;
        return *this;
    }
    inline Vec2& operator-=(const Vec2& rhs) {
        y -= rhs.y;
        x -= rhs.x;
        return *this;
    }
    template <typename S> inline Vec2& operator*=(const S& rhs) const {
        y *= rhs;
        x *= rhs;
        return *this;
    }
    inline Vec2& operator*=(const Vec2& rhs) {
        *this = (*this) * rhs;
        return *this;
    }
    inline Vec2& operator/=(const Vec2& rhs) {
        *this = (*this) / rhs;
        return *this;
    }
    inline bool operator!=(const Vec2& rhs) const { return x != rhs.x || y != rhs.y; }
    inline bool operator==(const Vec2& rhs) const { return x == rhs.x && y == rhs.y; }
    inline void rotate(const double& rad) { *this = rotated(rad); }
    inline Vec2<double> rotated(const double& rad) const { return (*this) * rotation(rad); }
    static inline Vec2<double> rotation(const double& rad) { return Vec2(sin(rad), cos(rad)); }
    static inline Vec2<double> rotation_deg(const double& deg) { return rotation(PI * deg / 180.0); }
    inline Vec2<double> rounded() const { return Vec2<double>(round(y), round(x)); }
    inline Vec2<double> inv() const { // x + yj とみなす
        const double norm_sq = l2_norm_square();
        ASSERT(norm_sq != 0.0, "Zero division!");
        return Vec2(-y / norm_sq, x / norm_sq);
    }
    inline double l2_norm() const { return sqrt(x * x + y * y); }
    inline double l2_norm_square() const { return x * x + y * y; }
    inline T l1_norm() const { return std::abs(x) + std::abs(y); }
    inline double abs() const { return l2_norm(); }
    inline double phase() const { // [-PI, PI) のはず
        return atan2(y, x);
    }
    inline double phase_deg() const { // [-180, 180) のはず
        return phase() / PI * 180.0;
    }
};
template <typename T, typename S> inline Vec2<T> operator*(const S& lhs, const Vec2<T>& rhs) { return rhs * lhs; }
template <typename T> ostream& operator<<(ostream& os, const Vec2<T>& vec) {
    os << vec.y << ' ' << vec.x;
    return os;
}

// 乱数
struct Random {
    using ull = unsigned long long;
    ull seed;
    inline Random(ull aSeed) : seed(aSeed) { ASSERT(seed != 0ull, "Seed should not be 0."); }
    const inline ull& next() {
        seed ^= seed << 9;
        seed ^= seed >> 7;
        return seed;
    }
    // (0.0, 1.0)
    inline double random() { return (double)next() / (double)ULLONG_MAX; }
    // [0, right)
    inline int randint(const int right) { return next() % (ull)right; }
    // [left, right)
    inline int randint(const int left, const int right) { return next() % (ull)(right - left) + left; }
};

// 2 次元配列
template <class T, int height, int width> struct Board {
    array<T, height * width> data;
    template <class Int> constexpr inline auto& operator[](const Vec2<Int>& p) { return data[width * p.y + p.x]; }
    template <class Int> constexpr inline const auto& operator[](const Vec2<Int>& p) const { return data[width * p.y + p.x]; }
    template <class Int> constexpr inline auto& operator[](const initializer_list<Int>& p) { return data[width * *p.begin() + *(p.begin() + 1)]; }
    template <class Int> constexpr inline const auto& operator[](const initializer_list<Int>& p) const {
        return data[width * *p.begin() + *(p.begin() + 1)];
    }
    constexpr inline void Fill(const T& fill_value) { fill(data.begin(), data.end(), fill_value); }
    void Print() const {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cout << data[width * y + x] << " \n"[x == width - 1];
            }
        }
    }
};

// キュー
template <class T, int max_size> struct Queue {
    array<T, max_size> data;
    int left, right;
    inline Queue() : data(), left(0), right(0) {}
    inline Queue(initializer_list<T> init) : data(init.begin(), init.end()), left(0), right(init.size()) {}

    inline bool empty() const { return left == right; }
    inline void push(const T& value) {
        data[right] = value;
        right++;
    }
    inline void pop() { left++; }
    const inline T& front() const { return data[left]; }
    template <class... Args> inline void emplace(const Args&... args) {
        data[right] = T(args...);
        right++;
    }
    inline void clear() {
        left = 0;
        right = 0;
    }
    inline int size() const { return right - left; }
};

// スタック  // コンストラクタ呼ぶタイミングとかが考えられてなくて良くない
template <class T, int max_size> struct Stack {
    array<T, max_size> data;
    int right;
    inline Stack() : data(), right(0) {}
    inline Stack(const int n) : data(), right(0) { resize(n); }
    inline Stack(const int n, const T& val) : data(), right(0) { resize(n, val); }
    inline Stack(const initializer_list<T>& init) : data(), right(init.size()) {
        memcpy(&data[0], init.begin(), sizeof(T) * init.size());
    }                                                           // これ大丈夫か？
    inline Stack(const Stack& rhs) : data(), right(rhs.right) { // コピー
        for (int i = 0; i < right; i++) {
            data[i] = rhs.data[i];
        }
    }
    template <class S> inline Stack(const Stack<S, max_size>& rhs) : data(), right(rhs.right) {
        for (int i = 0; i < right; i++) {
            data[i] = rhs.data[i];
        }
    }
    Stack& operator=(const Stack& rhs) {
        right = rhs.right;
        for (int i = 0; i < right; i++) {
            data[i] = rhs.data[i];
        }
        return *this;
    }
    Stack& operator=(const vector<T>& rhs) {
        right = (int)rhs.size();
        ASSERT(right <= max_size, "too big vector");
        for (int i = 0; i < right; i++) {
            data[i] = rhs[i];
        }
        return *this;
    }
    Stack& operator=(Stack&&) = default;
    inline bool empty() const { return 0 == right; }
    inline void push(const T& value) {
        ASSERT_RANGE(right, 0, max_size);
        data[right] = value;
        right++;
    }
    inline T pop() {
        right--;
        ASSERT_RANGE(right, 0, max_size);
        return data[right];
    }
    const inline T& top() const { return data[right - 1]; }
    template <class... Args> inline void emplace(const Args&... args) {
        ASSERT_RANGE(right, 0, max_size);
        data[right] = T(args...);
        right++;
    }
    inline void clear() { right = 0; }
    inline void insert(const int& idx, const T& value) {
        ASSERT_RANGE(idx, 0, right + 1);
        ASSERT_RANGE(right, 0, max_size);
        int i = right;
        right++;
        while (i != idx) {
            data[i] = data[i - 1];
            i--;
        }
        data[idx] = value;
    }
    inline void del(const int& idx) {
        ASSERT_RANGE(idx, 0, right);
        right--;
        for (int i = idx; i < right; i++) {
            data[i] = data[i + 1];
        }
    }
    inline int index(const T& value) const {
        for (int i = 0; i < right; i++) {
            if (value == data[i])
                return i;
        }
        return -1;
    }
    inline void remove(const T& value) {
        int idx = index(value);
        ASSERT(idx != -1, "not contain the value.");
        del(idx);
    }
    inline void resize(const int& sz) {
        ASSERT_RANGE(sz, 0, max_size + 1);
        for (; right < sz; right++) {
            data[right].~T();
            new (&data[right]) T();
        }
        right = sz;
    }
    inline void resize(const int& sz, const T& fill_value) {
        ASSERT_RANGE(sz, 0, max_size + 1);
        for (; right < sz; right++) {
            data[right].~T();
            new (&data[right]) T(fill_value);
        }
        right = sz;
    }
    inline int size() const { return right; }
    inline T& operator[](const int n) {
        ASSERT_RANGE(n, 0, right);
        return data[n];
    }
    inline const T& operator[](const int n) const {
        ASSERT_RANGE(n, 0, right);
        return data[n];
    }
    inline T* begin() { return (T*)data.data(); }
    inline const T* begin() const { return (const T*)data.data(); }
    inline T* end() { return (T*)data.data() + right; }
    inline const T* end() const { return (const T*)data.data() + right; }
    inline T& front() {
        ASSERT(right > 0, "no data.");
        return data[0];
    }
    const inline T& front() const {
        ASSERT(right > 0, "no data.");
        return data[0];
    }
    inline T& back() {
        ASSERT(right > 0, "no data.");
        return data[right - 1];
    }
    const inline T& back() const {
        ASSERT(right > 0, "no data.");
        return data[right - 1];
    }
    inline bool contains(const T& value) const {
        for (const auto& dat : *this) {
            if (value == dat)
                return true;
        }
        return false;
    }
    inline vector<T> ToVector() { return vector<T>(begin(), end()); }
    inline void Print(ostream& os = cout) {
        for (int i = 0; i < right; i++) {
            os << data[i] << (i == right - 1 ? "" : " ");
        }
        os << endl;
    }
};

template <class T, int size = 0x100000, class KeyType = unsigned> struct MinimumHashMap {
    // ハッシュの値が size 以下
    array<T, size> data;
    Stack<int, size> used;
    constexpr static KeyType mask = size - 1;
    inline MinimumHashMap() {
        static_assert((size & size - 1) == 0, "not pow of 2");
        memset(&data[0], (unsigned char)-1, sizeof(data));
    }
    inline T& operator[](const KeyType& key) {
        if (data[key] == (T)-1)
            used.push(key);
        return data[key];
    }
    inline void clear() {
        for (const auto& key : used)
            data[key] = (T)-1;
        used.right = 0;
    }
};

// 時間 (秒)
inline double time() {
    return static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now().time_since_epoch()).count()) * 1e-9;
}

// 重複除去
template <typename T> inline void deduplicate(vector<T>& vec) {
    sort(vec.begin(), vec.end());
    vec.erase(unique(vec.begin(), vec.end()), vec.end());
}

// 2 分法
template <typename T> inline int search_sorted(const vector<T>& vec, const T& a) { return lower_bound(vec.begin(), vec.end(), a) - vec.begin(); }

// argsort
template <typename T, int n, typename result_type, bool reverse = false> inline auto Argsort(const array<T, n>& vec) {
    array<result_type, n> res;
    iota(res.begin(), res.end(), 0);
    sort(res.begin(), res.end(), [&](const result_type& l, const result_type& r) { return reverse ? vec[l] > vec[r] : vec[l] < vec[r]; });
    return res;
}

// popcount  // SSE 4.2 を使うべき
inline int popcount(const unsigned int& x) {
#ifdef _MSC_VER
    return (int)__popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}
inline int popcount(const unsigned long long& x) {
#ifdef _MSC_VER
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}

// x >> n & 1 が 1 になる最小の n ( x==0 は未定義 )
inline int CountRightZero(const unsigned int& x) {
#ifdef _MSC_VER
    unsigned long r;
    _BitScanForward(&r, x);
    return (int)r;
#else
    return __builtin_ctz(x);
#endif
}
inline int CountRightZero(const unsigned long long& x) {
#ifdef _MSC_VER
    unsigned long r;
    _BitScanForward64(&r, x);
    return (int)r;
#else
    return __builtin_ctzll(x);
#endif
}

inline double MonotonicallyIncreasingFunction(const double& h, const double& x) {
    // 0 < h < 1
    // f(0) = 0, f(1) = 1, f(0.5) = h
    ASSERT(h > 0.0 && h < 1.0, "0 < h < 1 not satisfied");
    if (h == 0.5)
        return x;
    const double& a = (1.0 - 2.0 * h) / (h * h);
    return expm1(log1p(a) * x) / a;
}
inline double MonotonicFunction(const double& start, const double& end, const double& h, const double& x) {
    // h: x = 0.5 での進捗率
    return MonotonicallyIncreasingFunction(h, x) * (end - start) + start;
}

template <typename T> struct Slice {
    T *left, *right;
    inline Slice(T* const& l, T* const& r) : left(l), right(r) {}
    inline T* begin() { return left; }
    inline const T* begin() const { return (const T*)left; }
    inline T* end() { return right; }
    inline const T* end() const { return (const T*)right; }
};

struct Edge {
    int from, to;
    bool operator<(const Edge& rhs) const { return make_pair(from, to) < make_pair(rhs.from, rhs.to); }
};
template <int max_n, int max_m> struct Graph {
    int n, m;
    Stack<int, max_m> edges;
    Stack<int, max_n + 1> lefts;

    Graph() = default;
    template <class Container> Graph(const int& n_, Container edges_) {
        // edges_ は 0-origin
        n = n_;
        m = edges_.size();
        sort(edges_.begin(), edges_.end());
        edges.resize(m);
        lefts.resize(n + 1);
        for (int i = 0; i < m; i++) {
            edges[i] = edges_[i].to;
        }
        auto idx_edges = 0;
        for (int v = 0; v <= n; v++) {
            lefts[v] = idx_edges;
            while (idx_edges < m && edges_[idx_edges].from == v) {
                idx_edges++;
            }
        }
    }
    inline Slice<int> operator[](const int& v) { return Slice<int>(edges.begin() + lefts[v], edges.begin() + lefts[v + 1]); }
    inline Slice<int> operator[](const int& v) const { return Slice<int>(edges.begin() + lefts[v], edges.begin() + lefts[v + 1]); }
};

// ========================= ライブラリここまで =========================

namespace input {
constexpr auto N = 1000;             // タスク数
constexpr auto M = 20;               // 人数
auto K = 0;                          // 技能数 [10, 20]
auto R = 0;                          // 依存関係数 [1000, 3000]
auto d = array<array<int, 20>, N>(); // 各タスクの要求技能レベル
auto edges = Stack<Edge, 3000>();
auto G = Graph<1000, 3000>();
} // namespace input

namespace common {
struct FinishedTask {
    int task; // タスク番号
    int t;    // かかった時間
};
auto finished_task = array<Stack<FinishedTask, 300>, input::M>();
} // namespace common

namespace prediction {
// clang-format off
constexpr auto initial_expected_time_all = array<array<double, 41>, 21 - 10>{
    array<double, 41>{0.00000000,0.03195550,0.12764459,0.28633275,0.50694044,0.78798031,1.12752661,1.52334824,1.97293306,2.47349386,3.02206879,3.61554764,4.25077833,4.92452404,5.63366531,6.37513917,7.14593268,7.94328855,8.76460051,9.60745786,10.46961478,11.34906941,12.24398152,13.15267632,14.07363038,15.00545454,15.94688350,16.89680181,17.85415465,18.81803528,19.78759880,20.76210771,21.74088590,22.72332859,23.70890663,24.69715341,25.68765228,26.68003924,27.67399626,28.66925221,29.66557043},
                     {0.00000000,0.03375507,0.13479280,0.30232359,0.53503714,0.83109422,1.18821984,1.60379821,2.07483481,2.59814405,3.17032517,3.78784261,4.44711809,5.14465639,5.87701705,6.64087846,7.43313992,8.25086437,9.09140174,9.95229535,10.83129715,11.72641403,12.63579902,13.55780194,14.49092046,15.43382513,16.38529058,17.34423924,18.30969105,19.28077897,20.25672931,21.23684461,22.22052043,23.20720286,24.19642596,25.18777684,26.18089993,27.17548576,28.17127365,29.16803555,30.16557593},
                     {0.00000000,0.03548827,0.14162692,0.31753566,0.56164265,0.87179023,1.24533322,1.67921071,2.17002902,2.71417059,3.30775451,3.94685818,4.62761436,5.34621262,6.09895199,6.88235456,7.69317303,8.52843703,9.38540342,10.26159444,11.15480957,12.06301675,12.98440064,13.91734807,14.86038478,15.81222598,16.77172837,17.73783899,18.70964139,19.68631741,20.66714977,21.65150107,22.63881625,23.62861416,24.62047630,25.61404897,26.60902074,27.60512326,28.60214074,29.59988973,30.59821340},
                     {0.00000000,0.03714665,0.14821506,0.33211242,0.58706435,0.91056644,1.29962778,1.75074323,2.26007846,2.82356172,3.43693889,4.09592986,4.79630952,5.53395525,6.30500652,7.10582112,7.93306440,8.78371468,9.65500288,10.54446317,11.44988486,12.36926663,13.30081732,14.24295448,15.19427511,16.15352678,17.11959982,18.09151148,19.06839577,20.04949317,21.03413971,22.02175394,23.01184705,24.00398971,24.99781813,25.99302015,26.98932389,27.98650834,28.98439283,29.98282089,30.98167448},
                     {0.00000000,0.03870869,0.15450891,0.34609529,0.61139433,0.94759819,1.35134479,1.81867565,2.34532990,2.92680086,3.55844500,4.23561749,4.95378416,5.70860638,6.49607071,7.31242514,8.15425781,9.01847811,9.90231655,10.80329931,11.71922071,12.64813154,13.58830317,14.53817364,15.49638754,16.46173673,17.43317494,18.40977500,19.39072764,20.37533616,21.36299203,22.35316128,23.34540130,24.33933119,25.33462917,26.33102647,27.32829641,28.32625246,29.32474321,30.32364504,31.32285932},
                     {0.00000000,0.04025301,0.16053433,0.35936740,0.63448142,0.98269419,1.40018635,1.88262209,2.42528440,3.02326113,3.67153367,4.36510486,5.09926253,5.86947710,6.67155954,7.50165792,8.35629411,9.23235321,10.12708749,11.03803485,11.96302135,12.90010825,13.84759549,14.80397866,15.76794939,16.73836289,17.71420838,18.69461846,19.67885476,20.66625585,21.65626853,22.64841140,23.64228461,24.63755584,25.63394431,26.63121786,27.62918458,28.62768778,29.62660298,30.62582707,31.62528271},
                     {0.00000000,0.04173578,0.16637503,0.37229071,0.65689547,1.01667152,1.44742194,1.94436249,2.50234831,3.11597680,3.77991613,4.48891009,5.23793134,6.02228942,6.83769340,7.68023648,8.54641528,9.43309718,10.33751674,11.25724162,12.19012941,13.13428164,14.08805074,15.04999437,16.01884871,16.99350899,17.97302803,18.95659401,19.94350381,20.93315309,21.92504003,22.91873990,23.91389274,24.91020339,25.90742534,26.90535904,27.90384104,28.90274176,29.90195923,30.90140995,31.90103247},
                     {0.00000000,0.04309362,0.17184320,0.38445445,0.67803350,1.04875562,1.49195369,2.00242046,2.57457124,3.20266816,3.88101221,4.60405210,5.36657895,6.16374995,6.99116480,7.84486014,8.72128939,9.61734345,10.53027828,11.45767629,12.39743270,13.34770439,14.30687143,15.27354759,16.24650855,17.22471601,18.20727821,19.19342967,20.18251481,21.17398018,22.16736953,23.16229217,24.15843512,25.15553597,26.15338503,27.15180983,28.15067407,29.14986524,30.14929914,31.14890988,32.14864719},
                     {0.00000000,0.04448622,0.17728204,0.39646918,0.69879925,1.08008259,1.53524296,2.05863647,2.64426634,3.28603487,3.97790499,4.71409501,5.48917464,6.29818870,7.13662881,8.00048511,8.88624273,9.79077099,10.71135107,11.64559958,12.59144107,13.54707322,14.51093433,15.48169882,16.45818844,17.43941361,18.42452540,19.41281677,20.40367955,21.39661577,22.39120806,23.38710747,24.38403699,25.38176393,26.38009881,27.37889723,28.37804308,29.37744449,30.37703285,31.37675530,32.37657195},
                     {0.00000000,0.04581945,0.18253929,0.40799387,0.71869862,1.11006352,1.57662324,2.11227307,2.71065086,3.36525832,4.06976614,4.81811648,5.60475964,6.42460864,7.27311316,8.14623503,9.04042338,9.95256707,10.87996613,11.82027496,12.77147161,13.73179486,14.69973191,15.67400234,16.65349642,17.63727032,18.62453492,19.61461930,20.60696697,21.60111619,22.59668808,23.59337351,24.59092183,25.58912995,26.58783435,27.58691575,28.58627289,29.58582806,30.58552723,31.58532932,32.58520067},
                     {0.00000000,0.04709027,0.18759671,0.41913519,0.73789910,1.13896285,1.61645548,2.16385160,2.77432466,3.44105690,4.15743982,4.91723551,5.71466434,6.54457513,7.40233227,8.28388692,9.18570595,10.10469553,11.03821275,11.98395022,12.93990235,13.90435721,14.87586454,15.85318746,16.83526358,17.82121306,18.81028828,19.80186036,20.79542151,21.79055190,22.78690382,23.78420619,24.78223544,25.78081317,26.77979867,27.77908703,28.77859672,29.77826411,30.77804262,31.77789796,32.77780576},
};
// clang-format on

auto initial_expected_time = array<double, 41>();
auto initial_expected_skill = array<double, 11>{10.34388673, 9.84151079, 9.40491307, 9.02080870, 8.67996031, 8.37518411,
                                                8.10086558,  7.85121853, 7.62445622, 7.41485945, 7.22255002}; // [技能数 - 10] := スキルの予測値
auto expected_time = array<array<double, input::M>, input::N>();                                              // 期待所要時間
auto expected_skill = array<array<double, 20>, input::M>();                                                   // 各メンバーの能力の予測値
inline void PrintExpectedSkill(const int& member) {
#ifdef VISUALIZE
    cout << "#s " << member + 1;
    rep(skill, input::K) cout << " " << expected_skill[member][skill];
    cout << endl;
#endif
}
void Initialize() {
    ASSERT(input::K != 0, "input がまだだよ");
    initial_expected_time = initial_expected_time_all[input::K - 10];
    rep(task, input::N) {
        rep(member, input::M) {
            rep(skill, input::K) { expected_time[task][member] += initial_expected_time[input::d[task][skill]]; }
        }
    }
    rep(member, input::M) {
        rep(skill, input::K) { expected_skill[member][skill] = initial_expected_skill[input::K - 10]; }
        PrintExpectedSkill(member);
    }
}
inline void Update(const int& member) {
    // タスク-メンバー の時間を予測
    // TODO
    // for (const auto& task : task_queue) {
    //     expected_time[task][member] = hogehoge
    // }
}

} // namespace prediction

void GreedySolution() {
    enum class TaskStatus { NotStarted, InProgress, Finished };
    auto task_status = array<TaskStatus, input::N>();
    auto member_status = array<int, input::M>(); // -1: 空き
    fill(member_status.begin(), member_status.end(), -1);
    auto in_dims = array<int, input::N>(); // 入次数
    for (const auto& e : input::edges) {
        in_dims[e.to]++;
    }
    auto open_tasks = Stack<int, input::N>();
    rep(task, input::N) {
        if (in_dims[task] == 0) {
            open_tasks.push(task);
        }
    }
    auto open_members = Stack<int, input::N>();
    rep(member, input::M) { open_members.push(member); }
    // open_tasks.Print();
    // open_members.Print();

    // 1 日目
    {
        const auto m = min(20, open_tasks.size());
        cout << m;
        rep(i, m) {
            // 着手 ... open_tasks から pop, open_members から pop, member_status, task_status の更新
            const auto task = open_tasks[0];
            swap(open_tasks[0], open_tasks.back());
            open_tasks.pop();

            const auto member = open_members[0];
            swap(open_members[0], open_members.back());
            open_members.pop();

            member_status[member] = task;
            task_status[task] = TaskStatus::InProgress;
            cout << " " << member + 1 << " " << task + 1;
        }
        cout << endl;
    }

    // 2 日目以降
    struct Output {
        int member, task;
    };
    auto outputs = Stack<Output, 20>();
    while (true) {
        int n;
        cin >> n;
        if (n == -1)
            return;
        // cerr << "n=" << n << endl;

        rep(i, n) {
            // 完了 ... member_status, task_status in_dims の更新, open_tasks, open_members の追加
            int member;
            cin >> member;
            member--;
            // cerr << "member=" << member << endl;
            auto task = member_status[member];
            member_status[member] = -1;
            task_status[task] = TaskStatus::Finished;
            // cerr << "task=" << task << endl;
            for (const auto& u : input::G[task]) {
                // cerr << "u=" << u << endl;
                in_dims[u]--;
                if (in_dims[u] == 0) {
                    open_tasks.push(u);
                }
            }
            open_members.push(member);
        }

        // 着手
        int m = min(open_tasks.size(), open_members.size());
        // cerr << "m=" << m << endl;
        cout << m;
        rep(i, m) {
            // 着手 ... open_tasks から pop, open_members から pop, member_status, task_status の更新
            const auto task = open_tasks[0];
            swap(open_tasks[0], open_tasks.back());
            open_tasks.pop();

            const auto member = open_members[0];
            swap(open_members[0], open_members.back());
            open_members.pop();

            member_status[member] = task;
            task_status[task] = TaskStatus::InProgress;
            cout << " " << member + 1 << " " << task + 1;
        }
        cout << endl;
    }
}
void Solve() {
    // 1. 初期化
    {
        // 入力の読み込み
        int dummy;
        cin >> dummy >> dummy >> input::K >> input::R;
        input::edges.resize(input::R);
        rep(idx_task, input::N) {
            rep(idx_skill, input::K) { cin >> input::d[idx_task][idx_skill]; }
        }
        rep(idx_edges, input::R) {
            cin >> input::edges[idx_edges].from >> input::edges[idx_edges].to;
            input::edges[idx_edges].from--;
            input::edges[idx_edges].to--;
        }

        // グラフ作成
        new (&input::G) decltype(input::G)(input::N, input::edges);

        prediction::Initialize();
    }

    GreedySolution();
}

int main() {
    Solve();
    // TODO
}

#ifdef __clang__
#pragma clang attribute pop
#endif