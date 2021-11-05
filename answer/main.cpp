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
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")
#pragma GCC optimize("O3")
#pragma GCC optimize("Ofast")
//#pragma GCC optimize("unroll-loops")

#pragma clang attribute push(__attribute__((target("arch=skylake"))), apply_to = function)
/* 最後に↓を貼る
#ifdef __GNUC__
#pragma clang attribute pop
#endif
*/
#endif

// ========================== macroes ==========================

#define rep(i, n) for (ll(i) = 0; (i) < (n); (i)++)
#define rep1(i, n) for (ll(i) = 1; (i) <= (n); (i)++)
#define rep3(i, s, n) for (ll(i) = (s); (i) < (n); (i)++)

//#define NDEBUG

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
// TODO
}

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
    int n_started_tasks = 20;
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
    }
    // cerr << input::G[993].begin() << " " << input::G[993].end() << endl;

    GreedySolution();
}

int main() {
    Solve();
    // TODO
}

#ifdef __GNUC__
#pragma clang attribute pop
#endif