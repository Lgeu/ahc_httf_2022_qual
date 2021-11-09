// 参考: https://github.com/pakwah/Revised-Simplex-Method

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>
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

using namespace std;
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

namespace simplex {
using namespace std;

constexpr double epsilon1 = 0.00001;
constexpr double epsilon2 = 0.00000001;
constexpr auto MAX_N = 5000;
constexpr auto MAX_M = 240;
constexpr auto MAX_PIVOTS_SIZE = 2000;
static_assert(MAX_N % 8 == 0);
static_assert(MAX_M % 8 == 0);
constexpr auto MAX_N_COMPONENTS = 100000;

template <typename T> struct Slice {
    T *left, *right;
    inline Slice(T* const& l, T* const& r) : left(l), right(r) {}
    inline T* begin() { return left; }
    inline const T* begin() const { return (const T*)left; }
    inline T* end() { return right; }
    inline const T* end() const { return (const T*)right; }
    inline int size() const { return distance(left, right); }
    inline T& operator[](const int& idx) { return left[idx]; }
    inline const T& operator[](const int& idx) const { return left[idx]; }
};

struct SparseMatrixComponent {
    int row, col;
    double weight;
};

template <int max_n_cols, int max_n_components> struct CSCMatrix {
    struct Component {
        int row;
        double weight;
    };
    int n_cols, n_components;
    array<Component, max_n_components> components;
    array<int, max_n_cols + 1> lefts;

    CSCMatrix() = default;
    template <class ContainerOfSparseMatrixComponents> CSCMatrix(const int& n_cols_, ContainerOfSparseMatrixComponents edges_) {
        // edges_ は 0-origin, 同じ要素の重複は未定義動作を起こすので注意
        assert(n_cols_ <= max_n_cols);
        assert(edges_.size() <= max_n_components);
        n_cols = n_cols_;
        n_components = edges_.size();
        sort(edges_.begin(), edges_.end(), [](const auto& l, const auto& r) { return make_pair(l.col, l.row) < make_pair(r.col, r.row); });
        for (int i = 0; i < n_components; i++) {
            components[i] = {edges_[i].row, edges_[i].weight};
        }
        auto idx_edges = 0;
        for (int col = 0; col <= n_cols; col++) {
            lefts[col] = idx_edges;
            while (idx_edges < n_components && edges_[idx_edges].col == col) {
                idx_edges++;
            }
        }
    }
    inline Slice<Component> operator[](const int& col) {
        return Slice<Component>(components.begin() + lefts[col], components.begin() + lefts[col + 1]);
    }
    inline Slice<Component> operator[](const int& col) const {
        return Slice<Component>(components.begin() + lefts[col], components.begin() + lefts[col + 1]);
    }
};

struct LPProblem {
    // maximize c^T x
    // s.t. Ax <= b
    enum class Status { NONE, OPTIMAL, INFEASIBLE, UNBOUNDED };
    alignas(64) array<double, MAX_N + MAX_M> c; // n
    // alignas(64) array<array<double, MAX_N + MAX_M>, MAX_M> A; // m * n
    Stack<SparseMatrixComponent, MAX_N_COMPONENTS> A_components;
    alignas(64) array<double, MAX_M> b;
    alignas(64) array<double, MAX_N + MAX_M> x; // スラック変数を含む解
    int n, m;                                   // 変数の数、制約の数
    double z;                                   // 最適化した目的変数の値
    Status status;
    void PrintSolution(ostream& os = cout) {
        for (int i = 0; i < n; i++) {
            os << x[i] << " \n"[i == n - 1];
        }
    }
};

void Solve(LPProblem& lp, const int& max_iteration = 2000) {

    // スラック変数を含めた目的関数の係数にする
    fill(lp.c.begin() + lp.n, lp.c.begin() + (lp.n + lp.m), 0.0);

    // b と b のラベル
    static array<double, MAX_M> b;
    static array<int, MAX_M> b_labels;
    copy(lp.b.begin(), lp.b.begin() + lp.m, b.begin());
    iota(b_labels.begin(), b_labels.begin() + lp.m, lp.n);

    // 非基底変数のラベル、長さ n, 範囲 [0, n + m)
    static array<int, MAX_N> nonbasic;
    iota(nonbasic.begin(), nonbasic.end(), 0);

    // A のスラック変数の列を単位行列で初期化
    static auto A_components = Stack<SparseMatrixComponent, MAX_N_COMPONENTS>();
    A_components = lp.A_components;
    for (int row = 0; row < lp.m; ++row) {
        A_components.push({row, lp.n + row, 1.0});
    }

    // A を構築
    static auto A = CSCMatrix<MAX_N + MAX_M, MAX_N_COMPONENTS>();
    new (&A) decltype(A)(lp.n + lp.m, A_components);

    // b が 0 以上であることを確認
    for (int row = 0; row < lp.m; ++row) {
        if (b[row] < 0.0) {
            lp.status = LPProblem::Status::INFEASIBLE;
            return;
        }
    }

    // ここからシンプレックス法

    // イータ行列
    struct Eta {
        alignas(64) array<double, MAX_M> values; // m
        int col;
    };

    int counter = 1;                           // イテレーション回数
    static array<Eta, MAX_PIVOTS_SIZE> pivots; // 過去のピボットを表すイータ行列
    int pivots_size = 0;                       // イータ行列の個数
    double z = 0.0;                            // 目的関数の初期値

    // 改訂シンプレックス法
    while (true) {
        // イータ行列を使って y を計算 (yB = c_b を解く)
        static array<double, MAX_M> y; // 長さ m

        // y を c_b で初期化
        for (int row = 0; row < lp.m; ++row) {
            y[row] = lp.c[b_labels[row]];
        }

        // yB = c_b を y について解く
        // y = c_b B^{-1}
        //   = c_b ... E_2^{-1} E_1^{-1}
        for (int idx_pivots = pivots_size - 1; idx_pivots >= 0; idx_pivots--) {
            const Eta& pivot = pivots[idx_pivots];
            const int col_to_change = pivot.col;
            double y_original = y[col_to_change] + pivot.values[col_to_change] * y[col_to_change];
            for (int row = 0; row < lp.m; ++row) {
                y_original -= pivot.values[row] * y[row];
            }
            y[col_to_change] = y_original / pivot.values[col_to_change];
        }

        // 入れる (entering) 列を選ぶ
        // 被約費用 \bar{c_N} = c_N - ya (ただし a は An の列) を求めて、
        // 値が正となる第 s 成分を選ぶ

        struct Variable {
            int label, position;
            double value;
        };

        static array<Variable, MAX_N> cnbars; // \bar{c_N} の成分のうち、値が正であるもの
        auto cnbars_size = 0;

        for (int i = 0; i < lp.n; ++i) {
            const int& var_label = nonbasic[i];
            const double& cni = lp.c[var_label]; // c_N の i 番目
            double yai = 0.0;                    // ya の i 番目
            for (const auto& a : A[var_label]) {
                yai += y[a.row] * a.weight;
            }
            const double cnbar = cni - yai;
            if (cnbar > epsilon1) {
                cnbars[cnbars_size] = {var_label, i, cnbar};
                cnbars_size++;
            }
        }

        // 目的関数の係数の降順にソート
        sort(cnbars.begin(), cnbars.begin() + cnbars_size, [](const Variable& a, const Variable& b) { return a.value > b.value; });

        // cnbars が空なら entering する候補が無く、最適解が得られた
        if (cnbars_size == 0) {
            goto optimal;
        }

        int entering_variable_index = 0;

        // Bd = a を解いて d を求める
        // d = B^{-1} a
        //   = ... E_2^{-1} E_1^{-1} a
        static array<double, MAX_M> d;
        int leaving_label;
        int leaving_row;
        double smallest_t;
        while (true) {
            leaving_label = -1;
            leaving_row = -1;
            smallest_t = 1e100;

            if (entering_variable_index == cnbars_size) {
                goto optimal;
            }

            const auto& entering_label = cnbars[entering_variable_index].label;

            // d を、基底に追加する列 a で初期化
            fill(d.begin(), d.begin() + lp.m, 0.0);
            for (const auto& a : A[entering_label]) {
                d[a.row] = a.weight;
            }

            // イータ行列の逆行列を順に掛けて d を求める
            for (int idx_pivots = 0; idx_pivots < pivots_size; idx_pivots++) {
                const Eta& pivot = pivots[idx_pivots];
                const int& row_to_change = pivot.col;
                const double& d_original = d[row_to_change];
                const auto d_row_to_change_tmp = d_original / pivot.values[row_to_change];
                for (int row = 0; row < lp.m; ++row) {
                    d[row] -= pivot.values[row] * d_row_to_change_tmp;
                }
                d[row_to_change] = d_row_to_change_tmp;
            }

            // b = x_B - td >= 0 を満たす最大のスカラー t を求める。
            // d のうち、正である成分について x_B[i] / d[i] を求め、
            // もっとも小さくなるものが t であり、そのときの i が基底からから取り除く列になる。
            // d の成分がすべて 0 以下であれば、問題は非有界。

            // 最小の比に対応する行を選ぶ
            for (int row = 0; row < lp.m; ++row) {
                if (d[row] <= 0.0) {
                    continue;
                }
                double t_row = b[row] / d[row];
                if (t_row < smallest_t) {
                    leaving_label = b_labels[row];
                    leaving_row = row;
                    smallest_t = t_row;
                }
            }

            // 比率が計算されなければ非有界なので終了する
            if (leaving_label == -1) {
                lp.status = LPProblem::Status::UNBOUNDED;
                return;
            }

            // d が小さすぎる値なら、次の entering 変数を見る
            if (d[leaving_row] > epsilon2) {
                break;
            } else {
                entering_variable_index++;
                continue;
            }
        }

        // この時点で、基底に追加する変数と取り除く変数のペアが決定している

        // 追加する変数の値を 1 にして、b を更新する
        // (追加する変数と取り除く変数を入れ替え、それ以外の基底変数の値を更新する)
        const Variable& entering_variable = cnbars[entering_variable_index];
        b[leaving_row] = smallest_t;
        b_labels[leaving_row] = entering_variable.label;

        const auto tmp = b[leaving_row];
        for (int row = 0; row < lp.m; ++row) {
            b[row] -= d[row] * smallest_t;
        }
        b[leaving_row] = tmp;

        // 新しいイータ行列を格納
        pivots[pivots_size].col = leaving_row;
        for (int i = 0; i < lp.m; i++) {
            pivots[pivots_size].values[i] = d[i];
        }
        pivots_size++;

        nonbasic[entering_variable.position] = leaving_label;

        // 目的関数の値を増加させる
        z += entering_variable.value * smallest_t;
        counter++;

        if (pivots_size == max_iteration)
            goto optimal;
    }

optimal:
    for (int row = 0; row < lp.m; ++row) {
        lp.x[b_labels[row]] = b[row];
    }
    for (int col = 0; col < lp.n; ++col) {
        lp.x[nonbasic[col]] = 0.0;
    }
    lp.z = z;
    lp.status = LPProblem::Status::OPTIMAL;
}
} // namespace simplex

int main(int argc, const char* argv[]) {
    using namespace std;
    static simplex::LPProblem lp;

    // 制約の数、変数の数
    cin >> lp.m >> lp.n;

    // c
    for (int col = 0; col < lp.n; ++col) {
        cin >> lp.c[col];
    }

    // A, b
    for (int row = 0; row < lp.m; ++row) {
        for (int col = 0; col <= lp.n; ++col) {
            if (col == lp.n) {
                cin >> lp.b[row];
            } else {
                double a;
                cin >> a;
                if (a != 0.0) {
                    lp.A_components.push({row, col, a});
                }
            }
        }
    }

    assert(argc == 2);
    auto max_iter = stoi(argv[1]);
    simplex::Solve(lp, max_iter);
    simplex::Solve(lp, max_iter);
    simplex::Solve(lp, max_iter);
    simplex::Solve(lp, max_iter);
    simplex::Solve(lp, max_iter);
    cout << "optimal value: " << lp.z << endl;
    lp.PrintSolution();
}