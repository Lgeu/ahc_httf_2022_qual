// from https://github.com/pakwah/Revised-Simplex-Method

//
//  main.cpp
//  RevisedSimplex
//
//  Created by Xuan Baihua on 10/18/15.
//  Copyright (c) 2015 Xuan Baihua. All rights reserved.
//

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <x86intrin.h>

namespace simplex {
using namespace std;

static const double epsilon1 = 0.00091;
static const double epsilon2 = 0.00000001;

static int m; // 不等式の数
static int n; // 変数の数

struct variable {
    int label;
    double value;
};

// イータ行列
struct Eta {
    int col;
    vector<double> values; // m
};

constexpr auto DEBUG_SIMPLEX = false;
template <typename... Args> inline void DebugSimplex(const char* format, Args const&... args) {
    if constexpr (DEBUG_SIMPLEX) {
        printf(format, args...);
    }
}

template <size_t siz1, size_t siz2> void printMatrix(const array<array<double, siz2>, siz1>& matrix, int r, int c) {
    for (int row = 0; row < r; ++row) {
        for (int col = 0; col < c; ++col) {
            DebugSimplex("%10.3f ", matrix[row][col]);
        }
        DebugSimplex("\n");
    }
}
template <size_t siz1, size_t siz2, size_t siz3, size_t siz4>
void printLPInfo(const array<double, siz4>& c, const array<double, siz3>& b, const array<array<double, siz2>, siz1>& matrix) {
    DebugSimplex("m = %d ", m);
    DebugSimplex("n = %d \n", n);

    DebugSimplex("c = ");

    for (int i = 0; i < m + n; ++i) {
        DebugSimplex("%10.3f ", c[i]);
    }

    DebugSimplex("\nb = ");

    for (int i = 0; i < m; ++i) {
        DebugSimplex("%10.3f ", b[i]);
    }

    DebugSimplex("\nA = \n");

    printMatrix(matrix, m, n + m);
};

template <size_t siz1, size_t siz2>
void printVariables(const array<int, siz1>& nonbasic, const array<double, siz2>& b, const array<int, siz2>& b_labels) {
    DebugSimplex("N = { ");

    for (int i = 0; i < n; ++i) {
        DebugSimplex("x%d ", nonbasic[i] + 1);
    }

    DebugSimplex("} B = { ");

    // basic variables
    for (int i = 0; i < m; ++i) {
        DebugSimplex("x%d ", b_labels[i] + 1);
    }

    DebugSimplex("}\n");
};

template <size_t siz> void printBbar(const array<double, siz>& b) {
    DebugSimplex("bbar = ");

    for (int i = 0; i < m; ++i) {
        DebugSimplex("%10.3f ", b[i]);
    }

    DebugSimplex("\n");
}

template <size_t siz, size_t siz2> void printFinalVariables(array<double, siz>& b, array<int, siz>& b_labels, array<int, siz2>& nonbasic) {
    double varValues[m + n];

    for (int row = 0; row < m; ++row) {
        varValues[b_labels[row]] = b[row];
    }

    for (int col = 0; col < n; ++col) {
        varValues[nonbasic[col]] = 0.0;
    }

    printf("Decision variables: ");
    for (int i = 0; i < n; ++i) {
        printf("x%d = %5.3f ", i + 1, varValues[i]);
    }

    printf("\nSlack variables: ");
    for (int i = n; i < m + n; ++i) {
        printf("x%d = %5.3f ", i + 1, varValues[i]);
    }

    printf("\n");
}

void printFamilyOfSolutions(variable b[], int nonbasic[], vector<double> d, double largestCoeff, int enteringLabel, double z) {}

bool mComparator(variable v1, variable v2) { return v1.value > v2.value; }

constexpr auto MAX_N = 5000;
constexpr auto MAX_M = 240;
static_assert(MAX_N % 8 == 0);
static_assert(MAX_M % 8 == 0);

struct LPProblem {
    // maximize c^T x
    // s.t. Ax <= b
    alignas(32) array<double, MAX_N + MAX_M> c;               // n
    alignas(32) array<array<double, MAX_N + MAX_M>, MAX_M> A; // m * n
    alignas(32) array<double, MAX_M> b;
    int n, m; // 変数の数、制約の数
};

int Solve(LPProblem& lp) {
    m = lp.m;
    n = lp.n;

    // スラック変数を含めた目的関数の係数にする
    fill(lp.c.begin() + n, lp.c.begin() + (n + m), 0.0);

    // b のラベル
    static array<int, MAX_M> b_labels;
    iota(b_labels.begin(), b_labels.begin() + m, lp.n);

    // 非基底変数のラベル、長さ n, 範囲 [0, n + m)
    static array<int, MAX_N> nonbasic;
    iota(nonbasic.begin(), nonbasic.end(), 0);

    // A のスラック変数の列を単位行列で初期化
    for (int row = 0; row < m; ++row) {
        fill(&lp.A[row][n], &lp.A[row][n + m], 0.0);
        lp.A[row][n + row] = 1.0;
    }

    // printLPInfo(lp.c, lp.b, lp.A);
    // DebugSimplex("\n\n");
    // printVariables(nonbasic, lp.b, b_labels);
    // printBbar(lp.b);
    // DebugSimplex("\n");

    // b が 0 以上であることを確認
    for (int row = 0; row < m; ++row) {
        assert(lp.b[row] >= 0.0);
    }

    // ここからシンプレックス法

    int counter = 1;    // イテレーション回数
    vector<Eta> pivots; // 過去のピボットを表すイータ行列
    double z = 0.0;     // 目的関数の初期値

    // 改訂シンプレックス法
    while (true) {
        DebugSimplex("Iteration%d\n------------\n", counter);

        // イータ行列を使って y を計算 (yB = c_b を解く)
        static array<double, MAX_M> y; // 長さ m

        // y を c_b で初期化
        for (int row = 0; row < lp.m; ++row) {
            y[row] = lp.c[b_labels[row]];
        }

        // yB = c_b を y について解く
        // y = c_b B^{-1}
        //   = c_b ... E_2^{-1} E_1^{-1}
        for (auto it = pivots.crbegin(); it != pivots.crend(); ++it) {
            const Eta& pivot = *it;
            const int col_to_change = pivot.col;
            double y_original = y[col_to_change] + pivot.values[col_to_change] * y[col_to_change];
            assert((int)pivot.values.size() == lp.m);
            for (int row = 0; row < lp.m; ++row) {
                y_original -= pivot.values[row] * y[row];
            }
            y[col_to_change] = y_original / pivot.values[col_to_change];
        }

        // // 解いた y を出力
        // DebugSimplex("y = ");
        // for (int i = 0; i < lp.m; i++) {
        //     DebugSimplex("%10.3f ", y[i]);
        // }
        // DebugSimplex("\n");

        // 入れる (entering) 列を選ぶ
        // 被約費用 \bar{c_N} = c_N - ya (ただし a は An の列) を求めて、
        // 値が正となる第 s 成分を選ぶ

        static array<variable, MAX_N> cnbars; // \bar{c_N} の成分のうち、値が正であるもの
        auto cnbars_size = 0;

        int entering_label = nonbasic[0];
        double largest_coef = -1.0;

        // // print cnbar
        // DebugSimplex("cnbar: ");

        for (int i = 0; i < n; ++i) {
            const int& var_label = nonbasic[i];
            const double& cni = lp.c[var_label]; // c_N の i 番目
            double yai = 0.0;                    // ya の i 番目
            for (int idx_y = 0; idx_y < m; ++idx_y) {
                yai += y[idx_y] * lp.A[idx_y][var_label];
            }
            const double cnbar = cni - yai;

            // DebugSimplex("x%d %5.3f ", var_label + 1, cnbar);

            if (cnbar > epsilon1) {
                cnbars[cnbars_size] = {var_label, cnbar};
                cnbars_size++;
                if (cnbar > largest_coef) {
                    largest_coef = cnbar;
                    entering_label = var_label;
                }
            }
        }

        // 目的関数の係数の降順にソート
        sort(cnbars.begin(), cnbars.begin() + cnbars_size, mComparator);

        // DebugSimplex("\n");

        // cnbars が空なら entering する候補が無く、最適解が得られた
        if (cnbars_size == 0) {
            printf("\nNo entering var. Optimal value of %5.3f has been reached.\n", z);
            printFinalVariables(lp.b, b_labels, nonbasic);
            return 0;
        } else {
            DebugSimplex("Entering variable is x%d \n", entering_label + 1);
        }

        int entering_variable_index = 0;

        // Bd = a を解いて d を求める
        // d = B^{-1} a
        //   = ... E_2^{-1} E_1^{-1} a
        vector<double> d(m);
        int leaving_label;
        int leaving_row;
        double smallest_t;
        while (true) {
            leaving_label = -1;
            leaving_row = -1;
            smallest_t = -1;
            // if (entering_variable_index > 0) {
            //     DebugSimplex("\n\nRechoosing entering variable since the diagonal element in the eta column is close to zero.\n");
            // }

            if (entering_variable_index < cnbars_size) {
                entering_label = cnbars[entering_variable_index].label;
                // // 選び直したときの表示
                // if (entering_variable_index > 0) {
                //     DebugSimplex("Entering variable is x%d \n", entering_label + 1);
                // }
            } else {
                printf("\nNo entering var. Optimal value of %5.3f has been reached.\n", z);
                printFinalVariables(lp.b, b_labels, nonbasic);
                return 0;
            }

            // d を、既定に追加する列 a で初期化
            for (int row = 0; row < m; ++row) {
                d[row] = lp.A[row][entering_label];
            }

            // イータ行列を順に掛ける
            for (auto it = pivots.cbegin(); it != pivots.cend(); ++it) {
                const Eta& pivot = *it;
                const int& row_to_change = pivot.col;
                const double& d_original = d[row_to_change];
                const auto d_row_to_change_tmp = d_original / pivot.values[row_to_change];
                for (int row = 0; row < (int)d.size(); ++row) {
                    d[row] -= pivot.values[row] * d_row_to_change_tmp;
                }
                d[row_to_change] = d_row_to_change_tmp;
            }

            // print out d (abarj)
            DebugSimplex("d = ");

            for (auto iter = d.cbegin(); iter != d.cend(); ++iter) {
                DebugSimplex("%5.3f ", *iter);
            }

            DebugSimplex("\n");

            // compute t for each b[i].value / d[i]
            // where d[i] > 0
            // choose the corresponding i for the smallest ratio
            // as the leaving variable

            // initialize smallest_t to be the first ratio where
            // the coefficient of the entering variable in that row is negative
            for (int row = 0; row < (int)d.size(); ++row) {
                if (d[row] > 1e-5) {
                    leaving_label = b_labels[row];
                    leaving_row = row;
                    smallest_t = lp.b[row] / d[row];
                }
            }

            // if no ratio is computed, then the LP is unbounded
            if (leaving_label == -1) {
                DebugSimplex("\nThe given LP is unbounded. The family of solutions is:\n");
                // printFamilyOfSolutions(b, nonbasic, d, largestCoeff, enteringLabel, z);
                return 0;
            }

            // there is at least one ratio computed, print out the ratio(s)
            // and choose the row corresponding to the smallest ratio to leave
            DebugSimplex("ratio: ");

            for (int row = 0; row < (int)d.size(); ++row) {
                if (d[row] <= 1e-5) {
                    continue;
                }

                double t_row = lp.b[row] / d[row];

                if (t_row >= 0.0) {
                    DebugSimplex("x%d %5.3f ", b_labels[row] + 1, t_row);
                }

                if (t_row < smallest_t) {
                    leaving_label = b_labels[row];
                    leaving_row = row;
                    smallest_t = t_row;
                }
            }

            // check the diagonal element in the eta column
            // to see if the current choice of entering variable has to be rejected
            if (d[leaving_row] > epsilon2) {
                DebugSimplex("\nLeaving variable is x%d\n", leaving_label + 1);
                break;
            } else {
                entering_variable_index++;
                continue;
            }
        }

        // At this point we have a pair of entering and leaving variables
        // so that the entering variable is positive and the diagonal entry in the eta column
        // of the eta matrix is fairly far from zero

        // set the value of the entering varaible at t
        // modify b (change leaving variable to entering variable, change values of other basic vars)
        lp.b[leaving_row] = smallest_t;
        b_labels[leaving_row] = entering_label;

        const auto tmp = lp.b[leaving_row];
        for (int row = 0; row < lp.m; ++row) {
            lp.b[row] -= d[row] * smallest_t;
        }
        lp.b[leaving_row] = tmp;

        // push a new eta matrix onto the vector
        pivots.push_back({leaving_row, d});

        // print out the eta matrix representing the pivot at this iteration
        DebugSimplex("E%d = column %d: ", counter, leaving_row);

        for (auto iter = d.cbegin(); iter != d.cend(); ++iter) {
            DebugSimplex("%5.3f ", *iter);
        }

        DebugSimplex("\n");

        // ここがおかしい
        for (int i = 0; i < n; i++) {
            if (nonbasic[i] == entering_label) {
                nonbasic[i] = leaving_label;
                break;
            }
        }
        // nonbasic[entering_label] = leavingLabel;

        // // print out nonbasic and basic variable set after the pivot
        // printVariables(nonbasic, b);

        // // print out the new values of the basic variables
        // printBbar(b);

        // // print out the coefficient of the entering variable and the amount the entering variable has been increased
        // DebugSimplex("\nCoefficient of entering variable: %5.3f\nAmount increased for the entering variable is: %5.3f\n", largestCoeff,
        // smallest_t);

        // increase the value of the objective function
        double increasedValue = largest_coef * smallest_t;

        // print out the update to the objective function value
        DebugSimplex("Increased value: %5.3f\n", increasedValue);

        double originalZ = z;

        z += increasedValue;

        DebugSimplex("Value of the objective function changed from %5.3f to %5.3f\n\n\n", originalZ, z);

        counter++;
    }

    return 0;
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
                cin >> lp.A[row][col];
            }
        }
    }

    simplex::Solve(lp);
}