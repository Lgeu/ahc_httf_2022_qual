// from https://github.com/pakwah/Revised-Simplex-Method

//
//  main.cpp
//  RevisedSimplex
//
//  Created by Xuan Baihua on 10/18/15.
//  Copyright (c) 2015 Xuan Baihua. All rights reserved.
//

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

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

constexpr auto DEBUG_SIMPLEX = true;
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

    // Print out initial input values
    printLPInfo(lp.c, lp.b, lp.A);

    DebugSimplex("\n\n");

    // Print out initial nonbasic and basic variables
    printVariables(nonbasic, lp.b, b_labels);

    // Print out initial values of basic variables
    printBbar(lp.b);

    DebugSimplex("\n");

    // Check initial feasibility
    for (int row = 0; row < m; ++row) {
        if (lp.b[row] < 0.0) {
            DebugSimplex("The given linear program is infeasible, exiting the program.\n");
            return 0;
        }
    }

    // Initial basic solution is feasible, now proceed with the Simplex Method

    // A counter to remember the current iteration number
    int counter = 1;

    // An array of eta matrices representing previous pivots
    vector<Eta> pivots{};

    // Initial value of objective function
    double z = 0.0;

    // Revised Simplex Method
    while (true) {
        DebugSimplex("Iteration%d\n------------\n", counter);

        // compute y using eta matrices (yB = Cb)
        vector<double> y(m);

        // initialize y to be Cb
        for (int row = 0; row < m; ++row) {
            y[row] = lp.c[b_labels[row]];
        }

        // solving y in yB = Cb
        for (auto rIter = pivots.crbegin(); rIter != pivots.crend(); ++rIter) {
            Eta pivot = *rIter;
            int colToChange = pivot.col;
            double yOriginal = y[colToChange];

            for (int row = 0; row < (int)pivot.values.size(); ++row) {
                if (row != colToChange) {
                    yOriginal -= pivot.values[row] * y[row];
                }
            }

            double yNew = yOriginal / pivot.values[colToChange];
            y[colToChange] = yNew;
        }

        // print out solved y
        DebugSimplex("y = ");

        for (auto iter = y.cbegin(); iter != y.cend(); ++iter) {
            DebugSimplex("%10.3f ", *iter);
        }

        DebugSimplex("\n");

        // 入れる (entering) 列を選ぶ
        // 条件 Cn > ya, ただし "a" は An の列

        // 今回のイテレーションで目的関数の係数が正である変数を追跡するための
        vector<variable> cnbars;

        int enteringLabel = nonbasic[0];
        double largestCoeff = -1.0;

        // print cnbar
        DebugSimplex("cnbar: ");

        for (int i = 0; i < n; ++i) {
            int varLabel = nonbasic[i];
            double cni = lp.c[varLabel];
            double yai = 0.0;

            for (int yIndex = 0; yIndex < m; ++yIndex) {
                yai += y[yIndex] * lp.A[yIndex][varLabel];
            }

            double cnbar = cni - yai;

            DebugSimplex("x%d %5.3f ", varLabel + 1, cnbar);

            if (cnbar > epsilon1) {
                variable v = {varLabel, cnbar};

                cnbars.push_back(v);

                if (cnbar > largestCoeff) {
                    largestCoeff = cnbar;
                    enteringLabel = varLabel;
                }
            }
        }

        // sort the variables into descending order
        // based on their coefficients in the objective function
        sort(cnbars.begin(), cnbars.end(), mComparator);

        DebugSimplex("\n");

        // If the vector cnbars is empty, then there are no candidates for the entering variable

        if (cnbars.size() == 0) {
            printf("\nNo entering var. Optimal value of %5.3f has been reached.\n", z);
            printFinalVariables(lp.b, b_labels, nonbasic);
            return 0;
        } else {
            DebugSimplex("Entering variable is x%d \n", enteringLabel + 1);
        }

        int enteringVariable_index = 0;

        // compute the column d in Anbar
        // for the entering variable
        // using eta matrices (Bd = a)
        vector<double> d(m);

        int leavingLabel;
        int leavingRow;
        double smallest_t;

        while (true) {

            leavingLabel = -1;
            leavingRow = -1;
            smallest_t = -1;

            if (enteringVariable_index > 0) {
                DebugSimplex("\n\nRechoosing entering variable since the diagonal element in the eta column is close to zero.\n");
            }

            if (enteringVariable_index < (int)cnbars.size()) {
                enteringLabel = cnbars[enteringVariable_index].label;

                if (enteringVariable_index > 0) {
                    DebugSimplex("Entering variable is x%d \n", enteringLabel + 1);
                }
            } else {
                printf("\nNo entering var. Optimal value of %5.3f has been reached.\n", z);
                printFinalVariables(lp.b, b_labels, nonbasic);
                return 0;
            }

            // initialize d to be the entering column a
            for (int row = 0; row < m; ++row) {
                d[row] = lp.A[row][enteringLabel];
            }

            // Go through eta matrices from pivot 1 to pivot k
            for (auto iter = pivots.cbegin(); iter != pivots.cend(); ++iter) {
                Eta pivot = *iter;
                int rowToChange = pivot.col;
                double dOriginal = d[rowToChange];

                d[rowToChange] = dOriginal / pivot.values[rowToChange];

                for (int row = 0; row < (int)d.size(); ++row) {
                    if (row != rowToChange) {
                        d[row] = d[row] - pivot.values[row] * d[rowToChange];
                    }
                }
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
                    leavingLabel = b_labels[row];
                    leavingRow = row;
                    smallest_t = lp.b[row] / d[row];
                }
            }

            // if no ratio is computed, then the LP is unbounded
            if (leavingLabel == -1) {
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
                    leavingLabel = b_labels[row];
                    leavingRow = row;
                    smallest_t = t_row;
                }
            }

            // check the diagonal element in the eta column
            // to see if the current choice of entering variable has to be rejected
            if (d[leavingRow] > epsilon2) {
                DebugSimplex("\nLeaving variable is x%d\n", leavingLabel + 1);
                break;
            } else {
                enteringVariable_index++;
                continue;
            }
        }

        // At this point we have a pair of entering and leaving variables
        // so that the entering variable is positive and the diagonal entry in the eta column
        // of the eta matrix is fairly far from zero

        // set the value of the entering varaible at t
        // modify b (change leaving variable to entering variable, change values of other basic vars)
        lp.b[leavingRow] = smallest_t;
        b_labels[leavingRow] = enteringLabel;

        const auto tmp = lp.b[leavingRow];
        for (int row = 0; row < lp.m; ++row) {
            lp.b[row] -= d[row] * smallest_t;
        }
        lp.b[leavingRow] = tmp;

        // push a new eta matrix onto the vector
        Eta pivot = {leavingRow, d};
        pivots.push_back(pivot);

        // print out the eta matrix representing the pivot at this iteration
        DebugSimplex("E%d = column %d: ", counter, leavingRow);

        for (auto iter = d.cbegin(); iter != d.cend(); ++iter) {
            DebugSimplex("%5.3f ", *iter);
        }

        DebugSimplex("\n");

        // ここがおかしい
        for (auto&& nb : nonbasic) {
            if (nb == enteringLabel) {
                nb = leavingLabel;
                break;
            }
        }
        // nonbasic[enteringLabel] = leavingLabel;

        // // print out nonbasic and basic variable set after the pivot
        // printVariables(nonbasic, b);

        // // print out the new values of the basic variables
        // printBbar(b);

        // // print out the coefficient of the entering variable and the amount the entering variable has been increased
        // DebugSimplex("\nCoefficient of entering variable: %5.3f\nAmount increased for the entering variable is: %5.3f\n", largestCoeff,
        // smallest_t);

        // increase the value of the objective function
        double increasedValue = largestCoeff * smallest_t;

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