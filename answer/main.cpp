#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
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
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
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
//#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")
//#pragma GCC optimize("O3")
#pragma GCC target("avx,avx2,tune=native")
#pragma GCC optimize("Ofast")
//#pragma GCC optimize("unroll-loops")
#endif // defined(__clang__)
#endif // defined(__GNUC__)

constexpr auto DEBUG_SIMPLEX = true;

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
        cerr << #var << '=' << var << endl;                                                                                                          \
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

// ========================== 借り物 ==========================
// clang-format off
    // 正規分布ライブラリ
    // https://github.com/miloyip/normaldist-benchmark
    /*
    The MIT License (MIT)

    Copyright (c) 2015 Milo Yip

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    */
    // https://github.com/miloyip/normaldist-benchmark/blob/master/src/avx_mathfun.h
    /*
    AVX implementation of sin, cos, sincos, exp and log
    Based on "sse_mathfun.h", by Julien Pommier
    http://gruntthepeon.free.fr/ssemath/
    Copyright (C) 2012 Giovanni Garberoglio
    Interdisciplinary Laboratory for Computational Science (LISC)
    Fondazione Bruno Kessler and University of Trento
    via Sommarive, 18
    I-38123 Trento (Italy)
    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.
    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:
    1. The origin of this software must not be misrepresented; you must not
        claim that you wrote the original software. If you use this software
        in a product, an acknowledgment in the product documentation would be
        appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
        misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.
    (this is the zlib license)
    */

    #define _PS256_CONST(Name, Val) alignas(32) static const float _ps256_##Name[8] = {Val, Val, Val, Val, Val, Val, Val, Val}
    #define _PI32_CONST256(Name, Val) alignas(32) static const int _pi32_256_##Name[8] = {Val, Val, Val, Val, Val, Val, Val, Val}
    #define _PS256_CONST_TYPE(Name, Type, Val) alignas(32) static const Type _ps256_##Name[8] = {Val, Val, Val, Val, Val, Val, Val, Val}

    _PS256_CONST(1, 1.0f);
    _PS256_CONST(0p5, 0.5f);
    _PS256_CONST(minus_cephes_DP1, -0.78515625f);
    _PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
    _PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
    _PS256_CONST(sincof_p0, -1.9515295891E-4f);
    _PS256_CONST(sincof_p1, 8.3321608736E-3f);
    _PS256_CONST(sincof_p2, -1.6666654611E-1f);
    _PS256_CONST(coscof_p0, 2.443315711809948E-005f);
    _PS256_CONST(coscof_p1, -1.388731625493765E-003f);
    _PS256_CONST(coscof_p2, 4.166664568298827E-002f);
    _PS256_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI
    _PS256_CONST(cephes_SQRTHF, 0.707106781186547524f);
    _PS256_CONST(cephes_log_p0, 7.0376836292E-2f);
    _PS256_CONST(cephes_log_p1, -1.1514610310E-1f);
    _PS256_CONST(cephes_log_p2, 1.1676998740E-1f);
    _PS256_CONST(cephes_log_p3, -1.2420140846E-1f);
    _PS256_CONST(cephes_log_p4, +1.4249322787E-1f);
    _PS256_CONST(cephes_log_p5, -1.6668057665E-1f);
    _PS256_CONST(cephes_log_p6, +2.0000714765E-1f);
    _PS256_CONST(cephes_log_p7, -2.4999993993E-1f);
    _PS256_CONST(cephes_log_p8, +3.3333331174E-1f);
    _PS256_CONST(cephes_log_q1, -2.12194440e-4f);
    _PS256_CONST(cephes_log_q2, 0.693359375f);
    _PI32_CONST256(0, 0);
    _PI32_CONST256(1, 1);
    _PI32_CONST256(inv1, ~1);
    _PI32_CONST256(2, 2);
    _PI32_CONST256(4, 4);
    _PI32_CONST256(0x7f, 0x7f);
    /* the smallest non denormalized float number */
    _PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
    _PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
    _PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);
    _PS256_CONST_TYPE(sign_mask, unsigned, 0x80000000u);
    _PS256_CONST_TYPE(inv_sign_mask, unsigned, ~0x80000000u);

    inline void sincos256_ps(__m256 x, __m256* s, __m256* c) {

        __m256 xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
        __m256i imm0, imm2, imm4;

        sign_bit_sin = x;
        /* take the absolute value */
        x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);
        /* extract the sign bit (upper one) */
        sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(__m256*)_ps256_sign_mask);

        /* scale by 4/Pi */
        y = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_FOPI);

        /* store the integer part of y in imm2 */
        imm2 = _mm256_cvttps_epi32(y);

        /* j=(j+1) & (~1) (see the cephes sources) */
        imm2 = _mm256_add_epi32(imm2, *(__m256i*)_pi32_256_1);
        imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_inv1);

        y = _mm256_cvtepi32_ps(imm2);
        imm4 = imm2;

        /* get the swap sign flag for the sine */
        imm0 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_4);
        imm0 = _mm256_slli_epi32(imm0, 29);
        // __m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

        /* get the polynom selection mask for the sine*/
        imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_2);
        imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)_pi32_256_0);
        // __m256 poly_mask = _mm256_castsi256_ps(imm2);

        __m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
        __m256 poly_mask = _mm256_castsi256_ps(imm2);

        /* The magic pass: "Extended precision modular arithmetic"
        x = ((x - y * DP1) - y * DP2) - y * DP3; */
        xmm1 = *(__m256*)_ps256_minus_cephes_DP1;
        xmm2 = *(__m256*)_ps256_minus_cephes_DP2;
        xmm3 = *(__m256*)_ps256_minus_cephes_DP3;
        xmm1 = _mm256_mul_ps(y, xmm1);
        xmm2 = _mm256_mul_ps(y, xmm2);
        xmm3 = _mm256_mul_ps(y, xmm3);
        x = _mm256_add_ps(x, xmm1);
        x = _mm256_add_ps(x, xmm2);
        x = _mm256_add_ps(x, xmm3);

        imm4 = _mm256_sub_epi32(imm4, *(__m256i*)_pi32_256_2);
        imm4 = _mm256_andnot_si256(imm4, *(__m256i*)_pi32_256_4);
        imm4 = _mm256_slli_epi32(imm4, 29);

        __m256 sign_bit_cos = _mm256_castsi256_ps(imm4);

        sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

        /* Evaluate the first polynom  (0 <= x <= Pi/4) */
        __m256 z = _mm256_mul_ps(x, x);

        y = _mm256_mul_ps(y, z);
        y = _mm256_add_ps(y, *(__m256*)_ps256_coscof_p1);
        y = _mm256_mul_ps(y, z);
        y = _mm256_add_ps(y, *(__m256*)_ps256_coscof_p2);
        y = _mm256_mul_ps(y, z);
        y = _mm256_mul_ps(y, z);
        __m256 tmp = _mm256_mul_ps(z, *(__m256*)_ps256_0p5);
        y = _mm256_sub_ps(y, tmp);
        y = _mm256_add_ps(y, *(__m256*)_ps256_1);

        /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

        __m256 y2 = *(__m256*)_ps256_sincof_p0;
        y2 = _mm256_mul_ps(y2, z);
        y2 = _mm256_add_ps(y2, *(__m256*)_ps256_sincof_p1);
        y2 = _mm256_mul_ps(y2, z);
        y2 = _mm256_add_ps(y2, *(__m256*)_ps256_sincof_p2);
        y2 = _mm256_mul_ps(y2, z);
        y2 = _mm256_mul_ps(y2, x);
        y2 = _mm256_add_ps(y2, x);

        /* select the correct result from the two polynoms */
        xmm3 = poly_mask;
        __m256 ysin2 = _mm256_and_ps(xmm3, y2);
        __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
        y2 = _mm256_sub_ps(y2, ysin2);
        y = _mm256_sub_ps(y, ysin1);

        xmm1 = _mm256_add_ps(ysin1, ysin2);
        xmm2 = _mm256_add_ps(y, y2);

        /* update the sign */
        *s = _mm256_xor_ps(xmm1, sign_bit_sin);
        *c = _mm256_xor_ps(xmm2, sign_bit_cos);
    }
    inline __m256 log256_ps(__m256 x) {
        __m256i imm0;
        __m256 one = *(__m256*)_ps256_1;

        // __m256 invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
        __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

        x = _mm256_max_ps(x, *(__m256*)_ps256_min_norm_pos); /* cut off denormalized stuff */

        // can be done with AVX2
        imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

        /* keep only the fractional part */
        x = _mm256_and_ps(x, *(__m256*)_ps256_inv_mant_mask);
        x = _mm256_or_ps(x, *(__m256*)_ps256_0p5);

        // this is again another AVX2 instruction
        imm0 = _mm256_sub_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
        __m256 e = _mm256_cvtepi32_ps(imm0);

        e = _mm256_add_ps(e, one);

        /* part2:
        if( x < SQRTHF ) {
            e -= 1;
            x = x + x - 1.0;
        } else { x = x - 1.0; }
        */
        // __m256 mask = _mm256_cmplt_ps(x, *(__m256*)_ps256_cephes_SQRTHF);
        __m256 mask = _mm256_cmp_ps(x, *(__m256*)_ps256_cephes_SQRTHF, _CMP_LT_OS);
        __m256 tmp = _mm256_and_ps(x, mask);
        x = _mm256_sub_ps(x, one);
        e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
        x = _mm256_add_ps(x, tmp);

        __m256 z = _mm256_mul_ps(x, x);

        __m256 y = *(__m256*)_ps256_cephes_log_p0;
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p1);
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p2);
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p3);
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p4);
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p5);
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p6);
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p7);
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p8);
        y = _mm256_mul_ps(y, x);

        y = _mm256_mul_ps(y, z);

        tmp = _mm256_mul_ps(e, *(__m256*)_ps256_cephes_log_q1);
        y = _mm256_add_ps(y, tmp);

        tmp = _mm256_mul_ps(z, *(__m256*)_ps256_0p5);
        y = _mm256_sub_ps(y, tmp);

        tmp = _mm256_mul_ps(e, *(__m256*)_ps256_cephes_log_q2);
        x = _mm256_add_ps(x, y);
        x = _mm256_add_ps(x, tmp);
        x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
        return x;
    }

    _PS256_CONST_TYPE(lcg_a, uint32_t, 1664525);
    _PS256_CONST_TYPE(lcg_b, uint32_t, 1013904223);
    _PS256_CONST_TYPE(lcg_mask, uint32_t, 0x3F800000);

    class LCG {
    public:
        LCG() : x(_mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8)) {}

        __m256 operator()() {
            x = _mm256_add_epi32(_mm256_mullo_epi32(x, *(__m256i*)_ps256_lcg_a), *(__m256i*)_ps256_lcg_b);
            __m256i u = _mm256_or_si256(_mm256_srli_epi32(x, 9), *(__m256i*)_ps256_lcg_mask);
            __m256 f = _mm256_sub_ps(_mm256_castsi256_ps(u), *(__m256*)_ps256_1);
            return f;
        }

    private:
        __m256i x;
    };

    // https://github.com/miloyip/normaldist-benchmark/blob/master/src/boxmuller_avx.cpp
    static void normaldistf_boxmuller_avx(float* data, size_t count) {
        assert(count % 16 == 0);
        const __m256 twopi = _mm256_set1_ps(2.0f * 3.14159265358979323846f);
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 minustwo = _mm256_set1_ps(-2.0f);

        LCG r;
        for (size_t i = 0; i < count; i += 16) {
            __m256 u1 = _mm256_sub_ps(one, r()); // [0, 1) -> (0, 1]
            __m256 u2 = r();
            __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(minustwo, log256_ps(u1)));
            __m256 theta = _mm256_mul_ps(twopi, u2);
            __m256 sintheta, costheta;
            sincos256_ps(theta, &sintheta, &costheta);
            _mm256_store_ps(&data[i], _mm256_mul_ps(radius, costheta));
            _mm256_store_ps(&data[i + 8], _mm256_mul_ps(radius, sintheta));
        }
    }
// clang-format on
// ========================== ここまで借り物 ==========================

// 乱数
struct Random {
    using ull = unsigned long long;
    unsigned seed;
    inline Random(const unsigned& seed_) : seed(seed_) { ASSERT(seed != 0u, "Seed should not be 0."); }
    const inline unsigned& next() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    }
    // (0.0, 1.0)
    inline double random() { return (double)next() * (1.0 / (double)0x100000000ull); }
    // [0, right)
    inline int randint(const int& right) { return (ull)next() * right >> 32; }
    // [left, right)
    inline int randint(const int& left, const int& right) { return ((ull)next() * (right - left) >> 32) + left; }
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
    template <class Container> Stack& operator=(const Container& rhs) {
        right = rhs.size();
        ASSERT(right <= max_size, "Too big container.");
        for (int i = 0; i < right; i++) {
            data[i] = rhs.data[i];
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

namespace simplex {

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

template <int MAX_N, int MAX_M, int MAX_N_COMPONENTS> struct LPProblem {
    // maximize c^T x
    // s.t. Ax <= b

    static_assert(MAX_N % 8 == 0);
    static_assert(MAX_M % 8 == 0);

    // 問題を設定するには、n, m, c, A_components, b に値を入れる
    enum class Status { NONE, OPTIMAL, INFEASIBLE, UNBOUNDED };
    alignas(64) array<double, MAX_N + MAX_M> c; // n
    // alignas(64) array<array<double, MAX_N + MAX_M>, MAX_M> A; // m * n
    Stack<SparseMatrixComponent, MAX_N_COMPONENTS> A_components; // 係数の非ゼロ要素
    alignas(64) array<double, MAX_M> b;
    alignas(64) array<double, MAX_N + MAX_M> x; // スラック変数を含む解
    int n, m;                                   // 変数の数、制約の数
    double z;                                   // 最適化した目的変数の値
    Status status;
    void PrintSolution(ostream& os = cout) {
        for (int i = 0; i < n; i++) {
            os << x[i] << (i == n - 1 ? "" : " ");
        }
        os << endl;
    }
    void Reset(const int& n_, const int& m_) {
        n = n_;
        m = m_;
        A_components.clear();
        fill(b.begin(), b.begin() + m, 0.0);
        fill(c.begin(), c.begin() + n, 0.0);
    }
};

template <int MAX_N, int MAX_M, int MAX_N_COMPONENTS> void Solve(LPProblem<MAX_N, MAX_M, MAX_N_COMPONENTS>& lp, const int& max_iteration = 2000) {

    constexpr double epsilon1 = 0.00001;
    constexpr double epsilon2 = 0.00000001;
    constexpr auto MAX_PIVOTS_SIZE = 2000;

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
    static auto A_components = Stack<SparseMatrixComponent, MAX_N_COMPONENTS + MAX_M>();
    A_components = lp.A_components;
    for (int row = 0; row < lp.m; ++row) {
        A_components.push({row, lp.n + row, 1.0});
    }

    // A を構築
    static auto A = CSCMatrix<MAX_N + MAX_M, MAX_N_COMPONENTS + MAX_M>();
    new (&A) decltype(A)(lp.n + lp.m, A_components);

    // b が 0 以上であることを確認
    for (int row = 0; row < lp.m; ++row) {
        if (b[row] < 0.0) {
            lp.status = LPProblem<MAX_N, MAX_M, MAX_N_COMPONENTS>::Status::INFEASIBLE;
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
                lp.status = LPProblem<MAX_N, MAX_M, MAX_N_COMPONENTS>::Status::UNBOUNDED;
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
    lp.status = LPProblem<MAX_N, MAX_M, MAX_N_COMPONENTS>::Status::OPTIMAL;
    if constexpr (DEBUG_SIMPLEX) {
        cerr << "[simplex] iteration = " << pivots_size << endl;
    }
}
} // namespace simplex

// ========================== ライブラリここまで ==========================

#ifdef NDEBUG
constexpr auto DEBUG_STATS = false; // ここは固定
#else
constexpr auto DEBUG_STATS = true;
#endif

constexpr auto MAX_N_MINIMIZATION_TASKS = 100;
constexpr static auto EXPECTED_SKILL_EMA_ALPHA = 1e-5;
constexpr auto MCMC_N_SAMPLING = 1000;
constexpr auto QUEUE_UPDATE_FREQUENCY = 40;

namespace input {
constexpr auto N = 1000;             // タスク数
constexpr auto M = 20;               // 人数
auto K = 0;                          // 技能数 [10, 20]
auto R = 0;                          // 依存関係数 [1000, 3000]
auto d = array<array<int, 20>, N>(); // 各タスクの要求技能レベル
auto edges = Stack<Edge, 3000>();
auto G = Graph<1000, 3000>();
} // namespace input

// ========================== common ==========================

namespace common {
struct CompletedTask {
    int task; // タスク番号
    int t;    // かかった時間
};
auto completed_tasks = array<Stack<CompletedTask, 400>, input::M>();
enum class TaskStatus { NotStarted, InQueue, InProgress, Completed };
auto task_status = array<TaskStatus, input::N>();         // タスクの状態。open かどうかは関係ないことに注意
auto member_status = array<int, input::M>();              // -1: 空き, それ以外: 今やってるタスク
auto expected_complete_dates = array<double, input::M>(); // 終了予定時刻
auto starting_times = array<int, input::M>();             // メンバーがタスクを始めた時刻
auto in_dims = array<int, input::N>();                    // 入次数、0 になったら open (自由に実行できる)
auto open_members = Stack<int, input::N>();               // 手の空いたメンバー
auto rng = Random(3141592);                               // 乱数生成器
auto level = array<double, input::N>();                   // 後にどれくらいのタスクがつっかえてるか
auto task_queue = Stack<int, MAX_N_MINIMIZATION_TASKS>(); // 早めにこなしたいタスク
auto next_important_task = array<int, input::N + 1>();    // 次にキューに入れたいタスク (隣接リスト)
auto n_not_open_tasks_in_queue = 0;                       // キューに入っているタスクのうち、open でないものの数
auto day = 1;                                             // 現在の日付
auto n_completed_tasks = 0;                               // 完了したタスク数

struct SchedulingInfo {
    int member;
    double ratio;
};

auto scheduling_info = array<SchedulingInfo, input::N>(); // シンプレックス法の結果

} // namespace common

// ========================== prediction ==========================

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
constexpr auto initial_expected_skill =
    array<double, 11>{10.34388673, 9.84151079, 9.40491307, 9.02080870, 8.67996031, 8.37518411,
                      8.10086558,  7.85121853, 7.62445622, 7.41485945, 7.22255002}; // [技能数 - 10] := スキルの予測値
auto task_weights = array<double, input::N>();                         // タスクの重み: そのタスクに平均的にかかる時間
auto expected_time = array<array<double, input::M>, input::N>();       // 期待所要時間
auto expected_time_naive = array<array<double, input::M>, input::N>(); // 期待所要時間, 愚直計算
auto expected_skill = array<array<double, 20>, input::M>();            // 各メンバーの能力の予測値
// clang-format off
constexpr auto initial_histogram = array<array<double, 61>, 21 - 10>{
    array<double, 61>{0.03199358,0.06368178,0.06305036,0.06192850,0.06043657,0.05849638,0.05626998,0.05378124,0.05096481,0.04803811,0.04487573,0.04173182,0.03857524,0.03535695,0.03231937,0.02933553,0.02657940,0.02393169,0.02151936,0.01931420,0.01728974,0.01544169,0.01377925,0.01225258,0.01088206,0.00960582,0.00847646,0.00745854,0.00652507,0.00568167,0.00493471,0.00427288,0.00366315,0.00314178,0.00266986,0.00225166,0.00188762,0.00157227,0.00130075,0.00105789,0.00086575,0.00069207,0.00054223,0.00043122,0.00032470,0.00024561,0.00018435,0.00013126,0.00009303,0.00006432,0.00004107,0.00002563,0.00001604,0.00000882,0.00000480,0.00000203,0.00000081,0.00000018,0.00000003,0.00000000,0.00000000},
                     {0.03376853,0.06730558,0.06649401,0.06514447,0.06333945,0.06106338,0.05840555,0.05550430,0.05228855,0.04883704,0.04535258,0.04175391,0.03823655,0.03483059,0.03150548,0.02836674,0.02549720,0.02280790,0.02035080,0.01809479,0.01611119,0.01428379,0.01262889,0.01113925,0.00978007,0.00856836,0.00748140,0.00649721,0.00563434,0.00484465,0.00417430,0.00355859,0.00302185,0.00255495,0.00213271,0.00177047,0.00147055,0.00119538,0.00096963,0.00077909,0.00061493,0.00048418,0.00037631,0.00028338,0.00021351,0.00015686,0.00010989,0.00007861,0.00005309,0.00003512,0.00002257,0.00001339,0.00000731,0.00000372,0.00000198,0.00000067,0.00000032,0.00000005,0.00000003,0.00000001,0.00000000},
                     {0.03550163,0.07069985,0.06975901,0.06818148,0.06605969,0.06338745,0.06034109,0.05697538,0.05328152,0.04944696,0.04558107,0.04164214,0.03784412,0.03418324,0.03064046,0.02741236,0.02442631,0.02168933,0.01922626,0.01699257,0.01499427,0.01317532,0.01153803,0.01009785,0.00880508,0.00763689,0.00660473,0.00569153,0.00487017,0.00414601,0.00351975,0.00296470,0.00248343,0.00206292,0.00170522,0.00140036,0.00113917,0.00091457,0.00072737,0.00057604,0.00044395,0.00034435,0.00025742,0.00019642,0.00013958,0.00010069,0.00006859,0.00004748,0.00003112,0.00001967,0.00001201,0.00000670,0.00000365,0.00000173,0.00000096,0.00000024,0.00000007,0.00000003,0.00000000,0.00000000,0.00000000},
                     {0.03718201,0.07396736,0.07281658,0.07102568,0.06857943,0.06555263,0.06206051,0.05822965,0.05408930,0.04987308,0.04560328,0.04138685,0.03728455,0.03342632,0.02978358,0.02642884,0.02339038,0.02065261,0.01815687,0.01593963,0.01394439,0.01218318,0.01059281,0.00918734,0.00791205,0.00680538,0.00583923,0.00497558,0.00421030,0.00354632,0.00298121,0.00247869,0.00205301,0.00168822,0.00136790,0.00110548,0.00088508,0.00069564,0.00055104,0.00042467,0.00032256,0.00024254,0.00017866,0.00013073,0.00009200,0.00006327,0.00004319,0.00002822,0.00001872,0.00001125,0.00000592,0.00000326,0.00000182,0.00000076,0.00000029,0.00000009,0.00000002,0.00000001,0.00000000,0.00000000,0.00000000},
                     {0.03872673,0.07701037,0.07580035,0.07375336,0.07095388,0.06751025,0.06358346,0.05931414,0.05477163,0.05012883,0.04548370,0.04100679,0.03667316,0.03264081,0.02890239,0.02548712,0.02239612,0.01962104,0.01715157,0.01495075,0.01300114,0.01126086,0.00970668,0.00835010,0.00714246,0.00608711,0.00515962,0.00436164,0.00365676,0.00305388,0.00251927,0.00207197,0.00168809,0.00136779,0.00109856,0.00087447,0.00068675,0.00053903,0.00040685,0.00031286,0.00023472,0.00017229,0.00012454,0.00008774,0.00006020,0.00004175,0.00002718,0.00001670,0.00000994,0.00000624,0.00000329,0.00000160,0.00000084,0.00000039,0.00000018,0.00000004,0.00000001,0.00000000,0.00000000,0.00000000,0.00000000},
                     {0.04028545,0.08002691,0.07855541,0.07627479,0.07314183,0.06928536,0.06497279,0.06025511,0.05530280,0.05026639,0.04534181,0.04054774,0.03603184,0.03183653,0.02800787,0.02452781,0.02143495,0.01865768,0.01621021,0.01402524,0.01210552,0.01038964,0.00891036,0.00758423,0.00643425,0.00543379,0.00456671,0.00381725,0.00316559,0.00260599,0.00213695,0.00173619,0.00140253,0.00111851,0.00088272,0.00068707,0.00053649,0.00040996,0.00031140,0.00023327,0.00017175,0.00012031,0.00008527,0.00005969,0.00003970,0.00002635,0.00001669,0.00001033,0.00000588,0.00000361,0.00000189,0.00000093,0.00000041,0.00000023,0.00000003,0.00000001,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000},
                     {0.04172636,0.08285872,0.08124835,0.07864293,0.07515848,0.07093001,0.06619349,0.06106574,0.05568980,0.05026952,0.04506812,0.04005034,0.03535106,0.03106014,0.02712213,0.02365334,0.02049824,0.01777519,0.01532197,0.01317621,0.01126852,0.00962093,0.00817316,0.00690587,0.00580335,0.00486026,0.00404648,0.00334601,0.00274407,0.00224727,0.00180708,0.00145651,0.00115721,0.00091551,0.00071462,0.00055131,0.00042100,0.00031596,0.00023432,0.00017195,0.00012253,0.00008591,0.00006133,0.00004054,0.00002708,0.00001723,0.00001050,0.00000612,0.00000369,0.00000187,0.00000093,0.00000046,0.00000019,0.00000008,0.00000002,0.00000001,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000},
                     {0.04308467,0.08563306,0.08385052,0.08098699,0.07711523,0.07249952,0.06729076,0.06169871,0.05598336,0.05023011,0.04468411,0.03948060,0.03464652,0.03024074,0.02627106,0.02275824,0.01961381,0.01687941,0.01445990,0.01236921,0.01050192,0.00889981,0.00749624,0.00629082,0.00524710,0.00434332,0.00358321,0.00293265,0.00238734,0.00192290,0.00154185,0.00121980,0.00096102,0.00074568,0.00057561,0.00043446,0.00033044,0.00024334,0.00017736,0.00012590,0.00009037,0.00006176,0.00004078,0.00002731,0.00001752,0.00001066,0.00000642,0.00000390,0.00000207,0.00000106,0.00000047,0.00000022,0.00000009,0.00000004,0.00000002,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000},
                     {0.04447691,0.08831202,0.08634204,0.08315169,0.07891502,0.07390661,0.06826452,0.06226774,0.05612381,0.05009072,0.04431979,0.03890952,0.03395179,0.02945254,0.02543306,0.02187852,0.01876287,0.01605197,0.01365285,0.01158376,0.00979567,0.00823798,0.00688727,0.00572966,0.00473496,0.00389515,0.00317935,0.00257304,0.00207139,0.00164837,0.00130780,0.00102467,0.00079353,0.00061091,0.00046399,0.00034572,0.00025541,0.00018451,0.00013374,0.00009382,0.00006687,0.00004363,0.00002941,0.00001880,0.00001096,0.00000708,0.00000390,0.00000230,0.00000139,0.00000051,0.00000031,0.00000012,0.00000003,0.00000001,0.00000001,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000},
                     {0.04584259,0.09090329,0.08873591,0.08524864,0.08066216,0.07518321,0.06910030,0.06270610,0.05622191,0.04990324,0.04385543,0.03829591,0.03320446,0.02864108,0.02461063,0.02104400,0.01794952,0.01526989,0.01292120,0.01088178,0.00913917,0.00760772,0.00633316,0.00521979,0.00428711,0.00349048,0.00281703,0.00226341,0.00180690,0.00142254,0.00111569,0.00085902,0.00066132,0.00050052,0.00037454,0.00027576,0.00020097,0.00014453,0.00010045,0.00007008,0.00004711,0.00003160,0.00002031,0.00001239,0.00000726,0.00000448,0.00000264,0.00000155,0.00000063,0.00000034,0.00000015,0.00000006,0.00000002,0.00000001,0.00000001,0.00000001,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000},
                     {0.04708640,0.09340400,0.09105120,0.08723491,0.08228970,0.07640333,0.06988548,0.06308766,0.05627216,0.04963958,0.04338557,0.03763665,0.03248464,0.02786117,0.02380257,0.02026260,0.01717879,0.01452484,0.01223159,0.01021803,0.00853074,0.00705478,0.00579655,0.00475558,0.00386315,0.00312518,0.00250027,0.00199114,0.00157189,0.00122755,0.00095072,0.00072238,0.00054743,0.00040920,0.00030104,0.00022197,0.00015728,0.00011152,0.00007721,0.00005212,0.00003525,0.00002174,0.00001436,0.00000898,0.00000515,0.00000292,0.00000153,0.00000078,0.00000039,0.00000018,0.00000007,0.00000004,0.00000003,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000},
};
// clang-format on
struct Histogram {
    array<double, 61> data;
    double base;
    inline Histogram() : data(), base(EXPECTED_SKILL_EMA_ALPHA) {}
    inline void SetInitialValue() { data = initial_histogram[input::K - 10]; }
    inline void AddData(const int& idx) {
        // データ追加
        base *= 1.0 / (1.0 - EXPECTED_SKILL_EMA_ALPHA);
        data[idx] += base;
    }
    inline void Normalize() {
        // 割合に直す
        for (auto&& d : data) {
            d *= (EXPECTED_SKILL_EMA_ALPHA / base);
        }
        base = EXPECTED_SKILL_EMA_ALPHA;
    }
};

auto skill_histograms = array<array<Histogram, 20>, input::M>(); // 各メンバー各スキルの事後分布

inline void PrintExpectedSkill(const int& member) {
#ifdef VISUALIZE
    cout << "#s " << member + 1;
    rep(skill, input::K) cout << " " << expected_skill[member][skill];
    cout << endl;
#endif
}

namespace mh {
struct State {
    // 実際のスキル数値 = l2_norm / root_sum_square_skills_base * skills_base
    array<double, 20> skills_base; // パラメータ, 事前分布は正規分布
    // array<double, 20> log_prior_probabilities; // 各パラメータの事前確率の対数 = -skills_base * skills_base / 2
    // double sum_log_prior_probabilities;  // 事前確率の対数  -sum_square_skills_base / 2
    array<array<double, 20>, 400> ramps; // 各タスク各技能のランプ関数をとった値
    array<double, 400> sum_ramps;        // 各タスクの期待完了時間
    double log_likelihood;               // 対数尤度: sum(sum_ramps^2/6)
    double sum_square_skills_base;       // 2 乗和  sum(skill_base^2)
    double root_sum_square_skills_base;  // 2 乗和の平方根 (分母)
    double l2_norm;                      // パラメータ, 事前分布は一様分布
    double log_alpha;                    // 採択率に比例する感じのやつの対数
    int member;                          // メンバー
    // 最終的な α は パラメータ1の事前確率 x パラメータ2の事前確率 x ... x 尤度
    // 尤度 = f(max(1, Ramp(d1-s1) + Ramp(d2-s2) + ... + Ramp(dk-sk)) - 実際にかかった時間) のすべての完了タスクに対する総積
    // ただし f は正規分布 N(0, 6^2 / 12) の確率密度関数 (に比例する値) (本当は一様分布)
    // skill_base を変えたとき、l2_norm も合わせて変えると差分更新が効率的にできる

    // 採択率は α'/α = exp(log(α') - log(α))

    inline State() = default;
    inline State(const int& member_) {
        rep(skill, input::K) {
            skills_base[skill] = sqrt(2.0 / PI); // 0.7978845608
        }
        sum_square_skills_base = (double)(input::K * 2) / PI;
        root_sum_square_skills_base = sqrt(sum_square_skills_base);
        l2_norm = 40.0;
        const auto sum_log_prior_probabilities = sum_square_skills_base * -0.5;
        log_alpha = sum_log_prior_probabilities;
        member = member_;
    }

    inline void AddCompletedTask() {
        // common::completed_tasks を更新した後に呼ぶ
        const auto scale = l2_norm / root_sum_square_skills_base;
        const auto idx_completed_tasks = common::completed_tasks[member].size() - 1;
        const auto& completed_task = common::completed_tasks[member][idx_completed_tasks];
        ASSERT(sum_ramps[idx_completed_tasks] == 0.0, "初期化されてないよ");
        sum_ramps[idx_completed_tasks] = 0.0;
        rep(skill, input::K) {
            const auto s = abs(skills_base[skill]) * scale;
            ramps[idx_completed_tasks][skill] = max(0.0, input::d[completed_task.task][skill] - s);
            sum_ramps[idx_completed_tasks] += ramps[idx_completed_tasks][skill];
        }
        const auto w = max(1.0, sum_ramps[idx_completed_tasks]) - completed_task.t;
        log_likelihood += w * w * (-1.0 / (2.0 * 6.0 * 6.0 / 12.0));
        const auto sum_log_prior_probabilities = sum_square_skills_base * -0.5;
        log_alpha = sum_log_prior_probabilities + log_likelihood;
    }

    inline void Update() {
        using common::rng;
        const auto skill = rng.randint(input::K + 1);
        if (skill == input::K) {
            // 1. l2_norm だけ変更, 事前確率は変化しない
            constexpr static auto MCMC_Q_L2_NORM_RANGE = 10.0;
            const auto delta = (rng.random() - 0.5) * MCMC_Q_L2_NORM_RANGE;
            const auto tmp = l2_norm + delta;
            const auto new_l2_norm = tmp > 60.0 ? 120.0 - tmp : tmp < 20.0 ? 40.0 - tmp : tmp;
            const auto new_scale = new_l2_norm / root_sum_square_skills_base;
            static auto new_ramps = array<array<double, 20>, 400>();
            static auto new_sum_ramps = array<double, 400>();
            auto new_log_likelihood = 0.0;
            rep(idx_completed_tasks, common::completed_tasks[member].size()) {
                const auto& completed_task = common::completed_tasks[member][idx_completed_tasks];
                new_sum_ramps[idx_completed_tasks] = 0.0;
                rep(skl, input::K) {
                    const auto s = abs(skills_base[skl]) * new_scale;
                    new_ramps[idx_completed_tasks][skl] = max(0.0, input::d[completed_task.task][skl] - s);
                    new_sum_ramps[idx_completed_tasks] += new_ramps[idx_completed_tasks][skl];
                }
                const auto w = max(1.0, new_sum_ramps[idx_completed_tasks]) - completed_task.t;
                new_log_likelihood += w * w;
            }
            new_log_likelihood *= -1.0 / (2.0 * 6.0 * 6.0 / 12.0);
            const auto sum_log_prior_probabilities = sum_square_skills_base * -0.5; // これは変わらない
            const auto new_log_alpha = sum_log_prior_probabilities + new_log_likelihood;
            const auto p = exp(new_log_alpha - log_alpha); // 採択率
            const auto r = rng.random();
            // cerr << "1. 事前確率一定 " << p << " " << log_alpha << " " << new_log_alpha << endl;
            // CHECK(new_scale);
            // CHECK(sum_log_prior_probabilities);
            // CHECK(new_log_likelihood);
            // CHECK(new_sum_ramps[0]);
            if (r < p) {
                // 採択
                rep(idx_completed_tasks, common::completed_tasks[member].size()) {
                    rep(skl, input::K) { ramps[idx_completed_tasks][skl] = new_ramps[idx_completed_tasks][skl]; }
                    sum_ramps[idx_completed_tasks] = new_sum_ramps[idx_completed_tasks];
                }
                log_likelihood = new_log_likelihood;
                l2_norm = new_l2_norm;
                log_alpha = new_log_alpha;
            } else {
                // 棄却
                // 何もしない
            }
        } else {
            // 2. スケール一定
            // 計算量は O(メンバーがこなしたタスク数)
            const auto scale = l2_norm / root_sum_square_skills_base;
            double new_skill_base, delta_sum_square_skills_base, new_sum_square_skills_base, new_root_sum_square_skills_base, new_l2_norm;
            do {
                constexpr static auto MCMC_Q_RANGE = 2.0;
                const auto delta = (rng.random() - 0.5) * MCMC_Q_RANGE;
                new_skill_base = skills_base[skill] + delta;
                delta_sum_square_skills_base = new_skill_base * new_skill_base - skills_base[skill] * skills_base[skill];
                new_sum_square_skills_base = sum_square_skills_base + delta_sum_square_skills_base;
                new_root_sum_square_skills_base = sqrt(new_sum_square_skills_base);
                new_l2_norm = scale * new_root_sum_square_skills_base;
            } while (new_l2_norm < 20.0 || new_l2_norm > 60.0);
            const auto s = abs(new_skill_base) * scale;
            static auto new_ramps = array<double, 400>();     // common::completed_tasks[member].size() まで使う
            static auto new_sum_ramps = array<double, 400>(); // common::completed_tasks[member].size() まで使う
            auto new_log_likelihood = 0.0;
            // auto log_f = [](const double& x) { return (x * x) * (-1.0 / (2.0 * 6.0 * 6.0 / 12.0)); }; // -x^2 / 6
            rep(idx_completed_tasks, common::completed_tasks[member].size()) {
                const auto completed_task = common::completed_tasks[member][idx_completed_tasks];
                const auto ramp = max(0.0, input::d[completed_task.task][skill] - s);
                new_ramps[idx_completed_tasks] = ramp;
                new_sum_ramps[idx_completed_tasks] = sum_ramps[idx_completed_tasks] + ramp - ramps[idx_completed_tasks][skill];
                const auto w = max(1.0, new_sum_ramps[idx_completed_tasks]) - completed_task.t;
                new_log_likelihood += w * w;
            }
            new_log_likelihood *= -1.0 / (2.0 * 6.0 * 6.0 / 12.0);
            const auto new_sum_log_prior_probabilities = new_sum_square_skills_base * -0.5;
            const auto new_log_alpha = new_sum_log_prior_probabilities + new_log_likelihood;
            const auto p = exp(new_log_alpha - log_alpha); // 採択率
            // cerr << "2. スケール一定 " << p << " " << log_alpha << " " << new_log_alpha << endl;
            // CHECK(scale);
            // CHECK(new_sum_log_prior_probabilities);
            // CHECK(new_log_likelihood);
            const auto r = rng.random();
            if (r < p) {
                // 採択
                skills_base[skill] = new_skill_base;
                rep(idx_completed_tasks, common::completed_tasks[member].size()) {
                    ramps[idx_completed_tasks][skill] = new_ramps[idx_completed_tasks];
                    sum_ramps[idx_completed_tasks] = new_sum_ramps[idx_completed_tasks];
                }
                log_likelihood = new_log_likelihood;
                sum_square_skills_base = new_sum_square_skills_base;
                root_sum_square_skills_base = new_root_sum_square_skills_base;
                l2_norm = new_l2_norm;
                log_alpha = new_log_alpha;
                // auto tmp_sum = 0.0;
                // rep(skill, input::K) { tmp_sum += ramps[0][skill]; }
                // CHECK(sum_ramps[0]);
                // CHECK(tmp_sum);
            } else {
                // 棄却
                // 何もしない
            }
        }
    }
};
auto state = array<State, input::M>(); // [メンバー] := 現在のサンプル
} // namespace mh

void Initialize() {
    ASSERT(input::K != 0, "input がまだだよ");
    initial_expected_time = initial_expected_time_all[input::K - 10];
    // タスクに対する予測の初期化
    rep(task, input::N) {
        rep(skill, input::K) { task_weights[task] += initial_expected_time[input::d[task][skill]]; }
        rep(member, input::M) { expected_time[task][member] = task_weights[task]; }
    }
    // メンバーに対する予測の初期化
    rep(member, input::M) {
        rep(skill, input::K) {
            expected_skill[member][skill] = initial_expected_skill[input::K - 10];
            skill_histograms[member][skill].SetInitialValue();
        }
        // TODO: expected_squared_skill の初期化
        new (&mh::state[member]) remove_reference<decltype(mh::state[member])>::type(member);
        PrintExpectedSkill(member);
    }
}

inline void Update(const int& member) {
    // メンバーのスキルを予測
    auto& state = mh::state[member];
    rep(iteration, MCMC_N_SAMPLING) {
        state.Update();
        rep(skill, input::K) {
            const auto sampled_skill_value = min(60.0, abs(state.skills_base[skill]) * (state.l2_norm / state.root_sum_square_skills_base));
            expected_skill[member][skill] *= 1.0 - EXPECTED_SKILL_EMA_ALPHA;
            expected_skill[member][skill] += EXPECTED_SKILL_EMA_ALPHA * sampled_skill_value;
            skill_histograms[member][skill].AddData((int)round(sampled_skill_value));
        }
    }
}
} // namespace prediction

// ========================== main loop ==========================

inline void UpdateQueue() {
    // task_queue を更新する

    // 1. queue に task を追加する
    {
        int task = common::next_important_task[input::N];
        int last_task = input::N;
        while (common::task_queue.size() < MAX_N_MINIMIZATION_TASKS && task != input::N) {
            if (common::in_dims[task] != 0 && common::n_not_open_tasks_in_queue > 60) { // open でないタスクをキューに入れるのは 60 個とかに抑える
                last_task = task;
                task = common::next_important_task[task];
                continue;
            }
            common::task_queue.push(task);
            common::task_status[task] = common::TaskStatus::InQueue;
            if (common::in_dims[task] != 0)
                common::n_not_open_tasks_in_queue++;
            task = common::next_important_task[task];
            common::next_important_task[last_task] = task;
        }
    }

    // 2. queue 内の各タスクについて、かかる時間の予測を行う
    {
        static auto f = array<array<array<double, 41>, 20>, input::M>(); // [メンバー][スキル種別][要求スキル値] := そのスキルでかかる時間
        rep(member, input::M) {
            rep(skill, input::K) {
                // 2 階累積和で能力のヒストグラムから各要求スキル値でかかる時間へ変換する
                auto& hist = prediction::skill_histograms[member][skill];
                auto& fms = f[member][skill];
                hist.Normalize();

                if constexpr (DEBUG_STATS) {
                    // cout << "# skill_histograms[" << member << "][" << skill << "]=";
                    // auto sum_hist = 0.0;
                    // for (const auto& h : hist.data) {
                    //     cout << h << " ";
                    //     sum_hist += h;
                    // }
                    // cout << endl;
                    // cout << "# sum_hist=" << sum_hist << endl;
                }

                auto df = 0.0;
                rep1(i, 40) { // fms[0] は 0.0
                    df += hist.data[i - 1];
                    fms[i] = fms[i - 1] + df;
                }
            }
        }
        for (const auto& task : common::task_queue) {
            rep(member, input::M) {
                prediction::expected_time[task][member] = 0.0;
                prediction::expected_time_naive[task][member] = 0.0;
                rep(skill, input::K) {
                    // 作った表を引く
                    prediction::expected_time[task][member] += f[member][skill][input::d[task][skill]];
                    prediction::expected_time_naive[task][member] += max(0.0, input::d[task][skill] - prediction::expected_skill[member][skill]);
                }
                prediction::expected_time[task][member] =
                    max(1.0, prediction::expected_time[task][member]); // これはまあ正確ではない (そんなこと言ったら色んな場所が正確でゎない…)
                prediction::expected_time_naive[task][member] = max(1.0, prediction::expected_time_naive[task][member]);
                if constexpr (DEBUG_STATS) {
                    // cout << "# expected_time[" << task << "][" << member << "]=" << prediction::expected_time[task][member]
                    //      << "(naive:" << prediction::expected_time_naive[task][member] << ")" << endl;
                }
            }
        }
    }

    // 3. task の割当を行う
    {
        constexpr auto MAX_N = (MAX_N_MINIMIZATION_TASKS * input::M + 2 - 1) / 8 * 8 + 8;
        constexpr auto MAX_M = (MAX_N_MINIMIZATION_TASKS + input::M + 1 - 1) / 8 * 8 + 8;
        constexpr auto MAX_N_COMPONENTS = MAX_N_MINIMIZATION_TASKS * input::M * 2 + MAX_N_MINIMIZATION_TASKS + input::M + 1;
        const auto n_minimization_tasks = common::task_queue.size(); // 最適化するタスクの数

        static auto lp = simplex::LPProblem<MAX_N, MAX_M, MAX_N_COMPONENTS>();
        lp.Reset(n_minimization_tasks * input::M + 2, n_minimization_tasks + input::M + 1); // 変数の数、制約の数

        // 制約の設定
        const auto objective_variable = n_minimization_tasks * input::M;
        const auto one_variable = objective_variable + 1;
        const auto objective_offset = 2000.0; // この値から目的変数を引くと終了見込み日になる
        auto GetVariable = [&](const int& member, const int& idx_task_queue) { return n_minimization_tasks * member + idx_task_queue; };
        rep(member, input::M) {
            rep(idx_task_queue, n_minimization_tasks) {
                const auto& task = common::task_queue[idx_task_queue];
                // (1) 目的変数の値は、各メンバーの合計時間より大きい
                lp.A_components.push({member, GetVariable(member, idx_task_queue), prediction::expected_time[task][member]});
                // (2) 各タスクの割当合計は 1 以上
                lp.A_components.push({input::M + idx_task_queue, GetVariable(member, idx_task_queue), -1.0});
            }
            // (1)
            lp.A_components.push({member, objective_variable, 1.0});
            lp.b[member] =
                objective_offset -
                (common::member_status[member] == -1
                     ? (double)common::day
                     : max(common::day + 1.0, common::expected_complete_dates
                                                  [member])); // これも本当は現在までタスクが終わってないことを利用して事後分布から期待値が計算できる…
        }
        rep(idx_task_queue, n_minimization_tasks) {
            // (2)
            lp.A_components.push({input::M + idx_task_queue, one_variable, 1.0});
        }
        // (3) 定数 1
        lp.A_components.push({input::M + n_minimization_tasks, one_variable, 1.0});
        lp.b[input::M + n_minimization_tasks] = 1.0;

        // 目的関数の設定
        lp.c[objective_variable] = 1.0;
        lp.c[one_variable] = 10000.0;

        // 解く
        simplex::Solve(lp);

        // lp.PrintSolution(cerr);

        // 結果を取り出して scheduling_info に格納
        rep(idx_task_queue, n_minimization_tasks) {
            const auto& task = common::task_queue[idx_task_queue];

            auto sum_row = 0.0;
            auto best_member = 0;
            auto best_value = 0.0;
            rep(member, input::M) {
                const auto& x = lp.x[GetVariable(member, idx_task_queue)];
                sum_row += x;
                if (chmax(best_value, x))
                    best_member = member;
            }
            if (sum_row == 0.0) {
                // そんなことはないはずだが…
                ASSERT(false, "??????");
                sum_row = 1.0;
            }
            best_value /= sum_row; // 誤差対策で標準化
            common::scheduling_info[task] = {best_member, best_value};
        }
        if constexpr (DEBUG_STATS) {
            cout << "# lp.b:";
            rep(member, input::M) { cout << " " << lp.b[member]; }
            cout << endl;
            // cout << "# lp status: " << (int)lp.status << endl;
            // cout << "# lp.z: " << lp.z << endl;
            // cout << "# optimal value: " << objective_offset - lp.x[objective_variable] << endl;
            // cout << "# one: " << lp.x[one_variable] << endl;
            cout << "# result:" << endl;
            rep(member, input::M) {
                cout << "#  ";
                double sum = 0.0;
                rep(idx_task_queue, n_minimization_tasks) {
                    const auto& task = common::task_queue[idx_task_queue];
                    sum += prediction::expected_time[task][member] * lp.x[GetVariable(member, idx_task_queue)];
                    printf(" %3d(%3d) %3.1f", (int)prediction::expected_time[task][member], (int)prediction::expected_time_naive[task][member],
                           lp.x[GetVariable(member, idx_task_queue)]);
                }
                printf(" = %6.1f", sum);
                cout << endl;
            }

            // cout << "# scheduling_info (task,member,ratio): ";
            // rep(idx_task_queue, n_minimization_tasks) {
            //     const auto& task = common::task_queue[idx_task_queue];
            //     const auto& info = common::scheduling_info[task];
            //     cout << "(" << task << "," << info.member << "," << info.ratio << "),";
            // }
            // cout << endl;
        }
    }
}

inline void SolveLoop() {
    using namespace common;
    fill(member_status.begin(), member_status.end(), -1);

    rep(member, input::M) { open_members.push(member); }

    // 1 日目
    {
        auto chosen_task_idxs = Stack<int, input::M>();
        rep(i, task_queue.size()) {
            if (in_dims[task_queue[i]] != 0)
                continue;
            chosen_task_idxs.push(i);
            if (chosen_task_idxs.size() == input::M)
                break;
        }
        const auto m = chosen_task_idxs.size();
        cout << m;
        for (const auto& idx : chosen_task_idxs) {
            // 着手 ... open_tasks から pop, open_members から pop, member_status, task_status の更新
            const auto task = task_queue[idx];
            // swap(open_tasks[0], open_tasks.back());
            // open_tasks.pop();

            const auto member = open_members.back();
            open_members.pop();

            member_status[member] = task;
            starting_times[member] = 1;
            expected_complete_dates[member] = starting_times[member] + prediction::expected_time[task][member];
            task_status[task] = TaskStatus::InProgress;
            task_queue[idx] = -1;
            cout << " " << member + 1 << " " << task + 1;
        }
        cout << endl;

        int right = 0;
        rep(i, task_queue.size() - m) {
            while (task_queue[right] == -1) {
                right++;
            }
            task_queue[i] = task_queue[right];
            right++;
        }

        ASSERT(right == task_queue.size(), "算数がおかしい");
        task_queue.resize(task_queue.size() - m);
    }

    // 2 日目以降
    struct Output {
        int member, task;
    };
    for (day = 2;; day++) {
        int n;
        cin >> n;
        cout << "# day=" << day << " n=" << n << endl;
        if (n == -1) {
            return;
        } else if (n == 0) {
            cout << 0 << endl;
            continue;
        };

        auto queue_update_flag = false;
        rep(i, n) {
            // 完了 ... member_status, task_status in_dims の更新, open_tasks, open_members の追加
            int member;
            cin >> member;
            member--;
            auto task = member_status[member];
            member_status[member] = -1;
            task_status[task] = TaskStatus::Completed;
            completed_tasks[member].push({task, day - starting_times[member]});
            prediction::mh::state[member].AddCompletedTask();
            for (const auto& u : input::G[task]) {
                in_dims[u]--;
                if (in_dims[u] == 0) {
                    if (task_status[u] == TaskStatus::InQueue) {
                        n_not_open_tasks_in_queue--;
                    }
                }
            }
            open_members.push(member);
            prediction::Update(member);
            prediction::PrintExpectedSkill(member);
            n_completed_tasks++;
            if (n_completed_tasks % QUEUE_UPDATE_FREQUENCY == 0) {
                queue_update_flag = true;
            }
        }
        cout << "# n_completed_tasks=" << n_completed_tasks << endl;

        // タスクキューの更新
        if (queue_update_flag)
            UpdateQueue();

        // 着手
        struct TaskMember {
            int task, member;
        };
        static auto chosen = Stack<TaskMember, 20>();
        chosen.clear();
        for (const auto& member : open_members) {
            auto best_task = -1;
            auto best_task_priority = 0.0;
            for (const auto& task : task_queue) {
                if (in_dims[task] != 0)
                    continue;
                if (best_task == -1) {
                    best_task = task;
                    continue;
                }
                const auto& info = scheduling_info[task];
                auto priority = info.member == member ? info.ratio : 0.0;
                priority *= 1.0 + max(0.0, day - 900 + level[task]) * 0.02;
                if (best_task_priority != priority) {
                    if (best_task_priority < priority) {
                        best_task_priority = priority;
                        best_task = task;
                    }
                } else if (level[task] != level[best_task]) {
                    if (level[best_task] < level[task]) {
                        best_task = task;
                    }
                } else {
                    if (prediction::task_weights[best_task] < prediction::task_weights[task]) {
                        best_task = task;
                    }
                }
            }
            if (best_task != -1) {
                chosen.push({best_task, member});
                task_queue.remove(best_task); // メンバーは後で取り除く
            }
        }
        int m = chosen.size();
        cout << m;
        for (const auto& task_member : chosen) {
            // 着手 ... open_tasks から pop, open_members から pop, member_status, task_status の更新
            const auto& task = task_member.task;
            const auto member = task_member.member;
            open_members.remove(member);

            member_status[member] = task;
            starting_times[member] = day;
            expected_complete_dates[member] = starting_times[member] + prediction::expected_time[task][member];
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

        // 予測関連初期化
        prediction::Initialize();

        // タスクの優先度を設定
        sort(input::edges.begin(), input::edges.end());
        for (int i = input::R - 1; i >= 0; i--) {
            const auto& edge = input::edges[i];
            chmax(common::level[edge.from], common::level[edge.to] + prediction::task_weights[edge.to]);
        }
        if constexpr (DEBUG_STATS) {
            cout << "# level:";
            for (const auto& l : common::level)
                cout << " " << l;
            cout << endl;
        }

        // 入次数の初期化
        for (const auto& e : input::edges) {
            common::in_dims[e.to]++;
        }

        // next_important_task と task_queue の初期化
        {
            static auto order = array<int, input::N>();
            iota(order.begin(), order.end(), 0);
            sort(order.begin(), order.end(), [&](const int& l, const int& r) {
                if (common::level[l] != common::level[r])
                    return common::level[l] > common::level[r];
                return l < r;
            });
            common::next_important_task[input::N] = order[0];
            rep(i, input::N - 1) { common::next_important_task[order[i]] = order[i + 1]; }
            common::next_important_task[order[input::N - 1]] = input::N;

            int task = common::next_important_task[input::N];
            int last_task = input::N;
            while (common::task_queue.size() < MAX_N_MINIMIZATION_TASKS && task != input::N) {
                if (common::in_dims[task] != 0 && common::n_not_open_tasks_in_queue > 60) { // open でないタスクをキューに入れるのは 60 個とかに抑える
                    last_task = task;
                    task = common::next_important_task[task];
                    continue;
                }
                common::task_queue.push(task);
                common::task_status[task] = common::TaskStatus::InQueue;
                if (common::in_dims[task] != 0)
                    common::n_not_open_tasks_in_queue++;
                task = common::next_important_task[task];
                common::next_important_task[last_task] = task;
            }
        }
    }

    SolveLoop();
}

int main() {
    // std::this_thread::sleep_for(std::chrono::seconds(10));
    Solve();
    // TODO
}

#ifdef __clang__
#pragma clang attribute pop
#endif