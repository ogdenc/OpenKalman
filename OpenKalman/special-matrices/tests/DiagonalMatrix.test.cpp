/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests relating to Eigen3::DiagonalMatrix.
 */

#include "special-matrices.gtest.hpp"

#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

namespace
{
  using cdouble = std::complex<double>;

  using M51 = eigen_matrix_t<double, 5, 1>;
  using M50 = eigen_matrix_t<double, 5, dynamic_extent>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M30 = eigen_matrix_t<double, 3, dynamic_extent>;
  using M03 = eigen_matrix_t<double, dynamic_extent, 3>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M20 = eigen_matrix_t<double, 2, dynamic_extent>;
  using M02 = eigen_matrix_t<double, dynamic_extent, 2>;
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M01 = eigen_matrix_t<double, dynamic_extent, 1>;
  using M00 = eigen_matrix_t<double, dynamic_extent, dynamic_extent>;

  M51 m51 {make_native_matrix<double, 5, 1>(1, 2, 3, 4, 5)};
  M31 m31 {make_native_matrix<double, 3, 1>(1, 2, 3)};
  M01 m01_3 {m31};
  M00 m00_31 {m31};
  M33 m33 {make_native_matrix<double, 3, 3>(1, 0, 0, 0, 2, 0, 0, 0, 3)};
  M33 m33b = make_native_matrix<double, 3, 3>(4, 0, 0, 0, 5, 0, 0, 0, 6);
  M03 m03_3 {m33};
  M00 m00_33 {m33};
  M21 m21 {make_native_matrix<double, 2, 1>(1, 2)};
  M21 m01_2 {m21};

  using D5 = DiagonalMatrix<M51>;
  using D3 = DiagonalMatrix<M31>;
  using D2 = DiagonalMatrix<M21>;
  using D0 = DiagonalMatrix<M01>;

  D2 d2 {m21};
  D0 d0_2 {m21};
  D3 d3 {m31};
  D0 d0_3 {m31};
  D5 d5 {m51};
  D0 d0_5 {m51};
  DiagonalMatrix<const M51> d5_const {m51};

  template<typename Mat> using D = DiagonalMatrix<Mat>;
}


TEST(eigen3, Diagonal_static_checks)
{
  static_assert(writable<D<M31>>);
  static_assert(writable<D<M31&>>);
  static_assert(not writable<D<const M31>>);
  static_assert(not writable<D<const M31&>>);
  
  static_assert(modifiable<D<M31>, ZeroMatrix<double, 3, 3>>);
  static_assert(modifiable<D<M31>, IdentityMatrix<M22>>);
  static_assert(not modifiable<D<M31>, M31>);
  static_assert(modifiable<D<M31>, D<M31>>);
  static_assert(modifiable<D<M31>, const D<M31>>);
  static_assert(modifiable<D<M31>, D<const M31>>);
  static_assert(not modifiable<D<const M31>, D<M31>>);
  static_assert(not modifiable<D<M31>, D<M21>>);
  static_assert(modifiable<D<M31>&, D<M31>>);
  static_assert(modifiable<D<M31&>, D<M31>>);
  static_assert(not modifiable<D<M31&>, M31>);
  static_assert(not modifiable<const D<M31>&, D<M31>>);
  static_assert(not modifiable<D<const M31&>, D<M31>>);
  static_assert(not modifiable<D<const M31>&, D<M31>>);
}


TEST(eigen3, Diagonal_class)
{
  // default constructor and comma expression, .nested_matrix()
  D3 d3a;
  d3a << 1, 2, 3;
  EXPECT_TRUE(is_near(d3a, m33));
  EXPECT_TRUE(is_near(d3a.nested_matrix(), m31));

  // construct dynamic diagonal matrix from dynamic eigen matrix; fill with comma expression
  DiagonalMatrix d0a_3 {M01 {make_native_matrix<M31>(4, 5, 6)}};
  static_assert(std::is_same_v<decltype(d0a_3), D0>);
  d0a_3 << 1, 2, 3;
  EXPECT_TRUE(is_near(d0a_3, m33));
  EXPECT_TRUE(is_near(d0a_3.nested_matrix(), make_native_matrix<M31>(1, 2, 3)));

  // move constructor, deduction guide (column vector)
  DiagonalMatrix d3b {DiagonalMatrix {M31 {m31}}};
  static_assert(std::is_same_v<decltype(d3b), D3>);
  EXPECT_TRUE(is_near(d3b, m33));
  DiagonalMatrix d0b_3 {D0 {M31 {m31}}}; // construct dynamic from fixed matrix, then move constructor
  static_assert(std::is_same_v<decltype(d0b_3), D0>);
  EXPECT_TRUE(is_near(d0b_3, m33));
  EXPECT_TRUE(is_near(DiagonalMatrix {D0 {D3 {m31}}}, m33)); // construct dynamic from fixed diagonal, then move constructor

  // copy constructor
  DiagonalMatrix d3c {d3b};
  EXPECT_TRUE(is_near(d3c, m33));
  DiagonalMatrix d0c_3 {d0b_3};
  EXPECT_TRUE(is_near(d0c_3, m33));

  // column scalar constructor and corresponding deduction guide
  DiagonalMatrix d3d {1., 2, 3};
  EXPECT_TRUE(is_near(d3d, m33));
  D0 d0d_3 {1., 2, 3}; // does not rely on deduction guide
  EXPECT_TRUE(is_near(d0d_3, m33));

  // column vector constructors and deduction guide
  EXPECT_TRUE(is_near(DiagonalMatrix {M31 {m31}}.nested_matrix(), m31));
  EXPECT_TRUE(is_near(DiagonalMatrix {M01{m01_3}}.nested_matrix(), m31));
  EXPECT_TRUE(is_near(DiagonalMatrix {M00{m00_31}}.nested_matrix(), m31));

  // square matrix constructors and square matrix deduction guide
  EXPECT_TRUE(is_near(DiagonalMatrix {M33 {m33}}.nested_matrix(), m31));
  EXPECT_TRUE(is_near(D3 {1., 0, 0, 0, 2, 0, 0, 0, 3}.nested_matrix(), m31));
  EXPECT_TRUE(is_near(D3 {M00 {m33}}.nested_matrix(), m31));
  EXPECT_TRUE(is_near(DiagonalMatrix {M00 {m33}}.nested_matrix(), m31));
  EXPECT_TRUE(is_near(DiagonalMatrix {M03{m03_3}}.nested_matrix(), m31));

  // construct from zero matrix, and deduction guide (from non-DiagonalMatrix diagonal)
  static_assert(zero_matrix<decltype(DiagonalMatrix {ZeroMatrix<double, 3, 1>{}})>);
  static_assert(diagonal_matrix<decltype(DiagonalMatrix {ZeroMatrix<double, 3, 1>{}})>);
  static_assert(square_matrix<decltype(DiagonalMatrix {ZeroMatrix<double, 3, 1>{}})>);

  EXPECT_TRUE(is_near(DiagonalMatrix {ZeroMatrix<double, 3, 1>{}}, M33::Zero()));
  EXPECT_TRUE(is_near(DiagonalMatrix {ZeroMatrix<double, 3, 3>{}}, M33::Zero()));
  EXPECT_TRUE(is_near(DiagonalMatrix {(DiagonalMatrix {ZeroMatrix<double, 3, 3>{}}).nested_matrix()}, M33::Zero()));
  EXPECT_TRUE(is_near(D0 {ZeroMatrix<double, 3, 1>{}}, M33::Zero()));
  EXPECT_TRUE(is_near(D0 {ZeroMatrix<double, 3, 3>{}}, M33::Zero()));
  EXPECT_TRUE(is_near(DiagonalMatrix {(DiagonalMatrix {ZeroMatrix<double, dynamic_extent, dynamic_extent> {3, 3}}).nested_matrix()}, M33::Zero()));
  EXPECT_TRUE(is_near(DiagonalMatrix {ZeroMatrix<double, dynamic_extent, 1>{3}}, M33::Zero()));
  EXPECT_TRUE(is_near(DiagonalMatrix {ZeroMatrix<double, 3, dynamic_extent>{1}}, M33::Zero()));
  EXPECT_TRUE(is_near(DiagonalMatrix {ZeroMatrix<double, 3, dynamic_extent>{3}}, M33::Zero()));
  EXPECT_TRUE(is_near(D3 {ZeroMatrix<double, dynamic_extent, 3>{3}}, M33::Zero()));
  EXPECT_TRUE(is_near(DiagonalMatrix {ZeroMatrix<double, dynamic_extent, 3>{3}}, M33::Zero()));
  EXPECT_TRUE(is_near(D3 {ZeroMatrix<double, dynamic_extent, dynamic_extent>{3, 3}}, M33::Zero()));
  EXPECT_TRUE(is_near(DiagonalMatrix {ZeroMatrix<double, dynamic_extent, dynamic_extent>{3, 3}}, M33::Zero()));

  // construct from identity matrix, and deduction guide (from non-DiagonalMatrix diagonal)
  EXPECT_TRUE(is_near(DiagonalMatrix {MatrixTraits<D3>::identity()}, M33::Identity()));
  EXPECT_TRUE(is_near(DiagonalMatrix {M33::Identity()}, M33::Identity()));
  EXPECT_TRUE(is_near(DiagonalMatrix {0.7 * M33::Identity()}, M33::Identity() * 0.7));
  EXPECT_TRUE(is_near(DiagonalMatrix {((0.7 * M33::Identity()) * (0.3 * M33::Identity() * 0.7 + 0.7 * M33::Identity()) - M33::Identity() * 0.3)}, M33::Identity() * 0.337));
  EXPECT_TRUE(is_near(DiagonalMatrix {((0.7 * M33::Identity()) * (0.3 * M33::Identity() * 0.7 + 0.7 * M33::Identity()) - M33::Identity() * 0.3)}, M33::Identity() * 0.337));
  EXPECT_TRUE(is_near(DiagonalMatrix {MatrixTraits<D0>::identity(3)}, M33::Identity()));
  EXPECT_TRUE(is_near(DiagonalMatrix {M00::Identity(3, 3)}, M33::Identity()));
  EXPECT_TRUE(is_near(DiagonalMatrix {0.7 * M00::Identity(3, 3)}, M00::Identity(3, 3) * 0.7));

  M22 msa2 = make_native_matrix<M22>(9, 0, 0, 10);

  EXPECT_TRUE(is_near(DiagonalMatrix {SelfAdjointMatrix<M22, TriangleType::diagonal>{msa2}}, msa2));
  EXPECT_TRUE(is_near(DiagonalMatrix {SelfAdjointMatrix<M00, TriangleType::diagonal>{msa2}}, msa2));
  EXPECT_TRUE(is_near(DiagonalMatrix {SelfAdjointMatrix<M20, TriangleType::diagonal>{msa2}}, msa2));
  EXPECT_TRUE(is_near(DiagonalMatrix {SelfAdjointMatrix<M02, TriangleType::diagonal>{msa2}}, msa2));

  EXPECT_TRUE(is_near(DiagonalMatrix {SelfAdjointMatrix<D2, TriangleType::diagonal>{msa2}}, msa2));
  EXPECT_TRUE(is_near(DiagonalMatrix {SelfAdjointMatrix<D0, TriangleType::diagonal>{msa2}}, msa2));

  EXPECT_TRUE(is_near(DiagonalMatrix {SelfAdjointMatrix<D2, TriangleType::lower>{msa2}}, msa2));
  EXPECT_TRUE(is_near(DiagonalMatrix {SelfAdjointMatrix<D0, TriangleType::lower>{msa2}}, msa2));

  M22 mt2 = make_native_matrix<M22>(3, 0, 0, 3);

  EXPECT_TRUE(is_near(DiagonalMatrix {TriangularMatrix<M22, TriangleType::diagonal>{mt2}}, mt2));
  EXPECT_TRUE(is_near(DiagonalMatrix {TriangularMatrix<M00, TriangleType::diagonal>{mt2}}, mt2));
  EXPECT_TRUE(is_near(DiagonalMatrix {TriangularMatrix<M20, TriangleType::diagonal>{mt2}}, mt2));
  EXPECT_TRUE(is_near(DiagonalMatrix {TriangularMatrix<M02, TriangleType::diagonal>{mt2}}, mt2));

  EXPECT_TRUE(is_near(DiagonalMatrix {TriangularMatrix<D2, TriangleType::diagonal>{mt2}}, mt2));
  EXPECT_TRUE(is_near(DiagonalMatrix {TriangularMatrix<D0, TriangleType::diagonal>{mt2}}, mt2));

  EXPECT_TRUE(is_near(DiagonalMatrix {TriangularMatrix<D2, TriangleType::lower>{mt2}}, mt2));
  EXPECT_TRUE(is_near(DiagonalMatrix {TriangularMatrix<D0, TriangleType::lower>{mt2}}, mt2));

  ZeroMatrix<double, 3, 3> z33 {};

  // move assignment.
  d3c = {4., 5, 6};
  EXPECT_TRUE(is_near(d3c, m33b));
  d0c_3 = {4., 5, 6};
  EXPECT_TRUE(is_near(d0c_3, m33b));

  // copy assignment.
  d3d = d3c;
  EXPECT_TRUE(is_near(d3d, m33b));
  d0d_3 = d0c_3;
  EXPECT_TRUE(is_near(d0d_3, m33b));

  // assign from different-typed eigen_diagonal_expr
  d3c = DiagonalMatrix<ZeroMatrix<double, 3, 1>> {ZeroMatrix<double, 3, 1> {}};
  EXPECT_TRUE(is_near(d3c, z33));
  d0c_3 = DiagonalMatrix<ZeroMatrix<double, dynamic_extent, 1>> {ZeroMatrix<double, 0, 1> {3}};
  EXPECT_TRUE(is_near(d0c_3, z33));
  d3d = DiagonalMatrix<ZeroMatrix<double, dynamic_extent, 1>> {ZeroMatrix<double, 0, 1> {3}};
  EXPECT_TRUE(is_near(d3d, z33));
  d0d_3 = DiagonalMatrix<ZeroMatrix<double, 3, 1>> {ZeroMatrix<double, 3, 1> {}};
  EXPECT_TRUE(is_near(d0d_3, z33));

  // assign from zero
  d3c = ZeroMatrix<double, 3, 3> {};
  EXPECT_TRUE(is_near(d3c, z33));
  d0c_3 = ZeroMatrix<double, dynamic_extent, dynamic_extent> {3, 3};
  EXPECT_TRUE(is_near(d0c_3, z33));
  d3d = ZeroMatrix<double, dynamic_extent, dynamic_extent> {3, 3};
  EXPECT_TRUE(is_near(d3d, z33));
  d0d_3 = ZeroMatrix<double, 3, 3> {};
  EXPECT_TRUE(is_near(d0d_3, z33));

  // assign from identity
  d3c = M33::Identity();
  EXPECT_TRUE(is_near(d3c, M33::Identity()));
  d0c_3 = M00::Identity(3, 3);
  EXPECT_TRUE(is_near(d0c_3, M33::Identity()));
  d3d = M00::Identity(3, 3);
  EXPECT_TRUE(is_near(d3d, M33::Identity()));
  d0d_3 = M33::Identity();
  EXPECT_TRUE(is_near(d0d_3, M33::Identity()));

  // Assign from general diagonal matrices:

  D2 d2a {1, 2};
  D0 d0a_2 {1, 2};
  M22 m22b = make_native_matrix<M22>(9, 0, 0, 10);
  M22 m22c = make_native_matrix<M22>(3, 0, 0, 3);

  d2a = SelfAdjointMatrix<M22, TriangleType::diagonal>{m22b};
  EXPECT_TRUE(is_near(d2a, m22b));
  d0a_2 = SelfAdjointMatrix<M00, TriangleType::diagonal>{m22b};
  EXPECT_TRUE(is_near(d0a_2, m22b));
  d2a = TriangularMatrix<M22, TriangleType::diagonal>{m22c};
  EXPECT_TRUE(is_near(d2a, m22c));
  d0a_2 = TriangularMatrix<M00, TriangleType::diagonal>{m22c};
  EXPECT_TRUE(is_near(d0a_2, m22c));

  d2a = SelfAdjointMatrix<D2, TriangleType::diagonal>{D2 {m22b}};
  EXPECT_TRUE(is_near(d2a, m22b));
  d0a_2 = SelfAdjointMatrix<D0, TriangleType::diagonal>{D0 {m22b}};
  EXPECT_TRUE(is_near(d0a_2, m22b));
  d2a = TriangularMatrix<D2, TriangleType::diagonal>{D2 {m22c}};
  EXPECT_TRUE(is_near(d2a, m22c));
  d0a_2 = TriangularMatrix<D0, TriangleType::diagonal>{D0 {m22c}};
  EXPECT_TRUE(is_near(d0a_2, m22c));

  d2a = SelfAdjointMatrix<D2, TriangleType::lower>{D2 {m22b}};
  EXPECT_TRUE(is_near(d2a, m22b));
  d0a_2 = SelfAdjointMatrix<D0, TriangleType::lower>{D0 {m22b}};
  EXPECT_TRUE(is_near(d0a_2, m22b));
  d2a = TriangularMatrix<D2, TriangleType::lower>{D2 {m22c}};
  EXPECT_TRUE(is_near(d2a, m22c));
  d0a_2 = TriangularMatrix<D0, TriangleType::lower>{D0 {m22c}};
  EXPECT_TRUE(is_near(d0a_2, m22c));

  // Arithmetic

  d3a += d3b;
  EXPECT_TRUE(is_near(d3a, D3 {2., 4, 6}));
  d0a_3 += d0b_3;
  EXPECT_TRUE(is_near(d0a_3, D3 {2., 4, 6}));
  d3b += M33::Identity();
  EXPECT_TRUE(is_near(d3b, D3 {2., 3, 4}));
  d0b_3 += M00::Identity(3, 3);
  EXPECT_TRUE(is_near(d0b_3, D3 {2., 3, 4}));

  d3a -= d3;
  EXPECT_TRUE(is_near(d3a, m33));
  d0a_3 -= d3;
  EXPECT_TRUE(is_near(d0a_3, m33));
  d3b -= M33::Identity();
  EXPECT_TRUE(is_near(d3b, m33));
  d0b_3 -= M00::Identity(3, 3);
  EXPECT_TRUE(is_near(d0b_3, m33));

  d3a *= 3;
  EXPECT_TRUE(is_near(d3a, D3 {3., 6, 9}));
  d0a_3 *= 3;
  EXPECT_TRUE(is_near(d0a_3, D3 {3., 6, 9}));

  d3a /= 3;
  EXPECT_TRUE(is_near(d3a, D3 {1., 2, 3}));
  d0a_3 /= 3;
  EXPECT_TRUE(is_near(d0a_3, D3 {1., 2, 3}));

  d3a *= d3b;
  EXPECT_TRUE(is_near(d3a, D3 {1., 4, 9}));
  d0a_3 *= d0b_3;
  EXPECT_TRUE(is_near(d0a_3, D3 {1., 4, 9}));

  EXPECT_TRUE(is_near(d3a.square_root(), m33));
  EXPECT_TRUE(is_near(d3a.square(), DiagonalMatrix {1., 16, 81}));
  EXPECT_TRUE(is_near(d0a_3.square_root(), m33));
  EXPECT_TRUE(is_near(d0a_3.square(), DiagonalMatrix {1., 16, 81}));

  EXPECT_EQ((D3::rows()), 3);
  EXPECT_EQ((D3::cols()), 3);
  EXPECT_EQ((d0_3.rows()), 3);
  EXPECT_EQ((d0_3.cols()), 3);

  EXPECT_TRUE(is_near(D3::zero(), M33::Zero()));
  EXPECT_TRUE(is_near(D3::identity(), M33::Identity()));
  EXPECT_TRUE(is_near(D0::zero(3), M33::Zero(3, 3)));
  EXPECT_TRUE(is_near(D0::identity(3), M33::Identity(3, 3)));
}

TEST(eigen3, Diagonal_subscripts)
{
  static_assert(element_gettable<D3, 2>);
  static_assert(element_gettable<D3, 1>);
  static_assert(not element_gettable<D3, 3>);
  static_assert(element_settable<D3, 2>);
  static_assert(element_settable<D3, 1>);

  static_assert(element_gettable<D0, 2>);
  static_assert(element_gettable<D0, 1>);
  static_assert(not element_gettable<D0, 3>);
  static_assert(element_settable<D0, 2>);
  static_assert(element_settable<D0, 1>);

  D3 d3a {1, 2, 3};
  D0 d0a_3 {1, 2, 3};
  bool test;

  set_element(d3a, 5.5, 0);
  EXPECT_NEAR(get_element(d3a, 0), 5.5, 1e-8);
  set_element(d3a, 6.5, 1, 1);
  EXPECT_NEAR(get_element(d3a, 1), 6.5, 1e-8);
  test = false;
  try { set_element(d3a, 8.5, 2, 0); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_NEAR(get_element(d3a, 2, 0), 0, 1e-8);
  set_element(d3a, 7.5, 2);

  set_element(d0a_3, 5.5, 0);
  EXPECT_NEAR(get_element(d0a_3, 0), 5.5, 1e-8);
  set_element(d0a_3, 6.5, 1, 1);
  EXPECT_NEAR(get_element(d0a_3, 1), 6.5, 1e-8);
  test = false;
  try { set_element(d0a_3, 8.5, 2, 0); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_NEAR(get_element(d0a_3, 2, 0), 0, 1e-8);
  set_element(d0a_3, 7.5, 2);

  d3a(0) = 5;
  d3a(1) = 6;
  d3a(2, 2) = 7;
  test = false;
  try { d3a(1, 0) = 3; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);

  d0a_3(0) = 5;
  d0a_3(1) = 6;
  d0a_3(2, 2) = 7;
  test = false;
  try { d0a_3(1, 0) = 3; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);

  EXPECT_TRUE(is_near(d3a, DiagonalMatrix {5., 6, 7}));
  EXPECT_NEAR(d3a(0), 5, 1e-6);
  EXPECT_NEAR(d3a(1), 6, 1e-6);
  EXPECT_NEAR(d3a(2), 7, 1e-6);
  EXPECT_NEAR(d3a[0], 5, 1e-6);
  EXPECT_NEAR(d3a[1], 6, 1e-6);
  EXPECT_NEAR(d3a[2], 7, 1e-6);
  EXPECT_NEAR(d3a(0, 0), 5, 1e-6);
  EXPECT_NEAR(d3a(0, 1), 0, 1e-6);
  EXPECT_NEAR(d3a(0, 2), 0, 1e-6);
  EXPECT_NEAR(d3a(1, 0), 0, 1e-6);
  EXPECT_NEAR(d3a(1, 1), 6, 1e-6);
  EXPECT_NEAR(d3a(1, 2), 0, 1e-6);
  EXPECT_NEAR(d3a(2, 0), 0, 1e-6);
  EXPECT_NEAR(d3a(2, 1), 0, 1e-6);
  EXPECT_NEAR(d3a(2, 2), 7, 1e-6);

  EXPECT_TRUE(is_near(d0a_3, DiagonalMatrix {5., 6, 7}));
  EXPECT_NEAR(d0a_3(0), 5, 1e-6);
  EXPECT_NEAR(d0a_3(1), 6, 1e-6);
  EXPECT_NEAR(d0a_3(2), 7, 1e-6);
  EXPECT_NEAR(d0a_3[0], 5, 1e-6);
  EXPECT_NEAR(d0a_3[1], 6, 1e-6);
  EXPECT_NEAR(d0a_3[2], 7, 1e-6);
  EXPECT_NEAR(d0a_3(0, 0), 5, 1e-6);
  EXPECT_NEAR(d0a_3(0, 1), 0, 1e-6);
  EXPECT_NEAR(d0a_3(0, 2), 0, 1e-6);
  EXPECT_NEAR(d0a_3(1, 0), 0, 1e-6);
  EXPECT_NEAR(d0a_3(1, 1), 6, 1e-6);
  EXPECT_NEAR(d0a_3(1, 2), 0, 1e-6);
  EXPECT_NEAR(d0a_3(2, 0), 0, 1e-6);
  EXPECT_NEAR(d0a_3(2, 1), 0, 1e-6);
  EXPECT_NEAR(d0a_3(2, 2), 7, 1e-6);

  EXPECT_NEAR((D3 {1., 2, 3}).nested_matrix()[0], 1, 1e-6);
  EXPECT_NEAR((D3 {1., 2, 3}).nested_matrix()[1], 2, 1e-6);
  EXPECT_NEAR((D3 {1., 2, 3}).nested_matrix()[2], 3, 1e-6);

  EXPECT_NEAR((D0 {1., 2, 3}).nested_matrix()[0], 1, 1e-6);
  EXPECT_NEAR((D0 {1., 2, 3}).nested_matrix()[1], 2, 1e-6);
  EXPECT_NEAR((D0 {1., 2, 3}).nested_matrix()[2], 3, 1e-6);
}

TEST(eigen3, Diagonal_traits)
{
  static_assert(diagonal_matrix<decltype(DiagonalMatrix{2.})>);
  static_assert(diagonal_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(self_adjoint_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(triangular_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(lower_triangular_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(upper_triangular_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(not identity_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(not zero_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(covariance_nestable<decltype(DiagonalMatrix{2, 3})>);
  static_assert(covariance_nestable<decltype(M33::Identity())>);
  static_assert(identity_matrix<decltype(M33::Identity() * M33::Identity())>);
  static_assert(diagonal_matrix<decltype(0.3 * M33::Identity() + 0.7 * M33::Identity() * 0.7)>);
  static_assert(diagonal_matrix<decltype(0.7 * M33::Identity() * 0.7 - M33::Identity() * 0.3)>);
  static_assert(diagonal_matrix<decltype((0.7 * M33::Identity()) * (M33::Identity() + M33::Identity()))>);

  // MatrixTraits
  EXPECT_TRUE(is_near(MatrixTraits<D3>::make(m31), m33));
  EXPECT_TRUE(is_near(MatrixTraits<D3>::make(1, 2, 3), m33));
  EXPECT_TRUE(is_near(make_native_matrix<D3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3), m33));
  EXPECT_TRUE(is_near(MatrixTraits<D3>::zero(), M33::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<D3>::identity(), M33::Identity()));

  EXPECT_TRUE(is_near(MatrixTraits<D0>::make(m31), m33));
  EXPECT_TRUE(is_near(MatrixTraits<D0>::make(1, 2, 3), m33));
  EXPECT_TRUE(is_near(MatrixTraits<D0>::zero(3), M33::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<D0>::identity(3), M33::Identity()));
}

TEST(eigen3, Diagonal_overloads)
{
  EXPECT_TRUE(is_near(make_native_matrix(d3), m33));
  EXPECT_TRUE(is_near(make_native_matrix(d0_3), m33));

  EXPECT_EQ(row_count(d3), 3);
  EXPECT_EQ(row_count(d0_3), 3);

  EXPECT_EQ(column_count(d3), 3);
  EXPECT_EQ(column_count(d0_3), 3);

  EXPECT_TRUE(is_near(Cholesky_square(d3), D3 {1, 4, 9}));
  EXPECT_TRUE(is_near(Cholesky_square(d0_3), D3 {1, 4, 9}));
  EXPECT_TRUE(is_near(Cholesky_factor(D3 {1., 4, 9}), m33));
  EXPECT_TRUE(is_near(Cholesky_factor(D0 {1., 4, 9}), m33));
  EXPECT_TRUE(is_near(Cholesky_square(M11 {4}), M11 {16}));
  EXPECT_TRUE(is_near(Cholesky_factor(M11 {4}), M11 {2}));
  EXPECT_TRUE(is_near(Cholesky_square((M00 {1,1} << 4).finished()), M11 {16}));
  EXPECT_TRUE(is_near(Cholesky_factor((M00 {1,1} << 4).finished()), M11 {2}));
  EXPECT_TRUE(is_near(Cholesky_square(M22::Identity() * 0.1), D2 {0.01, 0.01}));
  EXPECT_TRUE(is_near(Cholesky_factor(M22::Identity() * 0.01), D2 {0.1, 0.1}));
  EXPECT_TRUE(is_near(Cholesky_square(M00::Identity(2, 2) * 0.1), D2 {0.01, 0.01}));
  EXPECT_TRUE(is_near(Cholesky_factor(M00::Identity(2, 2) * 0.01), D2 {0.1, 0.1}));
  EXPECT_TRUE(is_near(Cholesky_square(DiagonalMatrix {9.}), M11 {81}));
  EXPECT_TRUE(is_near(Cholesky_square(D0 {M11 {9}}), M11 {81}));
  EXPECT_TRUE(is_near(Cholesky_factor(DiagonalMatrix {9.}), M11 {3}));
  EXPECT_TRUE(is_near(Cholesky_factor(D0 {M11 {9}}), M11 {3}));

  EXPECT_TRUE(is_near(diagonal_of(d3), m31));
  EXPECT_TRUE(is_near(diagonal_of(d0_3), m31));

  EXPECT_TRUE(is_near(transpose(d3), d3));
  EXPECT_TRUE(is_near(transpose(DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}), DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}));

  EXPECT_TRUE(is_near(adjoint(d3), d3));
  EXPECT_TRUE(is_near(adjoint(DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}), DiagonalMatrix {cdouble(1,-2), cdouble(2,-3), 3}));

  EXPECT_NEAR(determinant(DiagonalMatrix {2., 3, 4}), 24, 1e-6);

  EXPECT_NEAR(trace(DiagonalMatrix {2., 3, 4}), 9, 1e-6);
  //
  auto s2 = DiagonalMatrix {3., 3};
  rank_update(s2, DiagonalMatrix {2., 2}, 4);
  EXPECT_TRUE(is_near(s2, MatrixTraits<M22>::make(5., 0, 0, 5)));
  s2 = DiagonalMatrix {3., 3};
  rank_update(s2, 2 * M22::Identity(), 4);
  EXPECT_TRUE(is_near(s2, MatrixTraits<M22>::make(5., 0, 0, 5)));
  //
  using M1by1 = eigen_matrix_t<double, 1, 1>;
  auto s2a = M1by1 {3.};
  rank_update(s2a, M1by1 {2.}, 4);
  EXPECT_TRUE(is_near(s2a, M1by1 {5.}));
  //
  EXPECT_TRUE(is_near(rank_update(DiagonalMatrix {3., 3}, DiagonalMatrix {2., 2}, 4), make_native_matrix<double, 2, 2>(5., 0, 0, 5.)));
  EXPECT_TRUE(is_near(rank_update(DiagonalMatrix {3., 3}, 2 * M22::Identity(), 4), make_native_matrix<M22>(5., 0, 0, 5.)));
  EXPECT_TRUE(is_near(rank_update(DiagonalMatrix {3., 3}, make_native_matrix<M22>(2, 0, 0, 2.), 4), make_native_matrix<M22>(5., 0, 0, 5.)));
  //
  EXPECT_TRUE(is_near(rank_update(M1by1 {3.}, M1by1 {2.}, 4), M1by1 {5.}));
  //
  EXPECT_TRUE(is_near(solve(d3, make_native_matrix<double, 3, 1>(4., 10, 18)),
    make_native_matrix(4., 5, 6)));
  EXPECT_TRUE(is_near(reduce_columns(d3), make_native_matrix(1., 2, 3)));
  EXPECT_TRUE(is_near(reduce_rows(d3), make_native_matrix(1., 2, 3)));
  EXPECT_TRUE(is_near(LQ_decomposition(d3), d3));
  EXPECT_TRUE(is_near(QR_decomposition(d3), d3));
  EXPECT_TRUE(is_near(LQ_decomposition(DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}), DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}));
  EXPECT_TRUE(is_near(QR_decomposition(DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}), DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}));
  EXPECT_TRUE(is_near(LQ_decomposition(M1by1 {4}), DiagonalMatrix {4.}));
  EXPECT_TRUE(is_near(QR_decomposition(M1by1 {4}), DiagonalMatrix {4.}));

  using N = std::normal_distribution<double>;

  D2 d3b = MatrixTraits<D2>::zero();
  D0 d0_2 {MatrixTraits<D0>::zero(2)};
  D0 d0_3 {MatrixTraits<D0>::zero(3)};
  for (int i=0; i<100; i++)
  {
    d3b = (d3b * i + randomize<D2>(N {1.0, 0.3}, 2.0)) / (i + 1);
    d0_2 = (d0_2 * i + randomize<D2>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
    d0_3 = (d0_3 * i + randomize<D0>(3, 3, N {1.0, 0.3})) / (i + 1);
  }
  D2 d2_offset = {1, 2};
  D3 d0_3_offset = {1, 1, 1};
  EXPECT_TRUE(is_near(d3b, d2_offset, 0.1));
  EXPECT_FALSE(is_near(d3b, d2_offset, 1e-6));
  EXPECT_TRUE(is_near(d0_2, d2_offset, 0.1));
  EXPECT_FALSE(is_near(d0_2, d2_offset, 1e-6));
  EXPECT_TRUE(is_near(d0_3, d0_3_offset, 0.1));
  EXPECT_FALSE(is_near(d0_3, d0_3_offset, 1e-6));
}

TEST(eigen3, Diagonal_blocks)
{
  EXPECT_TRUE(is_near(concatenate_diagonal(d3, D2 {4, 5}), d5));
  EXPECT_TRUE(is_near(concatenate_diagonal(d3, D0 {4, 5}), d5));
  EXPECT_TRUE(is_near(concatenate_diagonal(d0_3, D2 {4, 5}), d5));
  EXPECT_TRUE(is_near(concatenate_diagonal(d0_3, D0 {4, 5}), d5));

  auto m_1234vert = make_native_matrix<4,2>(1., 0, 0, 2, 3, 0, 0, 4);
  EXPECT_TRUE(is_near(concatenate_vertical(D2 {1, 2}, D2 {3, 4}), m_1234vert));
  EXPECT_TRUE(is_near(concatenate_vertical(D2 {1, 2}, D0 {3, 4}), m_1234vert));
  EXPECT_TRUE(is_near(concatenate_vertical(D0 {1, 2}, D2 {3, 4}), m_1234vert));
  EXPECT_TRUE(is_near(concatenate_vertical(D0 {1, 2}, D0 {3, 4}), m_1234vert));

  auto m_1234horiz = make_native_matrix<2,4>(1., 0, 3, 0, 0, 2, 0, 4);
  EXPECT_TRUE(is_near(concatenate_horizontal(D2 {1, 2}, D2 {3, 4}), m_1234horiz));
  EXPECT_TRUE(is_near(concatenate_horizontal(D2 {1, 2}, D0 {3, 4}), m_1234horiz));
  EXPECT_TRUE(is_near(concatenate_horizontal(D0 {1, 2}, D2 {3, 4}), m_1234horiz));
  EXPECT_TRUE(is_near(concatenate_horizontal(D0 {1, 2}, D0 {3, 4}), m_1234horiz));

  EXPECT_TRUE(is_near(split_diagonal(d5), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal(d0_5), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(d5), std::tuple {d3, D2 {4, 5}}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(d5_const), std::tuple {d3, D2 {4, 5}}));
  //EXPECT_TRUE(is_near(split_diagonal<3, 2>(d0_5), std::tuple {d3, D2 {4, 5}}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(d5), std::tuple {D2 {1, 2}, D2 {3, 4}}));
  //EXPECT_TRUE(is_near(split_diagonal<2, 2>(d0_5), std::tuple {D2 {1, 2}, D2 {3, 4}}));

  EXPECT_TRUE(is_near(split_vertical(d5), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical(d0_5), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(d5),
    std::tuple {make_native_matrix<double, 3, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0),
               make_native_matrix<double, 2, 5>(
                 0, 0, 0, 4, 0,
                 0, 0, 0, 0, 5)}));
  /*EXPECT_TRUE(is_near(split_vertical<3, 2>(d0_5),
    std::tuple {make_native_matrix<double, 3, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0),
                make_native_matrix<double, 2, 5>(
                  0, 0, 0, 4, 0,
                  0, 0, 0, 0, 5)}));*/
  EXPECT_TRUE(is_near(split_vertical<2, 2>(d5),
    std::tuple {make_native_matrix<double, 2, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0),
               make_native_matrix<double, 2, 5>(
                 0, 0, 3, 0, 0,
                 0, 0, 0, 4, 0)}));
  /*EXPECT_TRUE(is_near(split_vertical<2, 2>(d0_5),
    std::tuple {make_native_matrix<double, 2, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0),
                make_native_matrix<double, 2, 5>(
                  0, 0, 3, 0, 0,
                  0, 0, 0, 4, 0)}));*/

  EXPECT_TRUE(is_near(split_horizontal(d5), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(d0_5), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(d5),
    std::tuple {make_native_matrix<double, 5, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      0, 0, 0,
      0, 0, 0),
               make_native_matrix<double, 5, 2>(
                 0, 0,
                 0, 0,
                 0, 0,
                 4, 0,
                 0, 5)}));
  /*EXPECT_TRUE(is_near(split_horizontal<3, 2>(d0_5),
    std::tuple {make_native_matrix<double, 5, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      0, 0, 0,
      0, 0, 0),
                make_native_matrix<double, 5, 2>(
                  0, 0,
                  0, 0,
                  0, 0,
                  4, 0,
                  0, 5)}));*/
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(d5),
    std::tuple {make_native_matrix<double, 5, 2>(
      1, 0,
      0, 2,
      0, 0,
      0, 0,
      0, 0),
               make_native_matrix<double, 5, 2>(
                 0, 0,
                 0, 0,
                 3, 0,
                 0, 4,
                 0, 0)}));
  /*EXPECT_TRUE(is_near(split_horizontal<2, 2>(d0_5),
    std::tuple {make_native_matrix<double, 5, 2>(
      1, 0,
      0, 2,
      0, 0,
      0, 0,
      0, 0),
                make_native_matrix<double, 5, 2>(
                  0, 0,
                  0, 0,
                  3, 0,
                  0, 4,
                  0, 0)}));*/

  EXPECT_TRUE(is_near(column(d3, 2), make_native_matrix(0., 0, 3)));
  EXPECT_TRUE(is_near(column<1>(d3), make_native_matrix(0., 2, 0)));
  
  EXPECT_TRUE(is_near(row(d3, 2), eigen_matrix_t<double, 1, 3>(0., 0, 3)));
  EXPECT_TRUE(is_near(row<1>(d3), eigen_matrix_t<double, 1, 3>(0., 2, 0)));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, d3),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, d3),
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      0, 3, 2,
      0, 1, 5)));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, d3),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }, d3),
    make_native_matrix<double, 3, 3>(
      1, 0, 0,
      1, 3, 1,
      2, 2, 5)));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, d3),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, d3),
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      1, 4, 3,
      2, 3, 7)));
}

TEST(eigen3, Diagonal_arithmetic)
{
  auto d3a = d3;
  auto d3b = DiagonalMatrix {4., 5, 6};
  auto i = M33::Identity();
  auto z = ZeroMatrix<double, 3, 3> {};
  EXPECT_TRUE(is_near(d3a + d3b, DiagonalMatrix {5., 7, 9})); static_assert(eigen_diagonal_expr<decltype(d3a + d3b)>);
  EXPECT_TRUE(is_near(d3a + i, DiagonalMatrix {2., 3, 4})); static_assert(eigen_diagonal_expr<decltype(d3a + i)>);
  EXPECT_TRUE(is_near(d3a + z, d3a)); static_assert(eigen_diagonal_expr<decltype(d3a + z)>);
  EXPECT_TRUE(is_near(i + i, DiagonalMatrix {2., 2, 2})); static_assert(diagonal_matrix<decltype(i + i)>);
  EXPECT_TRUE(is_near(i + z, i)); static_assert(identity_matrix<decltype(i + z)>);
  EXPECT_TRUE(is_near(z + i, i)); static_assert(identity_matrix<decltype(z + i)>);
  EXPECT_TRUE(is_near(z + z, z)); static_assert(eigen_zero_expr<decltype(z + z)>);

  EXPECT_TRUE(is_near(d3a - d3b, DiagonalMatrix {-3., -3, -3})); static_assert(diagonal_matrix<decltype(d3a - d3b)>);
  EXPECT_TRUE(is_near(d3a - i, DiagonalMatrix {0., 1, 2})); static_assert(diagonal_matrix<decltype(d3a - i)>);
  EXPECT_TRUE(is_near(d3a - z, d3a)); static_assert(diagonal_matrix<decltype(d3a - z)>);
  EXPECT_TRUE(is_near(i - i, z)); static_assert(zero_matrix<decltype(i - i)>);
  EXPECT_TRUE(is_near(i * (i - i), z)); static_assert(zero_matrix<decltype(i - i)>);
  EXPECT_TRUE(is_near(z * (i - i), z)); static_assert(eigen_zero_expr<decltype(z * (i - i))>);
  EXPECT_TRUE(is_near(i - z, i)); static_assert(identity_matrix<decltype(i - z)>);
  EXPECT_TRUE(is_near(z - i, -i)); static_assert(diagonal_matrix<decltype(z - i)>);
  EXPECT_TRUE(is_near(z - z, z)); static_assert(eigen_zero_expr<decltype(z - z)>);

  EXPECT_TRUE(is_near(d3a * 4, DiagonalMatrix {4., 8, 12})); static_assert(eigen_diagonal_expr<decltype(d3a * 4)>);
  EXPECT_TRUE(is_near(4 * d3a, DiagonalMatrix {4., 8, 12})); static_assert(eigen_diagonal_expr<decltype(4 * d3a)>);
  EXPECT_TRUE(is_near(d3a / 2, DiagonalMatrix {0.5, 1, 1.5})); static_assert(diagonal_matrix<decltype(d3a / 2)>);
  static_assert(diagonal_matrix<decltype(d3a / 0)>);
  EXPECT_TRUE(is_near(i * 4, DiagonalMatrix {4., 4, 4})); static_assert(diagonal_matrix<decltype(i * 4)>);
  EXPECT_TRUE(is_near(4 * i, DiagonalMatrix {4., 4, 4})); static_assert(diagonal_matrix<decltype(4 * i)>);
  static_assert(diagonal_matrix<decltype(i * 0)>);
  static_assert(not zero_matrix<decltype(i * 0)>);
  EXPECT_TRUE(is_near(i / 2, DiagonalMatrix {0.5, 0.5, 0.5})); static_assert(diagonal_matrix<decltype(i / 2)>);
  EXPECT_TRUE(is_near(z * 4, z)); static_assert(zero_matrix<decltype(z * 4)>);
  EXPECT_TRUE(is_near(4 * z, z)); static_assert(zero_matrix<decltype(4 * z)>);
  EXPECT_TRUE(is_near(z / 4, z)); static_assert(zero_matrix<decltype(z / 4)>);
  static_assert(zero_matrix<decltype(z / 0)>); // This is not technically true, but it's indeterminate at compile time.
  EXPECT_TRUE(is_near((i - i) * 4, z)); static_assert(zero_matrix<decltype((i - i) * 4)>);
  EXPECT_TRUE(is_near((i - i) / 4, z)); static_assert(not zero_matrix<decltype((i - i) / 4)>);
  static_assert(diagonal_matrix<decltype(i / 0)>);

  EXPECT_TRUE(is_near(-d3a, DiagonalMatrix {-1., -2, -3})); static_assert(eigen_diagonal_expr<decltype(-d3a)>);
  EXPECT_TRUE(is_near(-i, DiagonalMatrix {-1., -1, -1})); static_assert(diagonal_matrix<decltype(-i)>);
  EXPECT_TRUE(is_near(-z, z)); static_assert(eigen_zero_expr<decltype(z)>);

  EXPECT_TRUE(is_near(d3a * d3b, DiagonalMatrix {4., 10, 18})); static_assert(eigen_diagonal_expr<decltype(d3a * d3b)>);
  EXPECT_TRUE(is_near(d3a * ConstantMatrix<double, 4, 3, 3> {}, make_native_matrix<M33>(4, 4, 4, 8, 8, 8, 12, 12, 12)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 4, 3, 3> {} * d3a, make_native_matrix<M33>(4, 8, 12, 4, 8, 12, 4, 8, 12)));
  EXPECT_TRUE(is_near(d3a * i, d3a)); static_assert(eigen_diagonal_expr<decltype(d3a * i)>);
  EXPECT_TRUE(is_near(i * d3a, d3a)); static_assert(eigen_diagonal_expr<decltype(i * d3a)>);
  EXPECT_TRUE(is_near(d3a * z, z)); static_assert(eigen_zero_expr<decltype(d3a * z)>);
  EXPECT_TRUE(is_near(z * d3a, z)); static_assert(eigen_zero_expr<decltype(z * d3a)>);
  EXPECT_TRUE(is_near(i * i, i)); static_assert(identity_matrix<decltype(i * i)>);
  EXPECT_TRUE(is_near(z * z, z)); static_assert(eigen_zero_expr<decltype(z * z)>);

  EXPECT_TRUE(is_near(d3a * SelfAdjointMatrix {
    1., 2, 3,
    2, 4, 5,
    3, 5, 6}, make_native_matrix<3,3>(
      1., 2, 3,
      4, 8, 10,
      9, 15, 18)));
  EXPECT_TRUE(is_near(SelfAdjointMatrix {
    1., 2, 3,
    2, 4, 5,
    3, 5, 6} * d3, make_native_matrix<3,3>(
    1., 4, 9,
    2, 8, 15,
    3, 10, 18)));
  EXPECT_TRUE(is_near(d3 * TriangularMatrix {
    1., 0, 0,
    2, 3, 0,
    4, 5, 6}, make_native_matrix<3,3>(
    1., 0, 0,
    4, 6, 0,
    12, 15, 18)));
  EXPECT_TRUE(is_near(TriangularMatrix {
    1., 0, 0,
    2, 3, 0,
    4, 5, 6} * d3, make_native_matrix<3,3>(
    1., 0, 0,
    2, 6, 0,
    4, 10, 18)));
  EXPECT_TRUE(is_near(d3 * make_native_matrix<double, 3, 3>(
    1, 2, 3,
    4, 5, 6,
    7, 8, 9), make_native_matrix<3,3>(
    1., 2, 3,
    8, 10, 12,
    21, 24, 27)));
  EXPECT_TRUE(is_near(make_native_matrix<double, 3, 3>(
    1, 2, 3,
    4, 5, 6,
    7, 8, 9) * d3, make_native_matrix<3,3>(
    1., 4, 9,
    4, 10, 18,
    7, 16, 27)));
}

TEST(eigen3, Diagonal_references)
{
  DiagonalMatrix<const M31> n {4, 5, 6};
  DiagonalMatrix<M31> x = d3;
  DiagonalMatrix<M31&> x_l {x};
  EXPECT_TRUE(is_near(x_l, d3));
  DiagonalMatrix x_l2 {x_l};
  static_assert(std::is_lvalue_reference_v<nested_matrix_t<decltype(x_l2)>>);
  EXPECT_TRUE(is_near(x_l, d3));
  DiagonalMatrix<const M31&> x_lc = x_l;
  EXPECT_TRUE(is_near(x_lc, d3));
  x = n;
  EXPECT_TRUE(is_near(x_l, n));
  EXPECT_TRUE(is_near(x_l2, n));
  EXPECT_TRUE(is_near(x_lc, n));
  x_l2[0] = 1;
  x_l2[1] = 2;
  x_l2[2] = 3;
  EXPECT_TRUE(is_near(x, d3));
  EXPECT_TRUE(is_near(x_l, d3));
  EXPECT_TRUE(is_near(x_lc, d3));
  EXPECT_TRUE(is_near(DiagonalMatrix<M31&> {d3}.nested_matrix(), (M31 {} << 1, 2, 3).finished() ));

  M31 p; p << 10, 11, 12;
  M31 q; q << 13, 14, 15;
  DiagonalMatrix yl {p};
  static_assert(std::is_lvalue_reference_v<nested_matrix_t<decltype(yl)>>);
  EXPECT_TRUE(is_near(diagonal_of(yl), p));
  DiagonalMatrix yr {(M31 {} << 13, 14, 15).finished() * 1.0};
  static_assert(not std::is_reference_v<nested_matrix_t<decltype(yr)>>);
  EXPECT_TRUE(is_near(diagonal_of(yr), q));
  yl = DiagonalMatrix {q};
  EXPECT_TRUE(is_near(p, q));
  p = (M31 {} << 16, 17, 18).finished();
  EXPECT_TRUE(is_near(diagonal_of(yl), p));
}