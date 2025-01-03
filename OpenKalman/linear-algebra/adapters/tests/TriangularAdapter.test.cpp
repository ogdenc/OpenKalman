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
 * \brief Tests relating to TriangularAdapter.
 */

#include "adapters.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


namespace
{
  using cdouble = std::complex<double>;

  using M11 = eigen_matrix_t<double, 1, 1>;
  using M1x = eigen_matrix_t<double, 1, dynamic_size_v>;
  using Mx1 = eigen_matrix_t<double, dynamic_size_v, 1>;
  using Mxx = eigen_matrix_t<double, dynamic_size_v, dynamic_size_v>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M2x = eigen_matrix_t<double, 2, dynamic_size_v>;
  using Mx2 = eigen_matrix_t<double, dynamic_size_v, 2>;

  using CM2 = eigen_matrix_t<std::complex<double>, 2, 2>;
  using D2 = DiagonalAdapter<eigen_matrix_t<double, 2, 1>>;
  using Lower = TriangularAdapter<M22, TriangleType::lower>;
  using Upper = TriangularAdapter<M22, TriangleType::upper>;
  using CLower = TriangularAdapter<CM2, TriangleType::lower>;
  using CUpper = TriangularAdapter<CM2, TriangleType::upper>;
  using Diagonal = TriangularAdapter<M22, TriangleType::diagonal>;
  using Diagonal2 = TriangularAdapter<D2, TriangleType::diagonal>;
  using Diagonal3 = TriangularAdapter<D2, TriangleType::lower>;

  template<typename...Args>
  inline auto mat22(Args...args) { return make_dense_writable_matrix_from<M22>(args...); }

  template<typename T> using Tl = TriangularAdapter<T, TriangleType::lower>;
  template<typename T> using Tu = TriangularAdapter<T, TriangleType::upper>;
}


TEST(special_matrices, TriangularAdapter_static_checks)
{
  static_assert(writable<Tl<M22>>);
  static_assert(writable<Tl<M22&>>);
  static_assert(not writable<Tl<const M22>>);
  static_assert(not writable<Tl<const M22&>>);

  static_assert(writable<Tu<M22>>);
  static_assert(writable<Tu<M22&>>);
  static_assert(not writable<Tu<const M22>>);
  static_assert(not writable<Tu<const M22&>>);
  
  static_assert(diagonal_matrix<Lower>);
  static_assert(diagonal_matrix<Upper>);
  static_assert(diagonal_matrix<Diagonal>);
  static_assert(diagonal_matrix<Diagonal2>);
  static_assert(diagonal_matrix<Diagonal3>);
  static_assert(diagonal_matrix<Tl<M2x>>);
  static_assert(diagonal_matrix<Tl<Mx2>>);
  static_assert(diagonal_matrix<Tl<Mxx>>);

  static_assert(triangular_matrix<Lower, TriangleType::lower>);
  static_assert(not triangular_matrix<Upper, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Diagonal, TriangleType::lower>);
  static_assert(triangular_matrix<Diagonal2, TriangleType::lower>);
  static_assert(triangular_matrix<Diagonal3, TriangleType::lower>);
  static_assert(triangular_matrix<Tl<M2x>, TriangleType::lower>);
  static_assert(triangular_matrix<Tl<Mx2>, TriangleType::lower>);
  static_assert(triangular_matrix<Tl<Mxx>, TriangleType::lower>);
  static_assert(triangular_matrix<Tu<Mxx>, TriangleType::lower, Likelihood::maybe>);

  static_assert(triangular_matrix<Upper, TriangleType::upper>);
  static_assert(not triangular_matrix<Lower, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Diagonal, TriangleType::upper>);
  static_assert(triangular_matrix<Diagonal2, TriangleType::upper>);
  static_assert(triangular_matrix<Diagonal3, TriangleType::upper>);
  static_assert(triangular_matrix<Tu<M2x>, TriangleType::upper>);
  static_assert(triangular_matrix<Tu<Mx2>, TriangleType::upper>);
  static_assert(triangular_matrix<Tu<Mxx>, TriangleType::upper>);
  static_assert(triangular_matrix<Tl<Mxx>, TriangleType::upper, Likelihood::maybe>);

  static_assert(square_matrix<Lower>);
  static_assert(square_matrix<Upper>);
  static_assert(square_matrix<Diagonal>);
  static_assert(square_matrix<Diagonal2>);
  static_assert(square_matrix<Diagonal3>);
  static_assert(square_matrix<Tl<Mxx>>);
  static_assert(square_matrix<Tu<Mxx>>);

  // \todo Fill in other traits
  
  static_assert(modifiable<Tl<M22>, ZeroAdapter<eigen_matrix_t<double, 2, 2>>>);
  static_assert(modifiable<Tu<M22>, ZeroAdapter<eigen_matrix_t<double, 2, 2>>>);
  static_assert(modifiable<Tl<M22>, Eigen3::IdentityMatrix<M22>>);
  static_assert(modifiable<Tu<M22>, Eigen3::IdentityMatrix<M22>>);
  static_assert(not modifiable<Tu<M22::IdentityReturnType>, Tu<M22::IdentityReturnType>>);
  static_assert(modifiable<Tl<M22>, D2>);
  static_assert(modifiable<Tu<M22>, D2>);
  static_assert(not modifiable<Tl<M22>, M22>);
  static_assert(not modifiable<Tu<M22>, M22>);
  static_assert(modifiable<Tl<M22>, Tl<M22>>);
  static_assert(modifiable<Tu<M22>, Tu<M22>>);
  static_assert(modifiable<Tl<M22>, const Tl<M22>>);
  static_assert(modifiable<Tu<M22>, const Tu<M22>>);
  static_assert(modifiable<Tl<M22>, Tl<const M22>>);
  static_assert(modifiable<Tu<M22>, Tu<const M22>>);
  static_assert(not modifiable<Tl<const M22>, Tl<M22>>);
  static_assert(not modifiable<Tu<const M22>, Tu<M22>>);
  static_assert(not modifiable<Tl<M22>, Tl<eigen_matrix_t<double, 3, 3>>>);
  static_assert(not modifiable<Tu<M22>, Tu<eigen_matrix_t<double, 3, 3>>>);
  static_assert(modifiable<Tl<M22>&, Tl<M22>>);
  static_assert(modifiable<Tu<M22>&, Tu<M22>>);
  static_assert(modifiable<Tl<M22&>, Tl<M22>>);
  static_assert(modifiable<Tu<M22&>, Tu<M22>>);
  static_assert(not modifiable<Tl<M22&>, M22>);
  static_assert(not modifiable<Tu<M22&>, M22>);
  static_assert(not modifiable<const Tl<M22>&, Tl<M22>>);
  static_assert(not modifiable<const Tu<M22>&, Tu<M22>>);
  static_assert(not modifiable<Tl<const M22&>, Tl<M22>>);
  static_assert(not modifiable<Tu<const M22&>, Tu<M22>>);
  static_assert(not modifiable<Tl<const M22>&, Tl<M22>>);
  static_assert(not modifiable<Tu<const M22>&, Tu<M22>>);
}


TEST(special_matrices, TriangularAdapter_class)
{
  M22 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  D2 d2 {3, 3};
  //
  Lower l1;
  l1 << 3, 0, 1, 3;
  EXPECT_TRUE(is_near(l1.nested_matrix(), ml));
  Upper u1;
  u1 << 3, 1, 0, 3;
  EXPECT_TRUE(is_near(u1.nested_matrix(), mu));
  Diagonal d1;
  d1 << 3, 3;
  EXPECT_TRUE(is_near(d1.nested_matrix(), mat22(3, 0, 0, 3)));
  EXPECT_TRUE(is_near(d1, mat22(3, 0, 0, 3)));
  d1.template triangularView<Eigen::Lower>() = make_dense_writable_matrix_from<M22>(2, 5, 6, 2);
  EXPECT_TRUE(is_near(d1, mat22(2, 0, 0, 2)));
  Diagonal2 d1b;
  d1b << 3, 3;
  EXPECT_TRUE(is_near(d1b.nested_matrix(), mat22(3, 0, 0, 3)));
  EXPECT_TRUE(is_near(d1b, mat22(3, 0, 0, 3)));
  d1b.template triangularView<Eigen::Lower>() = make_dense_writable_matrix_from<M22>(2, 5, 6, 2);
  EXPECT_TRUE(is_near(d1b, mat22(2, 0, 0, 2)));
  Diagonal3 d1c;
  d1c << 3, 3;
  EXPECT_TRUE(is_near(d1c.nested_matrix(), mat22(3, 0, 0, 3)));
  EXPECT_TRUE(is_near(d1c, mat22(3, 0, 0, 3)));
  d1c.template triangularView<Eigen::Lower>() = make_dense_writable_matrix_from<M22>(2, 5, 6, 2);
  EXPECT_TRUE(is_near(d1c, mat22(2, 0, 0, 2)));
  //
  Lower l2 {make_dense_writable_matrix_from<M22>(3, 0, 1, 3)};
  EXPECT_TRUE(is_near(l1, l2));
  Upper u2 {make_dense_writable_matrix_from<M22>(3, 1, 0, 3)};
  EXPECT_TRUE(is_near(u1, u2));
  //
  EXPECT_TRUE(is_near(Lower(DiagonalAdapter {3., 4}), make_dense_writable_matrix_from<M22>(3, 0, 0, 4)));
  EXPECT_TRUE(is_near(Upper(DiagonalAdapter {3., 4}), make_dense_writable_matrix_from<M22>(3, 0, 0, 4)));
  //
  EXPECT_TRUE(is_near(Lower(make_zero_matrix_like<M22>()), M22::Zero()));
  EXPECT_TRUE(is_near(Upper(make_zero_matrix_like<M22>()), M22::Zero()));
  //
  EXPECT_EQ(Lower::rows(), 2);
  EXPECT_EQ(Lower::cols(), 2);
  EXPECT_TRUE(is_near(make_zero_matrix_like<Lower>(), M22::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<Upper>(), M22::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Lower>(), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Upper>(), M22::Identity()));
  //
  Lower l3(l2); // copy constructor
  EXPECT_TRUE(is_near(l3, ml));
  Upper u3(u2); // copy constructor
  EXPECT_TRUE(is_near(u3, mu));
  //
  Lower l4 = Lower {3, 0, 1, 3}; // move constructor
  EXPECT_TRUE(is_near(l4, ml));
  Upper u4 = Upper{3, 1, 0, 3}; // move constructor
  EXPECT_TRUE(is_near(u4, mu));
  //
  Lower l5 = TriangularAdapter<decltype(M22::Zero()), TriangleType::lower>(M22::Zero()); // compatible triangular matrix
  EXPECT_TRUE(is_near(l5, M22::Zero()));
  Upper u5 = TriangularAdapter<decltype(M22::Zero()), TriangleType::upper>(M22::Zero()); // compatible triangular matrix
  EXPECT_TRUE(is_near(u5, M22::Zero()));
  //
  Lower l7 {ml.triangularView<Eigen::Lower>()};
  EXPECT_TRUE(is_near(l7, ml));
  Upper u7 {mu.triangularView<Eigen::Upper>()};
  EXPECT_TRUE(is_near(u7, mu));
  //
  Lower l9 = Diagonal {3., 3}; // Construct from a diagonal triangular matrix.
  EXPECT_TRUE(is_near(l9, mat22(3, 0, 0, 3)));
  Upper u9 = Diagonal {3., 3};
  EXPECT_TRUE(is_near(u9, mat22(3, 0, 0, 3)));
  Diagonal d9 = Diagonal {3., 3};
  EXPECT_TRUE(is_near(d9, mat22(3, 0, 0, 3)));
  Diagonal d9_b = Diagonal2 {3., 3};
  EXPECT_TRUE(is_near(d9_b, mat22(3, 0, 0, 3)));
  Diagonal d9_c = Diagonal3 {3., 3};
  EXPECT_TRUE(is_near(d9_c, mat22(3, 0, 0, 3)));
  Diagonal2 d9b = Diagonal {3., 3};
  EXPECT_TRUE(is_near(d9b, mat22(3, 0, 0, 3)));
  Diagonal2 d9b_b = Diagonal2 {3., 3};
  EXPECT_TRUE(is_near(d9b_b, mat22(3, 0, 0, 3)));
  Diagonal2 d9b_c = Diagonal3 {3., 3};
  EXPECT_TRUE(is_near(d9b_c, mat22(3, 0, 0, 3)));
  Diagonal3 d9c = Diagonal {3., 3};
  EXPECT_TRUE(is_near(d9c, mat22(3, 0, 0, 3)));
  Diagonal3 d9c_b = Diagonal2 {3., 3};
  EXPECT_TRUE(is_near(d9c_b, mat22(3, 0, 0, 3)));
  Diagonal3 d9c_c = Diagonal3 {3., 3};
  EXPECT_TRUE(is_near(d9c_c, mat22(3, 0, 0, 3)));
  //
  Diagonal d10 {3, 3}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d10, mat22(3, 0, 0, 3)));
  Diagonal2 d10b {3, 3};
  EXPECT_TRUE(is_near(d10b, mat22(3, 0, 0, 3)));
  Diagonal3 d10c {3, 3};
  EXPECT_TRUE(is_near(d10c, mat22(3, 0, 0, 3)));
  //
  l3 = l5; // copy assignment
  EXPECT_TRUE(is_near(l3, M22::Zero()));
  u3 = u5; // copy assignment
  EXPECT_TRUE(is_near(u3, M22::Zero()));
  //
  l5 = Lower {3., 0, 1, 3}; // move assignment
  EXPECT_TRUE(is_near(l5, ml));
  u5 = Upper {3., 1, 0, 3}; // move assignment
  EXPECT_TRUE(is_near(u5, mu));
  //
  l2 = TriangularAdapter<decltype(M22::Zero()), TriangleType::lower>(M22::Zero()); // copy assignment from compatible triangular matrix
  EXPECT_TRUE(is_near(l2, M22::Zero()));
  u2 = TriangularAdapter<decltype(M22::Zero()), TriangleType::upper>(M22::Zero()); // copy assignment from compatible triangular matrix
  EXPECT_TRUE(is_near(u2, M22::Zero()));
  //
  l9 = Diagonal {4., 4}; // copy from a diagonal triangular matrix
  EXPECT_TRUE(is_near(l9, mat22(4, 0, 0, 4)));
  l4 = l9;
  EXPECT_TRUE(is_near(l4, mat22(4, 0, 0, 4)));
  l9 = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(l9, mat22(5, 0, 0, 5)));
  l9 = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(l9, mat22(6, 0, 0, 6)));
  u9 = Diagonal {4., 4};
  EXPECT_TRUE(is_near(u9, mat22(4, 0, 0, 4)));
  u4 = u9;
  EXPECT_TRUE(is_near(u4, mat22(4, 0, 0, 4)));
  u9 = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(u9, mat22(5, 0, 0, 5)));
  u9 = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(u9, mat22(6, 0, 0, 6)));
  d9 = Diagonal {4., 4};
  EXPECT_TRUE(is_near(d9, mat22(4, 0, 0, 4)));
  d9 = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(d9, mat22(5, 0, 0, 5)));
  d9 = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(d9, mat22(6, 0, 0, 6)));
  d9b_b = Diagonal {7., 7};
  d9b = d9b_b;
  EXPECT_TRUE(is_near(d9b, mat22(7, 0, 0, 7)));
  l4 = d9b;
  EXPECT_TRUE(is_near(l4, mat22(7, 0, 0, 7)));
  u4 = d9b;
  EXPECT_TRUE(is_near(u4, mat22(7, 0, 0, 7)));
  d9b = Diagonal {4., 4};
  EXPECT_TRUE(is_near(d9b, mat22(4, 0, 0, 4)));
  d9b = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(d9b, mat22(5, 0, 0, 5)));
  d9b = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(d9b, mat22(6, 0, 0, 6)));
  d9c = d9b_b;
  EXPECT_TRUE(is_near(d9c, mat22(7, 0, 0, 7)));
  l4 = d9c;
  EXPECT_TRUE(is_near(l4, mat22(7, 0, 0, 7)));
  u4 = d9c;
  EXPECT_TRUE(is_near(u4, mat22(7, 0, 0, 7)));
  d9c = Diagonal {4., 4};
  EXPECT_TRUE(is_near(d9c, mat22(4, 0, 0, 4)));
  d9c = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(d9c, mat22(5, 0, 0, 5)));
  d9c = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(d9c, mat22(6, 0, 0, 6)));
  //
  l3 = make_zero_matrix_like<M22>();
  EXPECT_TRUE(is_near(l3, make_zero_matrix_like<M22>()));
  u3 = make_zero_matrix_like<M22>();
  EXPECT_TRUE(is_near(u3, make_zero_matrix_like<M22>()));
  l3 = make_identity_matrix_like<M22>();
  EXPECT_TRUE(is_near(l3, make_identity_matrix_like<M22>()));
  u3 = make_identity_matrix_like<M22>();
  EXPECT_TRUE(is_near(u3, make_identity_matrix_like<M22>()));
  //
  auto tl = ml.triangularView<Eigen::Lower>();
  auto tu = mu.triangularView<Eigen::Upper>();
  l2 = tl; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(l2, ml));
  u2 = tu; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(u2, mu));
  //
  l4 = ml.triangularView<Eigen::Lower>(); // assign from rvalue reference to TriangularBase derived object
  EXPECT_TRUE(is_near(l4, ml));
  u4 = mu.triangularView<Eigen::Upper>(); // assign from rvalue reference to TriangularBase derived object
  EXPECT_TRUE(is_near(u4, mu));
  //
  l4 = ZeroAdapter<eigen_matrix_t<double, 2, 2>> {};
  EXPECT_TRUE(is_near(l4, ZeroAdapter<eigen_matrix_t<double, 2, 2>> {}));
  u4 = ZeroAdapter<eigen_matrix_t<double, 2, 2>> {};
  EXPECT_TRUE(is_near(u4, ZeroAdapter<eigen_matrix_t<double, 2, 2>> {}));
  //
  l4 = M22::Identity();
  EXPECT_TRUE(is_near(l4, M22::Identity()));
  u4 = M22::Identity();
  EXPECT_TRUE(is_near(u4, M22::Identity()));
  //
  l4 = d2;
  EXPECT_TRUE(is_near(l4, d2));
  u4 = d2;
  EXPECT_TRUE(is_near(u4, d2));
  //
  l4 = {3, 0, 1, 3}; // assign from a list of scalars
  EXPECT_TRUE(is_near(l4, ml));
  u4 = {3, 1, 0, 3}; // assign from a list of scalars
  EXPECT_TRUE(is_near(u4, mu));
  //
  l1 += l2;
  EXPECT_TRUE(is_near(l1, mat22(6., 0, 2, 6)));
  u1 += u2;
  EXPECT_TRUE(is_near(u1, mat22(6., 2, 0, 6)));
  //
  l1 -= TriangularAdapter {3., 0, 1, 3};
  EXPECT_TRUE(is_near(l1, mat22(3., 0, 1, 3)));
  u1 -= Upper {3., 1, 0, 3};
  EXPECT_TRUE(is_near(u1, mat22(3., 1, 0, 3)));
  //
  l1 *= 3;
  EXPECT_TRUE(is_near(l1, mat22(9., 0, 3, 9)));
  u1 *= 3;
  EXPECT_TRUE(is_near(u1, mat22(9., 3, 0, 9)));
  //
  l1 /= 3;
  EXPECT_TRUE(is_near(l1, mat22(3., 0, 1, 3)));
  u1 /= 3;
  EXPECT_TRUE(is_near(u1, mat22(3., 1, 0, 3)));
  //
  l2 *= l1;
  EXPECT_TRUE(is_near(l2, mat22(9., 0, 6, 9)));
  u2 *= u1;
  EXPECT_TRUE(is_near(u2, mat22(9., 6, 0, 9)));
  //
  EXPECT_TRUE(is_near(l1, mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(u1, mat22(3., 1, 0, 3)));
}


TEST(special_matrices, TriangularAdapter_subscripts)
{
  static_assert(element_gettable<Lower, 2>);
  static_assert(not element_gettable<Lower, 1>);
  static_assert(element_gettable<Upper, 2>);
  static_assert(not element_gettable<Upper, 1>);
  static_assert(element_gettable<Diagonal, 2>);
  static_assert(element_gettable<Diagonal, 1>);
  static_assert(element_gettable<Diagonal2, 2>);
  static_assert(element_gettable<Diagonal2, 1>);
  static_assert(element_gettable<Diagonal3, 2>);
  static_assert(element_gettable<Diagonal3, 1>);

  static_assert(element_settable<Lower&, 2>);
  static_assert(not element_settable<Lower&, 1>);
  static_assert(element_settable<Upper&, 2>);
  static_assert(not element_settable<Upper&, 1>);
  static_assert(element_settable<Diagonal&, 2>);
  static_assert(element_settable<Diagonal&, 1>);
  static_assert(element_settable<Diagonal2&, 2>);
  static_assert(element_settable<Diagonal2&, 1>);
  static_assert(element_settable<Diagonal3&, 2>);
  static_assert(element_settable<Diagonal3&, 1>);

  static_assert(not element_settable<const Lower&, 2>);
  static_assert(not element_settable<const Lower&, 1>);
  static_assert(not element_settable<const Upper&, 2>);
  static_assert(not element_settable<const Upper&, 1>);
  static_assert(not element_settable<const Diagonal&, 2>);
  static_assert(not element_settable<const Diagonal&, 1>);
  static_assert(not element_settable<const Diagonal2&, 2>);
  static_assert(not element_settable<const Diagonal2&, 1>);
  static_assert(not element_settable<const Diagonal3&, 2>);
  static_assert(not element_settable<const Diagonal3&, 1>);

  static_assert(not element_settable<TriangularAdapter<const M22, TriangleType::lower>&, 2>);
  static_assert(not element_settable<TriangularAdapter<const D2, TriangleType::lower>&, 2>);
  static_assert(not element_settable<TriangularAdapter<const D2, TriangleType::lower>&, 1>);
  static_assert(not element_settable<TriangularAdapter<DiagonalAdapter<const eigen_matrix_t<double, 2, 1>>, TriangleType::lower>&, 2>);
  static_assert(not element_settable<TriangularAdapter<DiagonalAdapter<const eigen_matrix_t<double, 2, 1>>, TriangleType::lower>&, 1>);

  auto l1 = Lower {3, 0, 1, 3};
  set_element(l1, 1.1, 1, 0);
  EXPECT_NEAR(get_element(l1, 1, 0), 1.1, 1e-8);
  bool test = false;
  try { set_element(l1, 2.1, 0, 1); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  set_element(l1, 0, 0, 1);
  EXPECT_NEAR(get_element(l1, 0, 1), 0, 1e-8);

  auto u1 = Upper {3, 1, 0, 3};
  set_element(u1, 1.1, 0, 1);
  EXPECT_NEAR(get_element(u1, 0, 1), 1.1, 1e-8);
  test = false;
  try { set_element(u1, 2.1, 1, 0); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  set_element(u1, 0, 1, 0);
  EXPECT_NEAR(get_element(u1, 1, 0), 0, 1e-8);

  EXPECT_EQ(l1(0, 1), 0);
  EXPECT_EQ(u1(1, 0), 0);
  //
  EXPECT_EQ(l1(1, 1), 3);
  EXPECT_EQ(u1(1, 1), 3);
  //
  l1(0, 0) = 5;
  EXPECT_NEAR(l1(0, 0), 5, 1e-8);
  l1(1, 0) = 6;
  EXPECT_NEAR(l1(1, 0), 6, 1e-8);
  l1(1, 1) = 8;
  test = false;
  try { l1(0, 1) = 7; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  l1(0, 1) = 0;
  EXPECT_TRUE(is_near(l1, mat22(5, 0, 6, 8)));

  u1(0, 0) = 5;
  EXPECT_NEAR(u1(0, 0), 5, 1e-8);
  u1(0, 1) = 6;
  EXPECT_NEAR(u1(0, 1), 6, 1e-8);
  u1(1, 1) = 8;
  test = false;
  try { u1(1, 0) = 7; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  u1(1, 0) = 0;
  EXPECT_TRUE(is_near(u1, mat22(5, 6, 0, 8)));
  //
  auto d9 = Diagonal {9, 10};
  d9(0, 0) = 7.1;
  EXPECT_NEAR(d9(0), 7.1, 1e-8);
  d9(1) = 8.1;
  EXPECT_NEAR(d9(1, 1), 8.1, 1e-8);
  test = false;
  try { d9(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  d9(1, 0) = 0;
  EXPECT_TRUE(is_near(d9, mat22(7.1, 0, 0, 8.1)));

  auto d9b = Diagonal2 {9, 10};
  d9b(0, 0) = 7.1;
  EXPECT_NEAR(d9b(0, 0), 7.1, 1e-8);
  d9b(1, 1) = 8.1;
  EXPECT_NEAR(d9b(1, 1), 8.1, 1e-8);
  test = false;
  try { d9b(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  d9b(1, 0) = 0;
  EXPECT_TRUE(is_near(d9b, mat22(7.1, 0, 0, 8.1)));

  auto d9c = Diagonal3 {9, 10};
  d9c(0, 0) = 7.1;
  EXPECT_NEAR(d9c(0, 0), 7.1, 1e-8);
  d9c(1, 1) = 8.1;
  EXPECT_NEAR(d9c(1, 1), 8.1, 1e-8);
  test = false;
  try { d9c(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  d9c(1, 0) = 0;
  EXPECT_TRUE(is_near(d9c, mat22(7.1, 0, 0, 8.1)));
  //
  EXPECT_NEAR((TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::diagonal> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::lower> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::upper> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((Diagonal {2, 3})(0), 2, 1e-6);
  EXPECT_NEAR((Diagonal {2, 3})(1), 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})(0), 2, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})(1), 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})(0), 2, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})(1), 3, 1e-6);
  EXPECT_NEAR((Diagonal {2, 3})[0], 2, 1e-6);
  EXPECT_NEAR((Diagonal {2, 3})[1], 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})[0], 2, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})[1], 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})[0], 2, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})[1], 3, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal {3., 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(0, 0), 3, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(0, 1), 1, 1e-6);
  EXPECT_NEAR((Diagonal {3., 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(0, 1), 0, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(1, 0), 1, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal {3., 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(1, 0), 0, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal {3., 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(1, 1), 3, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 4}).nested_matrix()(0, 0), 3, 1e-6);
  EXPECT_NEAR((Lower {3., 0, 1, 4}).nested_matrix()(1, 0), 1, 1e-6);
  EXPECT_NEAR((Lower {3., 0, 1, 4}).nested_matrix()(1, 1), 4, 1e-6);
}


TEST(special_matrices, TriangularAdapter_view)
{
  Lower l1 {3, 0, 1, 3};
  EXPECT_TRUE(is_near(M22 {l1.view()}, mat22(3, 0, 1, 3)));
  EXPECT_TRUE(is_near(M22 {std::as_const(l1).view()}, mat22(3, 0, 1, 3)));
  EXPECT_TRUE(is_near(M22 {Lower {3, 0, 1, 3}.view()}, mat22(3, 0, 1, 3)));
  EXPECT_TRUE(is_near(M22 {const_cast<const Lower&&>(Lower {3, 0, 1, 3}).view()}, mat22(3, 0, 1, 3)));

  EXPECT_TRUE(is_near(l1.view() * make_dense_writable_matrix_from<M22>(3, 1, 0, 3), mat22(9, 3, 3, 10)));
  EXPECT_TRUE(is_near(std::as_const(l1).view() * make_dense_writable_matrix_from<M22>(3, 1, 0, 3), mat22(9, 3, 3, 10)));
  EXPECT_TRUE(is_near(Lower {3, 0, 1, 3}.view() * make_dense_writable_matrix_from<M22>(3, 1, 0, 3), mat22(9, 3, 3, 10)));
  EXPECT_TRUE(is_near(const_cast<const Lower&&>(Lower {3, 0, 1, 3}).view() * make_dense_writable_matrix_from<M22>(3, 1, 0, 3), mat22(9, 3, 3, 10)));

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(3, 1, 0, 3) * l1.view(), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(3, 1, 0, 3) * std::as_const(l1).view(), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(3, 1, 0, 3) * Lower {3, 0, 1, 3}.view(), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(3, 1, 0, 3) * const_cast<const Lower&&>(Lower {3, 0, 1, 3}).view(), mat22(10, 3, 3, 9)));

  Upper u1 {3, 1, 0, 3};
  EXPECT_TRUE(is_near(M22 {u1.view()}, mat22(3, 1, 0, 3)));
  EXPECT_TRUE(is_near(M22 {std::as_const(u1).view()}, mat22(3, 1, 0, 3)));
  EXPECT_TRUE(is_near(M22 {Upper {3, 1, 0, 3}.view()}, mat22(3, 1, 0, 3)));
  EXPECT_TRUE(is_near(M22 {const_cast<const Upper&&>(Upper {3, 1, 0, 3}).view()}, mat22(3, 1, 0, 3)));

  EXPECT_TRUE(is_near(u1.view() * make_dense_writable_matrix_from<M22>(3, 0, 1, 3), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(std::as_const(u1).view() * make_dense_writable_matrix_from<M22>(3, 0, 1, 3), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(Upper {3, 1, 0, 3}.view() * make_dense_writable_matrix_from<M22>(3, 0, 1, 3), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(const_cast<const Upper&&>(Upper {3, 1, 0, 3}).view() * make_dense_writable_matrix_from<M22>(3, 0, 1, 3), mat22(10, 3, 3, 9)));

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(3, 0, 1, 3) * u1.view(), mat22(9, 3, 3, 10)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(3, 0, 1, 3) * std::as_const(u1).view(), mat22(9, 3, 3, 10)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(3, 0, 1, 3) * Upper {3, 1, 0, 3}.view(), mat22(9, 3, 3, 10)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from<M22>(3, 0, 1, 3) * const_cast<const Upper&&>(Upper {3, 1, 0, 3}).view(), mat22(9, 3, 3, 10)));
}


TEST(special_matrices, TriangularAdapter_make)
{
  static_assert(zero_matrix<decltype(make_triangular_matrix<TriangleType::upper>(make_zero_matrix_like<M22>()))>);
  static_assert(zero_matrix<decltype(make_triangular_matrix<TriangleType::lower>(make_zero_matrix_like<M22>()))>);
  static_assert(zero_matrix<decltype(make_triangular_matrix(make_zero_matrix_like<M22>()))>);
  static_assert(triangular_matrix<decltype(make_triangular_matrix<TriangleType::upper>(make_dense_writable_matrix_from<M22>(3, 1, 0, 3))), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(make_triangular_matrix<TriangleType::lower>(make_dense_writable_matrix_from<M22>(3, 0, 1, 3))), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_triangular_matrix(make_dense_writable_matrix_from<M22>(3, 0, 1, 3))), TriangleType::lower>);
  static_assert(diagonal_matrix<decltype(make_triangular_matrix<TriangleType::upper>(DiagonalAdapter {3., 4}))>);
  static_assert(diagonal_matrix<decltype(make_triangular_matrix<TriangleType::lower>(DiagonalAdapter {3., 4}))>);
  static_assert(diagonal_matrix<decltype(make_triangular_matrix(DiagonalAdapter {3., 4}))>);

  static_assert(triangular_matrix<decltype(make_triangular_matrix<TriangleType::upper>(Upper {3, 1, 0, 3})), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(make_triangular_matrix<TriangleType::lower>(Lower {3, 0, 1, 3})), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_triangular_matrix(Upper {3, 1, 0, 3})), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(make_triangular_matrix(Lower {3, 0, 1, 3})), TriangleType::lower>);
}


TEST(special_matrices, TriangularAdapter_traits)
{
  M22 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  using Dl = TriangularAdapter<M22, TriangleType::lower>;
  using Du = TriangularAdapter<M22, TriangleType::upper>;
  //
  //
  EXPECT_TRUE(is_near(make_zero_matrix_like<Dl>(), M22::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<Du>(), M22::Zero()));
  //
  EXPECT_TRUE(is_near(make_identity_matrix_like<Dl>(), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Du>(), M22::Identity()));

  static_assert(triangular_matrix<Lower, TriangleType::lower>);
  static_assert(triangular_matrix<Upper, TriangleType::upper>);
  static_assert(diagonal_matrix<Diagonal>);
  static_assert(diagonal_matrix<Diagonal2>);
  static_assert(diagonal_matrix<Diagonal3>);
  static_assert(zero_matrix<decltype(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>()))>);
  static_assert(zero_matrix<decltype(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>()))>);
  static_assert(identity_matrix<decltype(TriangularAdapter<decltype(make_identity_matrix_like<M22>()), TriangleType::lower>(make_identity_matrix_like<M22>()))>);
}


TEST(special_matrices, TriangularAdapter_overloads)
{
  M22 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  //
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Lower(3., 0, 1, 3)), ml));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(Upper(3., 1, 0, 3)), mu));
  //
  EXPECT_TRUE(is_near(make_self_contained(Lower(M22::Zero())), make_dense_writable_matrix_from<M22>(0, 0, 0, 0)));
  EXPECT_TRUE(is_near(make_self_contained(Upper(M22::Zero())), make_dense_writable_matrix_from<M22>(0, 0, 0, 0)));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Lower {9, 3, 3, 10} * 2))>, Lower>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Upper {9, 3, 3, 10} * 2))>, Upper>);
  //
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(4))), eigen_matrix_t<double, 1, 1>(16)));
  static_assert(one_by_one_matrix<decltype(Cholesky_square(TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(4))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<decltype(M22::Identity()), TriangleType::lower>(M22::Identity())), M22::Identity()));
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<decltype(M22::Identity()), TriangleType::upper>(M22::Identity())), M22::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_square(TriangularAdapter<decltype(M22::Identity()), TriangleType::lower>(M22::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_square(TriangularAdapter<decltype(M22::Identity()), TriangleType::upper>(M22::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>())), M22::Zero()));
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>())), M22::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_square(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>())))>);
  static_assert(zero_matrix<decltype(Cholesky_square(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<decltype(DiagonalAdapter{2., 3}), TriangleType::lower>(DiagonalAdapter{2., 3})), DiagonalAdapter{4., 9}));
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<decltype(DiagonalAdapter{2., 3}), TriangleType::upper>(DiagonalAdapter{2., 3})), DiagonalAdapter{4., 9}));
  static_assert(internal::diagonal_expr<decltype(Cholesky_square(TriangularAdapter<decltype(DiagonalAdapter{2., 3}), TriangleType::lower>(DiagonalAdapter{2., 3})))>);
  static_assert(internal::diagonal_expr<decltype(Cholesky_square(TriangularAdapter<decltype(DiagonalAdapter{2., 3}), TriangleType::upper>(DiagonalAdapter{2., 3})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<M22, TriangleType::diagonal>(ml)), DiagonalAdapter{9., 9}));
  static_assert(internal::diagonal_expr<decltype(Cholesky_square(TriangularAdapter<M22, TriangleType::diagonal>(ml)))>);
  //
  Lower lower {3., 0, 1, 3};
  EXPECT_TRUE(is_near(Cholesky_square(lower), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(Cholesky_square(Lower {3., 0, 1, 3}), mat22(9., 3, 3, 10)));
  static_assert(hermitian_adapter<decltype(Cholesky_square(Lower {3, 0, 1, 3})), HermitianAdapterType::lower>);
  //
  Upper upper {3., 1, 0, 3};
  EXPECT_TRUE(is_near(Cholesky_square(upper), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(Cholesky_square(Upper {3., 1, 0, 3}), mat22(9., 3, 3, 10)));
  static_assert(hermitian_adapter<decltype(Cholesky_square(Upper {3, 1, 0, 3})), HermitianAdapterType::upper>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(81)));
  EXPECT_TRUE(is_near(Cholesky_square(TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::upper>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(81)));
  //
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(4))), eigen_matrix_t<double, 1, 1>(2)));
  static_assert(one_by_one_matrix<decltype(Cholesky_factor(TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(4))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<decltype(M22::Identity()), TriangleType::lower>(M22::Identity())), M22::Identity()));
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<decltype(M22::Identity()), TriangleType::upper>(M22::Identity())), M22::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_factor(TriangularAdapter<decltype(M22::Identity()), TriangleType::lower>(M22::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_factor(TriangularAdapter<decltype(M22::Identity()), TriangleType::upper>(M22::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>())), M22::Zero()));
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>())), M22::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_factor(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>())))>);
  static_assert(zero_matrix<decltype(Cholesky_factor(TriangularAdapter<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<decltype(DiagonalAdapter{4., 9}), TriangleType::lower>(DiagonalAdapter{4., 9})), DiagonalAdapter{2., 3}));
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<decltype(DiagonalAdapter{4., 9}), TriangleType::upper>(DiagonalAdapter{4., 9})), DiagonalAdapter{2., 3}));
  static_assert(internal::diagonal_expr<decltype(Cholesky_factor(TriangularAdapter<decltype(DiagonalAdapter{4., 9}), TriangleType::lower>(DiagonalAdapter{4., 9})))>);
  static_assert(internal::diagonal_expr<decltype(Cholesky_factor(TriangularAdapter<decltype(DiagonalAdapter{4., 9}), TriangleType::upper>(DiagonalAdapter{4., 9})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<M22, TriangleType::diagonal>(ml)), DiagonalAdapter{std::sqrt(3.), std::sqrt(3.)}));
  static_assert(internal::diagonal_expr<decltype(Cholesky_factor(TriangularAdapter<M22, TriangleType::diagonal>(ml)))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(3)));
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularAdapter<eigen_matrix_t<double, 1, 1>, TriangleType::upper>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(3)));
  //
  //
  EXPECT_TRUE(is_near(diagonal_of(Lower {3., 0, 1, 3}), make_eigen_matrix<double, 2, 1>(3, 3)));
  EXPECT_TRUE(is_near(diagonal_of(Upper {3., 1, 0, 3}), make_eigen_matrix<double, 2, 1>(3, 3)));
  //
  EXPECT_TRUE(is_near(transpose(Lower {3., 0, 1, 3}), mu));
  EXPECT_TRUE(is_near(transpose(Upper {3., 1, 0, 3}), ml));
  EXPECT_TRUE(is_near(transpose(CLower {cdouble(3,1), 0, cdouble(1,2), cdouble(3,-1)}), CUpper {cdouble(3,1), cdouble(1,2), 0, cdouble(3,-1)}));
  EXPECT_TRUE(is_near(transpose(CUpper {cdouble(3,1), cdouble(1,2), 0, cdouble(3,-1)}), CLower {cdouble(3,1), 0, cdouble(1,2), cdouble(3,-1)}));
  //
  EXPECT_TRUE(is_near(adjoint(Lower {3., 0, 1, 3}), mu));
  EXPECT_TRUE(is_near(adjoint(Upper {3., 1, 0, 3}), ml));
  EXPECT_TRUE(is_near(adjoint(CLower {cdouble(3,1), 0, cdouble(1,2), cdouble(3,-1)}), CUpper {cdouble(3,-1), cdouble(1,-2), 0, cdouble(3,1)}));
  EXPECT_TRUE(is_near(adjoint(CUpper {cdouble(3,1), cdouble(1,2), 0, cdouble(3,-1)}), CLower {cdouble(3,-1), 0, cdouble(1,-2), cdouble(3,1)}));
  //
  EXPECT_NEAR(determinant(Lower {3., 0, 1, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(Upper {3., 1, 0, 3}), 9, 1e-6);
  //
  EXPECT_NEAR(trace(Lower {3., 0, 1, 3}), 6, 1e-6);
  EXPECT_NEAR(trace(Upper {3., 1, 0, 3}), 6, 1e-6);
  //
  EXPECT_TRUE(is_near(average_reduce<1>(Lower {3., 0, 1, 3}), make_eigen_matrix(1.5, 2)));
  EXPECT_TRUE(is_near(average_reduce<1>(Upper {3., 1, 0, 3}), make_eigen_matrix(2, 1.5)));
  //
  EXPECT_TRUE(is_near(average_reduce<0>(Lower {3., 0, 1, 3}), make_eigen_matrix<double, 1, 2>(2, 1.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(Upper {3., 1, 0, 3}), make_eigen_matrix<double, 1, 2>(1.5, 2)));
}


TEST(special_matrices, TriangularAdapter_contract)
{
  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::Upper>(), m33.template triangularView<Eigen::Upper>()),
    make_dense_object_from<M33>(1, 12, 42, 0, 25, 84, 0, 0, 81)));
  static_assert(triangular_matrix<decltype(contract(m33.template triangularView<Eigen::Upper>(),
    m33.template triangularView<Eigen::Upper>())), TriangleType::upper>);
  EXPECT_TRUE(is_near(contract(m33.template triangularView<Eigen::Lower>(), m33.template triangularView<Eigen::Lower>()),
    make_dense_object_from<M33>(1, 0, 0, 24, 25, 0, 102, 112, 81)));
  static_assert(triangular_matrix<decltype(contract(m33.template triangularView<Eigen::Lower>(),
    m33.template triangularView<Eigen::Lower>())), TriangleType::lower>);

  EXPECT_TRUE(is_near(contract(m3x_3.template triangularView<Eigen::Upper>(), mx3_3.template triangularView<Eigen::Upper>()),
    make_dense_object_from<M33>(1, 12, 42, 0, 25, 84, 0, 0, 81)));
  static_assert(triangular_matrix<decltype(contract(m3x_3.template triangularView<Eigen::Upper>(),
    mx3_3.template triangularView<Eigen::Upper>())), TriangleType::upper>);
  EXPECT_TRUE(is_near(contract(m3x_3.template triangularView<Eigen::Lower>(), mx3_3.template triangularView<Eigen::Lower>()),
    make_dense_object_from<M33>(1, 0, 0, 24, 25, 0, 102, 112, 81)));
  static_assert(triangular_matrix<decltype(contract(m3x_3.template triangularView<Eigen::Lower>(),
    mx3_3.template triangularView<Eigen::Lower>())), TriangleType::lower>);
}


TEST(special_matrices, TriangularAdapter_solve)
{
  EXPECT_TRUE(is_near(solve(Lower {3., 0, 1, 3}, make_eigen_matrix<double, 2, 1>(3, 7)), make_eigen_matrix(1., 2)));
  EXPECT_TRUE(is_near(solve(Upper {3., 1, 0, 3}, make_eigen_matrix<double, 2, 1>(3, 9)), make_eigen_matrix(0., 3)));
  //
  EXPECT_TRUE(is_near(solve(Lower {3., 0, 1, 3}, make_eigen_matrix<double, 2, 1>(3, 7)), make_eigen_matrix<double, 2, 1>(1, 2)));
  EXPECT_TRUE(is_near(solve(Upper {3., 1, 0, 3}, make_eigen_matrix<double, 2, 1>(3, 9)), make_eigen_matrix<double, 2, 1>(0, 3)));

  auto m22_3104 = make_dense_object_from<M22>(3, 1, 0, 4);
  auto m2x_3104 = M2x {m22_3104};
  auto mx2_3104 = Mx2 {m22_3104};
  auto mxx_3104 = Mxx {m22_3104};

  auto m22_5206 = make_dense_object_from<M22>(5, 2, 0, 6);

  auto m22_1512024 = make_eigen_matrix<double, 2, 2>(15, 12, 0, 24);
  auto m2x_1512024 = M2x {m22_1512024};
  auto mx2_1512024 = Mx2 {m22_1512024};
  auto mxx_1512024 = Mxx {m22_1512024};

  static_assert(triangular_matrix<decltype(solve(m22_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>())), TriangleType::upper>);
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::Upper>(), m2x_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::Upper>(), mxx_1512024.template triangularView<Eigen::Upper>()), m22_5206));

  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::Upper>(), m2x_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::Upper>(), mxx_1512024.template triangularView<Eigen::Upper>()), m22_5206));

  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::Upper>(), m2x_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  static_assert(triangular_matrix<decltype(solve(mx2_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>())), TriangleType::upper>);
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::Upper>(), mxx_1512024.template triangularView<Eigen::Upper>()), m22_5206));

  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::Upper>(), m2x_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::Upper>(), mxx_1512024.template triangularView<Eigen::Upper>()), m22_5206));
}


TEST(special_matrices, TriangularAdapter_decompositions)
{
  EXPECT_TRUE(is_near(LQ_decomposition(Lower {3., 0, 1, 3}), mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(QR_decomposition(Upper {3., 1, 0, 3}), mat22(3., 1, 0, 3)));
  EXPECT_TRUE(is_near(Cholesky_square(LQ_decomposition(Upper {3., 1, 0, 3})), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(Cholesky_square(QR_decomposition(Lower {3., 0, 1, 3})), mat22(10, 3, 3, 9)));
}


TEST(special_matrices, TriangularAdapter_blocks_lower)
{
  auto ma = TriangularAdapter<eigen_matrix_t<double, 3, 3>, TriangleType::lower> {1, 0, 0,
                                                                                     2, 4, 0,
                                                                                     3, 5, 6};
  auto mb = TriangularAdapter<eigen_matrix_t<double, 3, 3>, TriangleType::lower> {4, 0, 0,
                                                                                     5, 7, 0,
                                                                                     6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3}, mb),
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}));
  static_assert(triangular_matrix<decltype(concatenate_diagonal(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3}, mb)), TriangleType::lower>);

  EXPECT_TRUE(is_near(concatenate_vertical(ma, mb),
    make_eigen_matrix<6,3>(1., 0, 0,
                                    2, 4, 0,
                                    3, 5, 6,
                                    4, 0, 0,
                                    5, 7, 0,
                                    6, 8, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(ma, mb),
    make_eigen_matrix<3, 6>(1., 0, 0, 4, 0, 0,
                                     2, 4, 0, 5, 7, 0,
                                     3, 5, 6, 6, 8, 9)));
  EXPECT_TRUE(is_near(split_diagonal(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3}, mb}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3},
               TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {4., 0, 5, 7}}));
  EXPECT_TRUE(is_near(split_vertical(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 0, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_eigen_matrix<3,5>(0., 0, 4, 0, 0,
                                               0, 0, 5, 7, 0,
                                               0, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 0, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_eigen_matrix<2,5>(0., 0, 4, 0, 0,
                                               0, 0, 5, 7, 0)}));
  EXPECT_TRUE(is_near(split_horizontal(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 0, 0,
                                                                              0, 0, 5, 7, 0,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 0, 2, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 0, 0, 5, 7, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 0, 2, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,2>(0., 0, 0, 0, 4, 0, 5, 7, 6, 8)}));

  EXPECT_TRUE(is_near(column(mb, 2), make_eigen_matrix(0., 0, 9)));
  EXPECT_TRUE(is_near(column<1>(mb), make_eigen_matrix(0., 7, 8)));
  EXPECT_TRUE(is_near(row(mb, 2), make_eigen_matrix<double, 1, 3>(6., 8, 9)));
  EXPECT_TRUE(is_near(row<1>(mb), make_eigen_matrix<double, 1, 3>(5., 7, 0)));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 1, 1,
      6, 8, 1,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 1, 2,
      5, 8, 2,
      6, 9, 11)));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 1, 1,
      6, 8, 1,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 0, 0,
      6, 8, 1,
      8, 10, 11)));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 1, 1,
      6, 8, 1,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 1, 2,
      6, 9, 3,
      8, 11, 13)));
}


TEST(special_matrices, TriangularAdapter_blocks_upper)
{
  auto ma = TriangularAdapter<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {1, 2, 3,
                                                                                     0, 4, 5,
                                                                                     0, 0, 6};
  auto mb = TriangularAdapter<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {4., 5, 6,
                                                                                     0, 7, 8,
                                                                                     0, 0, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3}, mb),
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}));
  static_assert(triangular_matrix<decltype(concatenate_diagonal(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3}, mb)), TriangleType::upper>);

  EXPECT_TRUE(is_near(concatenate_vertical(ma, mb),
    make_eigen_matrix<6,3>(1., 2, 3,
                                    0, 4, 5,
                                    0, 0, 6,
                                    4, 5, 6,
                                    0, 7, 8,
                                    0, 0, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(ma, mb),
    make_eigen_matrix<3,6>(1., 2, 3, 4, 5, 6,
                                    0, 4, 5, 0, 7, 8,
                                    0, 0, 6, 0, 0, 9)));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3}, mb}));
  EXPECT_TRUE(is_near(split_diagonal(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3},
               TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {4., 5, 0, 7}}));
  EXPECT_TRUE(is_near(split_vertical(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 2, 0, 0, 0,
                                               0, 3, 0, 0, 0),
               make_eigen_matrix<3,5>(0., 0, 4, 5, 6,
                                               0, 0, 0, 7, 8,
                                               0, 0, 0, 0, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 2, 0, 0, 0,
                                               0, 3, 0, 0, 0),
               make_eigen_matrix<2,5>(0., 0, 4, 5, 6,
                                               0, 0, 0, 7, 8)}));
  EXPECT_TRUE(is_near(split_horizontal(TriangularAdapter<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 2, 0, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 5, 6, 0, 7, 8, 0, 0, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    TriangularAdapter<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 2, 0, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,2>(0., 0, 0, 0, 4, 5, 0, 7, 0, 0)}));

  EXPECT_TRUE(is_near(column(mb, 2), make_eigen_matrix(6., 8, 9)));
  EXPECT_TRUE(is_near(column<1>(mb), make_eigen_matrix(5., 7, 0)));

  EXPECT_TRUE(is_near(row(mb, 2), make_eigen_matrix<double, 1, 3>(0., 0, 9)));
  EXPECT_TRUE(is_near(row<1>(mb), make_eigen_matrix<double, 1, 3>(0., 7, 8)));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 6, 7,
      1, 8, 9,
      1, 1, 10)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 6, 8,
      0, 8, 10,
      0, 1, 11)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 6, 7,
      1, 8, 9,
      1, 1, 10)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 6, 8,
      1, 9, 11,
      2, 3, 13)));
}


TEST(special_matrices, TriangularAdapter_arithmetic_lower)
{
  auto ma = Lower {4., 0, 5, 6};
  auto mb = Lower {1., 0, 2, 3};
  auto d = DiagonalAdapter<eigen_matrix_t<double, 2, 1>> {1, 3};
  auto i = M22::Identity();
  auto z = ZeroAdapter<eigen_matrix_t<double, 2, 2>> {};

  EXPECT_TRUE(is_near(ma + mb, mat22(5, 0, 7, 9))); static_assert(triangular_matrix<decltype(ma + mb), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma + d, mat22(5, 0, 5, 9))); static_assert(triangular_matrix<decltype(ma + d), TriangleType::lower>);
  EXPECT_TRUE(is_near(d + ma, mat22(5, 0, 5, 9))); static_assert(triangular_matrix<decltype(d + ma), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma + i, mat22(5, 0, 5, 7))); static_assert(triangular_matrix<decltype(ma + i), TriangleType::lower>);
  EXPECT_TRUE(is_near(i + ma, mat22(5, 0, 5, 7))); static_assert(triangular_matrix<decltype(i + ma), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma + z, mat22(4, 0, 5, 6))); static_assert(triangular_matrix<decltype(ma + z), TriangleType::lower>);
  EXPECT_TRUE(is_near(z + ma, mat22(4, 0, 5, 6))); static_assert(triangular_matrix<decltype(z + ma), TriangleType::lower>);

  EXPECT_TRUE(is_near(ma - mb, mat22(3, 0, 3, 3))); static_assert(triangular_matrix<decltype(ma - mb), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma - d, mat22(3, 0, 5, 3))); static_assert(triangular_matrix<decltype(ma - d), TriangleType::lower>);
  EXPECT_TRUE(is_near(d - ma, mat22(-3, 0, -5, -3))); static_assert(triangular_matrix<decltype(d - ma), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma - i, mat22(3, 0, 5, 5))); static_assert(triangular_matrix<decltype(ma - i), TriangleType::lower>);
  EXPECT_TRUE(is_near(i - ma, mat22(-3, 0, -5, -5))); static_assert(triangular_matrix<decltype(i - ma), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma - z, mat22(4, 0, 5, 6))); static_assert(triangular_matrix<decltype(ma - z), TriangleType::lower>);
  EXPECT_TRUE(is_near(z - ma, mat22(-4, 0, -5, -6))); static_assert(triangular_matrix<decltype(z - ma), TriangleType::lower>);

  EXPECT_TRUE(is_near(ma * 2, mat22(8, 0, 10, 12))); static_assert(triangular_matrix<decltype(ma * 2), TriangleType::lower>);
  EXPECT_TRUE(is_near(2 * ma, mat22(8, 0, 10, 12))); static_assert(triangular_matrix<decltype(2 * ma), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma / 2, mat22(2, 0, 2.5, 3))); static_assert(triangular_matrix<decltype(ma / 2), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(ma / 0), TriangleType::lower>);
  EXPECT_TRUE(is_near(-ma, mat22(-4, 0, -5, -6)));  static_assert(triangular_matrix<decltype(-ma), TriangleType::lower>);

  EXPECT_TRUE(is_near(TriangularAdapter<decltype(i), TriangleType::diagonal> {i} * 2, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(TriangularAdapter<decltype(i), TriangleType::diagonal> {i} * 2)>);
  EXPECT_TRUE(is_near(2 * TriangularAdapter<decltype(i), TriangleType::diagonal> {i}, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(2 * TriangularAdapter<decltype(i), TriangleType::diagonal> {i})>);
  EXPECT_TRUE(is_near(TriangularAdapter<decltype(i), TriangleType::diagonal> {i} / 0.5, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(TriangularAdapter<decltype(i), TriangleType::diagonal> {i} / 2)>);

  EXPECT_TRUE(is_near(ma * mb, mat22(4, 0, 17, 18))); static_assert(triangular_matrix<decltype(ma * mb), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma * d, mat22(4, 0, 5, 18))); static_assert(triangular_matrix<decltype(ma * d), TriangleType::lower>);
  EXPECT_TRUE(is_near(d * ma, mat22(4, 0, 15, 18))); static_assert(triangular_matrix<decltype(d * mb), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma * i, ma));  static_assert(triangular_matrix<decltype(ma * i), TriangleType::lower>);
  EXPECT_TRUE(is_near(i * ma, ma));  static_assert(triangular_matrix<decltype(i * ma), TriangleType::lower>);
  EXPECT_TRUE(is_near(ma * z, z));  static_assert(zero_matrix<decltype(ma * z)>);
  EXPECT_TRUE(is_near(z * ma, z));  static_assert(zero_matrix<decltype(z * ma)>);

  EXPECT_TRUE(is_near(mat22(4, 0, 5, 6) * mb, mat22(4, 0, 17, 18)));
  EXPECT_TRUE(is_near(ma * mat22(1, 0, 2, 3), mat22(4, 0, 17, 18)));
}


TEST(special_matrices, TriangularAdapter_arithmetic_upper)
{
  auto ma = Upper {4., 5, 0, 6};
  auto mb = Upper {1., 2, 0, 3};
  auto d = DiagonalAdapter<eigen_matrix_t<double, 2, 1>> {1, 3};
  auto i = M22::Identity();
  auto z = ZeroAdapter<eigen_matrix_t<double, 2, 2>> {};
  EXPECT_TRUE(is_near(ma + mb, mat22(5, 7, 0, 9))); static_assert(triangular_matrix<decltype(ma + mb), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma + d, mat22(5, 5, 0, 9))); static_assert(triangular_matrix<decltype(ma + d), TriangleType::upper>);
  EXPECT_TRUE(is_near(d + ma, mat22(5, 5, 0, 9))); static_assert(triangular_matrix<decltype(d + ma), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma + i, mat22(5, 5, 0, 7))); static_assert(triangular_matrix<decltype(ma + i), TriangleType::upper>);
  EXPECT_TRUE(is_near(i + ma, mat22(5, 5, 0, 7))); static_assert(triangular_matrix<decltype(i + ma), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma + z, mat22(4, 5, 0, 6))); static_assert(triangular_matrix<decltype(ma + z), TriangleType::upper>);
  EXPECT_TRUE(is_near(z + ma, mat22(4, 5, 0, 6))); static_assert(triangular_matrix<decltype(z + ma), TriangleType::upper>);

  EXPECT_TRUE(is_near(ma - mb, mat22(3, 3, 0, 3))); static_assert(triangular_matrix<decltype(ma - mb), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma - d, mat22(3, 5, 0, 3))); static_assert(triangular_matrix<decltype(ma - d), TriangleType::upper>);
  EXPECT_TRUE(is_near(d - ma, mat22(-3, -5, 0, -3))); static_assert(triangular_matrix<decltype(d - ma), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma - i, mat22(3, 5, 0, 5))); static_assert(triangular_matrix<decltype(ma - i), TriangleType::upper>);
  EXPECT_TRUE(is_near(i - ma, mat22(-3, -5, 0, -5))); static_assert(triangular_matrix<decltype(i - ma), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma - z, mat22(4, 5, 0, 6))); static_assert(triangular_matrix<decltype(ma - z), TriangleType::upper>);
  EXPECT_TRUE(is_near(z - ma, mat22(-4, -5, 0, -6))); static_assert(triangular_matrix<decltype(z - ma), TriangleType::upper>);

  EXPECT_TRUE(is_near(ma * 2, mat22(8, 10, 0, 12))); static_assert(triangular_matrix<decltype(ma * 2), TriangleType::upper>);
  EXPECT_TRUE(is_near(2 * ma, mat22(8, 10, 0, 12))); static_assert(triangular_matrix<decltype(2 * ma), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma / 2, mat22(2, 2.5, 0, 3))); static_assert(triangular_matrix<decltype(ma / 2), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(ma / 0), TriangleType::upper>);
  EXPECT_TRUE(is_near(-ma, mat22(-4, -5, 0, -6)));  static_assert(triangular_matrix<decltype(-ma), TriangleType::upper>);

  EXPECT_TRUE(is_near(ma * mb, mat22(4, 23, 0, 18))); static_assert(triangular_matrix<decltype(ma * mb), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma * d, mat22(4, 15, 0, 18))); static_assert(triangular_matrix<decltype(ma * d), TriangleType::upper>);
  EXPECT_TRUE(is_near(d * ma, mat22(4, 5, 0, 18))); static_assert(triangular_matrix<decltype(d * ma), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma * i, ma));  static_assert(triangular_matrix<decltype(ma * i), TriangleType::upper>);
  EXPECT_TRUE(is_near(i * ma, ma));  static_assert(triangular_matrix<decltype(i * ma), TriangleType::upper>);
  EXPECT_TRUE(is_near(ma * z, z));  static_assert(zero_matrix<decltype(ma * z)>);
  EXPECT_TRUE(is_near(z * ma, z));  static_assert(zero_matrix<decltype(z * ma)>);

  EXPECT_TRUE(is_near(mat22(4, 5, 0, 6) * mb, mat22(4, 23, 0, 18)));
  EXPECT_TRUE(is_near(ma * mat22(1, 2, 0, 3), mat22(4, 23, 0, 18)));
}


TEST(special_matrices, TriangularAdapter_arithmetic_mixed)
{
  auto m_upper = Upper {4., 5, 0, 6};
  auto m_lower = Lower {1., 0, 2, 3};
  EXPECT_TRUE(is_near(m_upper + m_lower, mat22(5, 5, 2, 9)));
  EXPECT_TRUE(is_near(m_lower + m_upper, mat22(5, 5, 2, 9)));
  EXPECT_TRUE(is_near(m_upper - m_lower, mat22(3, 5, -2, 3)));
  EXPECT_TRUE(is_near(m_lower - m_upper, mat22(-3, -5, 2, -3)));
  EXPECT_TRUE(is_near(m_upper * m_lower, mat22(14, 15, 12, 18)));
  EXPECT_TRUE(is_near(m_lower * m_upper, mat22(4, 5, 8, 28)));
}


TEST(special_matrices, TriangularAdapter_references)
{
  M22 m, n;
  m << 2, 0, 1, 2;
  n << 3, 0, 1, 3;
  using Tl = TriangularAdapter<M22, TriangleType::lower>;
  Tl x {m};
  TriangularAdapter<M22&, TriangleType::lower> x_lvalue {x};
  EXPECT_TRUE(is_near(x_lvalue, m));
  x = Tl {n};
  EXPECT_TRUE(is_near(x_lvalue, n));
  x_lvalue = Tl {m};
  EXPECT_TRUE(is_near(x, m));
  EXPECT_TRUE(is_near(TriangularAdapter<M22&, TriangleType::lower> {m}.nested_matrix(), mat22(2, 0, 1, 2)));
  //
  using V = TriangularAdapter<eigen_matrix_t<double, 3, 3>>;
  V v1 {1., 0, 0,
        2, 4, 0,
        3, -6, -3};
  bool test = false;
  try { v1(0, 1) = 3.2; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  v1(0, 1) = 0;
  EXPECT_EQ(v1(1,0), 2);
  TriangularAdapter<eigen_matrix_t<double, 3, 3>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  v2(2, 0) = 5.2;
  EXPECT_EQ(v1(2,0), 5.2);
  TriangularAdapter<eigen_matrix_t<double, 3, 3>> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  TriangularAdapter<const eigen_matrix_t<double, 3, 3>&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  TriangularAdapter<eigen_matrix_t<double, 3, 3>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}
