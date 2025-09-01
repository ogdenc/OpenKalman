/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests relating to diagonal_adapter.
 */

#include "adapters.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

namespace
{
  M51 m51 {make_dense_object_from<M51>(1, 2, 3, 4, 5)};

  M31 m31 {make_dense_object_from<M31>(1, 2, 3)};
  M3x m3x_1 {m31};
  Mx1 mx1_3 {m31};
  Mxx mxx_31 {m31};

  M13 m13 {make_dense_object_from<M13>(1, 2, 3)};
  M1x m1x_3 {m13};
  Mx3 mx3_1 {m13};
  Mxx mxx_13 {m13};

  M33 m33 {make_dense_object_from<M33>(1, 0, 0, 0, 2, 0, 0, 0, 3)};
  M33 m33b = make_dense_object_from<M33>(4, 0, 0, 0, 5, 0, 0, 0, 6);
  Mx3 mx3_3 {m33};
  Mxx mxx_33 {m33};

  M21 m21 {make_dense_object_from<M21>(1, 2)};
  M21 mx1_2 {m21};

  M11 m11 {make_dense_object_from<M11>(5)};
  M1x m1x_1 {m11};
  Mx1 mx1_1 {m11};
  Mxx mxx_11 {m11};

  using D5 = diagonal_adapter<M51>;
  using D3 = diagonal_adapter<M31>;
  using D2 = diagonal_adapter<M21>;
  using Dx = diagonal_adapter<Mx1>;

  D2 d2 {m21};
  Dx dx_2 {m21};
  D3 d3 {m31};
  Dx dx_3 {m31};
  D5 d5 {m51};
  Dx dx_5 {m51};
  diagonal_adapter<const M51> d5_const {m51};

  template<typename Mat> using D = diagonal_adapter<Mat>;
}


TEST(adapters, Diagonal_static_checks)
{
  static_assert(not writable<D<M31>>);
  static_assert(not writable<D<M31&>>);
  static_assert(not writable<D<const M31>>);
  static_assert(not writable<D<const M31&>>);
  
  static_assert(modifiable<D<M31>, zero_adapter<eigen_matrix_t<double, 3, 3>>>);
  static_assert(modifiable<D<M31>, Eigen3::IdentityMatrix<M33>>);
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


TEST(adapters, Diagonal_class)
{
  // default constructor and comma expression, .nested_object()
  D3 d3a;
  d3a << 1, 2, 3;
  EXPECT_TRUE(is_near(d3a, m33));
  EXPECT_TRUE(is_near(d3a.nested_object(), m31));

  // construct dynamic diagonal matrix from dynamic eigen matrix; fill with comma expression
  diagonal_adapter dxa_3 {Mx1 {make_dense_object_from<M31>(4, 5, 6)}};
  static_assert(std::is_same_v<decltype(dxa_3), Dx>);
  dxa_3 << 1, 2, 3;
  EXPECT_TRUE(is_near(dxa_3, m33));
  EXPECT_TRUE(is_near(dxa_3.nested_object(), make_dense_object_from<M31>(1, 2, 3)));

  // move constructor, deduction guide (column vector)
  diagonal_adapter d3b {diagonal_adapter {M31 {m31}}};
  static_assert(std::is_same_v<decltype(d3b), D3>);
  EXPECT_TRUE(is_near(d3b, m33));
  diagonal_adapter dxb_3 {Dx {M31 {m31}}}; // construct dynamic from fixed matrix, then move constructor
  static_assert(std::is_same_v<decltype(dxb_3), Dx>);
  EXPECT_TRUE(is_near(dxb_3, m33));
  EXPECT_TRUE(is_near(diagonal_adapter {Dx {D3 {m31}}}, m33)); // construct dynamic from fixed diagonal, then move constructor

  // copy constructor
  diagonal_adapter d3c {d3b};
  EXPECT_TRUE(is_near(d3c, m33));
  diagonal_adapter dxc_3 {dxb_3};
  EXPECT_TRUE(is_near(dxc_3, m33));

  // column scalar constructor
  D3 d3d {1., 2, 3};
  EXPECT_TRUE(is_near(d3d, m33));
  Dx dxd_3 {1., 2, 3};
  EXPECT_TRUE(is_near(dxd_3, m33));

  // column vector constructor and deduction guide
  EXPECT_TRUE(is_near(diagonal_adapter {m31}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {m3x_1}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {mx1_3}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {mxx_31}.nested_object(), m31));

  EXPECT_TRUE(is_near(diagonal_adapter {M31 {m31}}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {M3x {m3x_1}}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {Mx1 {mx1_3}}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {Mxx {mxx_31}}.nested_object(), m31));

  // diagonal constructor and diagonal deduction guide
  EXPECT_TRUE(is_near(diagonal_adapter {Eigen::DiagonalWrapper<M31> {m31}}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {Eigen::DiagonalWrapper<M3x> {m3x_1}}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {Eigen::DiagonalWrapper<Mx1> {mx1_3}}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {Eigen::DiagonalWrapper<Mxx> {mxx_31}}.nested_object(), m31));

  EXPECT_TRUE(is_near(diagonal_adapter {Eigen::DiagonalWrapper<M13> {m13}}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {Eigen::DiagonalWrapper<M1x> {m1x_3}}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {Eigen::DiagonalWrapper<Mx3> {mx3_1}}.nested_object(), m31));
  EXPECT_TRUE(is_near(diagonal_adapter {Eigen::DiagonalWrapper<Mxx> {mxx_13}}.nested_object(), m31));

  EXPECT_TRUE(is_near(diagonal_adapter {m11}.nested_object(), m11));
  EXPECT_TRUE(is_near(diagonal_adapter {m1x_1}.nested_object(), m11));
  EXPECT_TRUE(is_near(diagonal_adapter {mx1_1}.nested_object(), m11));
  EXPECT_TRUE(is_near(diagonal_adapter {mxx_11}.nested_object(), m11));

  EXPECT_TRUE(is_near(diagonal_adapter {M11 {m11}}.nested_object(), m11));
  EXPECT_TRUE(is_near(diagonal_adapter {M1x {m1x_1}}.nested_object(), m11));
  EXPECT_TRUE(is_near(diagonal_adapter {Mx1 {mx1_1}}.nested_object(), m11));
  EXPECT_TRUE(is_near(diagonal_adapter {Mxx {mxx_11}}.nested_object(), m11));

  // construct from zero matrix, and deduction guide (from non-diagonal_adapter diagonal)
  static_assert(zero<decltype(diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, 1>>{}})>);
  static_assert(diagonal_matrix<decltype(diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, 1>>{}})>);
  static_assert(square_shaped<decltype(diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, 1>>{}}), applicability::permitted> );
  static_assert(square_shaped<decltype(diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, 1>>{}})>);

  EXPECT_TRUE(is_near(diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, 1>>{}}, M33::Zero()));
  EXPECT_TRUE(is_near(diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, 3>>{}}, M33::Zero()));
  EXPECT_TRUE(is_near(diagonal_adapter {(diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, 3>>{}}).nested_object()}, M33::Zero()));
  EXPECT_TRUE(is_near(Dx {zero_adapter<eigen_matrix_t<double, 3, 1>>{}}, M33::Zero()));
  EXPECT_TRUE(is_near(Dx {zero_adapter<eigen_matrix_t<double, 3, 3>>{}}, M33::Zero()));
  EXPECT_TRUE(is_near(diagonal_adapter {(diagonal_adapter {zero_adapter<eigen_matrix_t<double, dynamic_size, dynamic_size>> {3, 3}}).nested_object()}, M33::Zero()));
  EXPECT_TRUE(is_near(diagonal_adapter {zero_adapter<eigen_matrix_t<double, dynamic_size, 1>>{3}}, M33::Zero()));
  EXPECT_TRUE(is_near(diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, dynamic_size>>{1}}, M33::Zero()));
  EXPECT_TRUE(is_near(diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, dynamic_size>>{3}}, M33::Zero()));
  EXPECT_TRUE(is_near(D3 {zero_adapter<eigen_matrix_t<double, dynamic_size, 3>>{3}}, M33::Zero()));
  EXPECT_TRUE(is_near(diagonal_adapter {zero_adapter<eigen_matrix_t<double, dynamic_size, 3>>{3}}, M33::Zero()));
  EXPECT_TRUE(is_near(D3 {zero_adapter<eigen_matrix_t<double, dynamic_size, dynamic_size>>{3, 3}}, M33::Zero()));
  EXPECT_TRUE(is_near(diagonal_adapter {zero_adapter<eigen_matrix_t<double, dynamic_size, dynamic_size>>{3, 3}}, M33::Zero()));

  // construct from identity matrix, and deduction guide (from non-diagonal_adapter diagonal)
  EXPECT_TRUE(is_near(diagonal_adapter {M33::Identity()}, M33::Identity()));
  EXPECT_TRUE(is_near(diagonal_adapter {0.7 * M33::Identity()}, M33::Identity() * 0.7));
  EXPECT_TRUE(is_near(diagonal_adapter {((0.7 * M33::Identity()) * (0.3 * M33::Identity() * 0.7 + 0.7 * M33::Identity()) - M33::Identity() * 0.3)}, M33::Identity() * 0.337));
  EXPECT_TRUE(is_near(diagonal_adapter {((0.7 * M33::Identity()) * (0.3 * M33::Identity() * 0.7 + 0.7 * M33::Identity()) - M33::Identity() * 0.3)}, M33::Identity() * 0.337));
  EXPECT_TRUE(is_near(diagonal_adapter {Mxx::Identity(3, 3)}, M33::Identity()));
  EXPECT_TRUE(is_near(diagonal_adapter {0.7 * Mxx::Identity(3, 3)}, Mxx::Identity(3, 3) * 0.7));

  M22 msa2 = make_dense_object_from<M22>(9, 0, 0, 10);

  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<M22, HermitianAdapterType::lower>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<Mxx, HermitianAdapterType::lower>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<M2x, HermitianAdapterType::lower>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<Mx2, HermitianAdapterType::lower>{msa2}}, msa2));

  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<D2, HermitianAdapterType::lower>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<Dx, HermitianAdapterType::lower>{msa2}}, msa2));

  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<D2, HermitianAdapterType::lower>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<Dx, HermitianAdapterType::lower>{msa2}}, msa2));

  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<M22, HermitianAdapterType::upper>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<Mxx, HermitianAdapterType::upper>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<M2x, HermitianAdapterType::upper>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<Mx2, HermitianAdapterType::upper>{msa2}}, msa2));

  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<D2, HermitianAdapterType::upper>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<Dx, HermitianAdapterType::upper>{msa2}}, msa2));

  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<D2, HermitianAdapterType::upper>{msa2}}, msa2));
  EXPECT_TRUE(is_near(diagonal_adapter {HermitianAdapter<Dx, HermitianAdapterType::upper>{msa2}}, msa2));

  M22 mt2 = make_dense_object_from<M22>(3, 0, 0, 3);

  EXPECT_TRUE(is_near(diagonal_adapter {TriangularAdapter<M22, triangle_type::diagonal>{mt2}}, mt2));
  EXPECT_TRUE(is_near(diagonal_adapter {TriangularAdapter<Mxx, triangle_type::diagonal>{mt2}}, mt2));
  EXPECT_TRUE(is_near(diagonal_adapter {TriangularAdapter<M2x, triangle_type::diagonal>{mt2}}, mt2));
  EXPECT_TRUE(is_near(diagonal_adapter {TriangularAdapter<Mx2, triangle_type::diagonal>{mt2}}, mt2));

  EXPECT_TRUE(is_near(diagonal_adapter {TriangularAdapter<D2, triangle_type::diagonal>{mt2}}, mt2));
  EXPECT_TRUE(is_near(diagonal_adapter {TriangularAdapter<Dx, triangle_type::diagonal>{mt2}}, mt2));

  EXPECT_TRUE(is_near(diagonal_adapter {TriangularAdapter<D2, triangle_type::lower>{mt2}}, mt2));
  EXPECT_TRUE(is_near(diagonal_adapter {TriangularAdapter<Dx, triangle_type::lower>{mt2}}, mt2));

  zero_adapter<eigen_matrix_t<double, 3, 3>> z33 {};

  // move assignment.
  d3c = {4., 5, 6};
  EXPECT_TRUE(is_near(d3c, m33b));
  dxc_3 = {4., 5, 6};
  EXPECT_TRUE(is_near(dxc_3, m33b));

  // copy assignment.
  d3d = d3c;
  EXPECT_TRUE(is_near(d3d, m33b));
  dxd_3 = dxc_3;
  EXPECT_TRUE(is_near(dxd_3, m33b));

  // assign from different-typed diagonal_adapter
  d3c = diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, 1>> {}};
  EXPECT_TRUE(is_near(d3c, z33));
  dxc_3 = diagonal_adapter {zero_adapter<eigen_matrix_t<double, dynamic_size, 1>> {3}};
  EXPECT_TRUE(is_near(dxc_3, z33));
  d3d = diagonal_adapter {zero_adapter<eigen_matrix_t<double, dynamic_size, 1>> {3}};
  EXPECT_TRUE(is_near(d3d, z33));
  dxd_3 = diagonal_adapter {zero_adapter<eigen_matrix_t<double, 3, 1>> {}};
  EXPECT_TRUE(is_near(dxd_3, z33));

  // assign from zero
  d3c = zero_adapter<eigen_matrix_t<double, 3, 3>> {};
  EXPECT_TRUE(is_near(d3c, z33));
  dxc_3 = zero_adapter<eigen_matrix_t<double, dynamic_size, dynamic_size>> {3, 3};
  EXPECT_TRUE(is_near(dxc_3, z33));
  d3d = zero_adapter<eigen_matrix_t<double, dynamic_size, dynamic_size>> {3, 3};
  EXPECT_TRUE(is_near(d3d, z33));
  dxd_3 = zero_adapter<eigen_matrix_t<double, 3, 3>> {};
  EXPECT_TRUE(is_near(dxd_3, z33));

  // assign from identity
  d3c = M33::Identity();
  EXPECT_TRUE(is_near(d3c, M33::Identity()));
  dxc_3 = Mxx::Identity(3, 3);
  EXPECT_TRUE(is_near(dxc_3, M33::Identity()));
  d3d = Mxx::Identity(3, 3);
  EXPECT_TRUE(is_near(d3d, M33::Identity()));
  dxd_3 = M33::Identity();
  EXPECT_TRUE(is_near(dxd_3, M33::Identity()));

  // Assign from general diagonal matrices:

  D2 d2a {1, 2};
  Dx d0a_2 {1, 2};
  M22 m22b = make_dense_object_from<M22>(9, 0, 0, 10);
  M22 m22c = make_dense_object_from<M22>(3, 0, 0, 3);

  d2a = TriangularAdapter<M22, triangle_type::diagonal>{m22c};
  EXPECT_TRUE(is_near(d2a, m22c));
  d0a_2 = TriangularAdapter<Mxx, triangle_type::diagonal>{m22c};
  EXPECT_TRUE(is_near(d0a_2, m22c));

  d2a = TriangularAdapter<D2, triangle_type::diagonal>{D2 {m22c}};
  EXPECT_TRUE(is_near(d2a, m22c));
  d0a_2 = TriangularAdapter<Dx, triangle_type::diagonal>{Dx {m22c}};
  EXPECT_TRUE(is_near(d0a_2, m22c));

  d2a = HermitianAdapter<D2, HermitianAdapterType::lower>{D2 {m22b}};
  EXPECT_TRUE(is_near(d2a, m22b));
  d0a_2 = HermitianAdapter<Dx, HermitianAdapterType::lower>{Dx {m22b}};
  EXPECT_TRUE(is_near(d0a_2, m22b));
  d2a = TriangularAdapter<D2, triangle_type::lower>{D2 {m22c}};
  EXPECT_TRUE(is_near(d2a, m22c));
  d0a_2 = TriangularAdapter<Dx, triangle_type::lower>{Dx {m22c}};
  EXPECT_TRUE(is_near(d0a_2, m22c));

  d2a = HermitianAdapter<D2, HermitianAdapterType::upper>{D2 {m22b}};
  EXPECT_TRUE(is_near(d2a, m22b));
  d0a_2 = HermitianAdapter<Dx, HermitianAdapterType::upper>{Dx {m22b}};
  EXPECT_TRUE(is_near(d0a_2, m22b));
  d2a = TriangularAdapter<D2, triangle_type::upper>{D2 {m22c}};
  EXPECT_TRUE(is_near(d2a, m22c));
  d0a_2 = TriangularAdapter<Dx, triangle_type::upper>{Dx {m22c}};
  EXPECT_TRUE(is_near(d0a_2, m22c));

  // Arithmetic

  d3a += d3b;
  EXPECT_TRUE(is_near(d3a, D3 {2., 4, 6}));
  dxa_3 += dxb_3;
  EXPECT_TRUE(is_near(dxa_3, D3 {2., 4, 6}));
  d3b += M33::Identity();
  EXPECT_TRUE(is_near(d3b, D3 {2., 3, 4}));
  dxb_3 += Mxx::Identity(3, 3);
  EXPECT_TRUE(is_near(dxb_3, D3 {2., 3, 4}));

  d3a -= d3;
  EXPECT_TRUE(is_near(d3a, m33));
  dxa_3 -= d3;
  EXPECT_TRUE(is_near(dxa_3, m33));
  d3b -= M33::Identity();
  EXPECT_TRUE(is_near(d3b, m33));
  dxb_3 -= Mxx::Identity(3, 3);
  EXPECT_TRUE(is_near(dxb_3, m33));

  d3a *= 3;
  EXPECT_TRUE(is_near(d3a, D3 {3., 6, 9}));
  dxa_3 *= 3;
  EXPECT_TRUE(is_near(dxa_3, D3 {3., 6, 9}));

  d3a /= 3;
  EXPECT_TRUE(is_near(d3a, D3 {1., 2, 3}));
  dxa_3 /= 3;
  EXPECT_TRUE(is_near(dxa_3, D3 {1., 2, 3}));

  d3a *= d3b;
  EXPECT_TRUE(is_near(d3a, D3 {1., 4, 9}));
  dxa_3 *= dxb_3;
  EXPECT_TRUE(is_near(dxa_3, D3 {1., 4, 9}));

  EXPECT_EQ((dx_3.rows()), 3);
  EXPECT_EQ((dx_3.cols()), 3);
}

TEST(adapters, Diagonal_subscripts)
{
  static_assert(writable_by_component<D3&>);
  static_assert(writable_by_component<D3&>);

  static_assert(writable_by_component<Dx&>);
  static_assert(writable_by_component<Dx&>);

  D3 d3a {1, 2, 3};
  Dx dxa_3 {1, 2, 3};
  bool test;

  set_component(d3a, 5.5, 0, 0);
  EXPECT_NEAR(get_component(d3a, 0, 0), 5.5, 1e-8);
  set_component(d3a, 6.5, 1, 1);
  EXPECT_NEAR(get_component(d3a, 1, 1), 6.5, 1e-8);
  test = false;
  try { set_component(d3a, 8.5, 2, 0); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_NEAR(get_component(d3a, 2, 0), 0, 1e-8);
  set_component(d3a, 7.5, 2, 2);

  set_component(dxa_3, 5.5, 0, 0);
  EXPECT_NEAR(get_component(dxa_3, 0, 0), 5.5, 1e-8);
  set_component(dxa_3, 6.5, 1, 1);
  EXPECT_NEAR(get_component(dxa_3, 1, 1), 6.5, 1e-8);
  test = false;
  try { set_component(dxa_3, 8.5, 2, 0); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_NEAR(get_component(dxa_3, 2, 0), 0, 1e-8);
  set_component(dxa_3, 7.5, 2, 2);

  d3a(0, 0) = 5;
  d3a({1, 1}) = 6;
  d3a(std::vector<std::size_t> {2, 2}) = 7;
  test = false;
  try { d3a(1, 0) = 3; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);

  dxa_3(0, 0) = 5;
  dxa_3(1, 1) = 6;
  dxa_3(2, 2) = 7;
  test = false;
  try { dxa_3(1, 0) = 3; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);

  EXPECT_TRUE(is_near(d3a, diagonal_adapter<M31> {5., 6, 7}));
  EXPECT_NEAR(d3a(0, 0), 5, 1e-6);
  EXPECT_NEAR(d3a(1, 1), 6, 1e-6);
  EXPECT_NEAR(d3a(2, 2), 7, 1e-6);
  EXPECT_NEAR(d3a(0, 0), 5, 1e-6);
  EXPECT_NEAR(d3a(0, 1), 0, 1e-6);
  EXPECT_NEAR(d3a(0, 2), 0, 1e-6);
  EXPECT_NEAR(d3a(1, 0), 0, 1e-6);
  EXPECT_NEAR(d3a(1, 1), 6, 1e-6);
  EXPECT_NEAR(d3a(1, 2), 0, 1e-6);
  EXPECT_NEAR(d3a(2, 0), 0, 1e-6);
  EXPECT_NEAR(d3a(2, 1), 0, 1e-6);
  EXPECT_NEAR(d3a(2, 2), 7, 1e-6);

  EXPECT_TRUE(is_near(dxa_3, diagonal_adapter<M31> {5., 6, 7}));
  EXPECT_NEAR(dxa_3(0, 0), 5, 1e-6);
  EXPECT_NEAR(dxa_3(1, 1), 6, 1e-6);
  EXPECT_NEAR(dxa_3(2, 2), 7, 1e-6);
  EXPECT_NEAR(dxa_3(0, 0), 5, 1e-6);
  EXPECT_NEAR(dxa_3(0, 1), 0, 1e-6);
  EXPECT_NEAR(dxa_3(0, 2), 0, 1e-6);
  EXPECT_NEAR(dxa_3(1, 0), 0, 1e-6);
  EXPECT_NEAR(dxa_3(1, 1), 6, 1e-6);
  EXPECT_NEAR(dxa_3(1, 2), 0, 1e-6);
  EXPECT_NEAR(dxa_3(2, 0), 0, 1e-6);
  EXPECT_NEAR(dxa_3(2, 1), 0, 1e-6);
  EXPECT_NEAR(dxa_3(2, 2), 7, 1e-6);

  EXPECT_NEAR((D3 {1., 2, 3}).nested_object()[0], 1, 1e-6);
  EXPECT_NEAR((D3 {1., 2, 3}).nested_object()[1], 2, 1e-6);
  EXPECT_NEAR((D3 {1., 2, 3}).nested_object()[2], 3, 1e-6);

  EXPECT_NEAR((Dx {1., 2, 3}).nested_object()[0], 1, 1e-6);
  EXPECT_NEAR((Dx {1., 2, 3}).nested_object()[1], 2, 1e-6);
  EXPECT_NEAR((Dx {1., 2, 3}).nested_object()[2], 3, 1e-6);

  D1 d1a {3};
  Dx dxa_1 {3};

  EXPECT_NEAR(d1a(0, 0), 5, 1e-6);
  EXPECT_NEAR(d1a(0), 5, 1e-6);
  EXPECT_NEAR(d1a[0], 5, 1e-6);
  EXPECT_NEAR(d1a({0, 0}), 5, 1e-6);
  EXPECT_NEAR(d1a(std::array<std::size_t, 2>{0, 0}), 5, 1e-6);
  EXPECT_NEAR(d1a(std::array<std::size_t, 1>{0}), 5, 1e-6);
}


TEST(adapters, Diagonal_traits)
{
  static_assert(diagonal_matrix<decltype(diagonal_adapter<M11>{2.})>);
  static_assert(diagonal_matrix<decltype(diagonal_adapter<M21>{2, 3})>);
  static_assert(hermitian_matrix<decltype(diagonal_adapter<M21>{2, 3})>);
  static_assert(triangular_matrix<decltype(diagonal_adapter<M21>{2, 3})>);
  static_assert(triangular_matrix<decltype(diagonal_adapter<M21>{2, 3}), triangle_type::lower>);
  static_assert(triangular_matrix<decltype(diagonal_adapter<M21>{2, 3}), triangle_type::upper>);
  static_assert(not identity_matrix<decltype(diagonal_adapter<M21>{2, 3})>);
  static_assert(not zero<decltype(diagonal_adapter<M21>{2, 3})>);
  static_assert(covariance_nestable<decltype(diagonal_adapter<M21>{2, 3})>);
  static_assert(covariance_nestable<decltype(M33::Identity())>);
  static_assert(identity_matrix<decltype(M33::Identity() * M33::Identity())>);
  static_assert(diagonal_matrix<decltype(0.3 * M33::Identity() + 0.7 * M33::Identity() * 0.7)>);
  static_assert(diagonal_matrix<decltype(0.7 * M33::Identity() * 0.7 - M33::Identity() * 0.3)>);
  static_assert(diagonal_matrix<decltype((0.7 * M33::Identity()) * (M33::Identity() + M33::Identity()))>);

  EXPECT_TRUE(is_near(make_dense_object_from<D3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3), m33));
}


TEST(adapters, to_diagonal)
{
  // See eigen-diagonal.test.cpp

  auto m11 = M11 {3};

  EXPECT_TRUE(is_near(to_diagonal(Mx1 {m11}), m11)); static_assert(internal::diagonal_expr<decltype(to_diagonal(Mx1 {m11}))>);
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Warray-bounds"
  EXPECT_TRUE(is_near(to_diagonal(Mxx {m11}), m11)); static_assert(internal::diagonal_expr<decltype(to_diagonal(Mxx {m11}))>);
//#pragma GCC diagnostic pop

  // zero input:
  auto z31 = M31::Identity() - M31::Identity();
  static_assert(zero<decltype(to_diagonal(z31))>);
  static_assert(zero<decltype(to_diagonal(std::move(z31)))>);
  EXPECT_TRUE(is_near(to_diagonal(z31), M33::Zero()));
}

TEST(adapters, diagonal_make_triangular_matrix)
{
  auto m22h = make_dense_object_from<M22>(3, 1, 1, 3);
  auto m22d = make_dense_object_from<M22>(3, 0, 0, 3);
  auto m22_uppert = Eigen::TriangularView<M22, Eigen::Upper> {m22h};
  auto m22_lowert = Eigen::TriangularView<M22, Eigen::Lower> {m22h};

  EXPECT_TRUE(is_near(make_triangular_matrix<triangle_type::lower>(m22_uppert), m22d));
  static_assert(internal::diagonal_expr<decltype(make_triangular_matrix<triangle_type::lower>(m22_uppert))>);
  static_assert(diagonal_matrix<decltype(make_triangular_matrix<triangle_type::lower>(m22_uppert))>);

  EXPECT_TRUE(is_near(make_triangular_matrix<triangle_type::upper>(m22_lowert), m22d));
  static_assert(internal::diagonal_expr<decltype(make_triangular_matrix<triangle_type::upper>(m22_lowert))>);
  static_assert(diagonal_matrix<decltype(make_triangular_matrix<triangle_type::upper>(m22_lowert))>);

  EXPECT_TRUE(is_near(make_triangular_matrix<triangle_type::diagonal>(m22h), m22d));
  static_assert(internal::diagonal_expr<decltype(make_triangular_matrix<triangle_type::diagonal>(m22h))>);
  EXPECT_TRUE(is_near(make_triangular_matrix<triangle_type::diagonal>(m22h), m22d));
  static_assert(internal::diagonal_expr<decltype(make_triangular_matrix<triangle_type::diagonal>(m22h))>);
}

TEST(adapters, diagonal_make_functions)
{
  auto m22 = M22 {};
  auto m2x_2 = M2x {m22};
  auto mx2_2 = Mx2 {m22};
  auto mxx_22 = Mxx {m22};

  EXPECT_TRUE(is_near(make_identity_matrix_like<M22>(), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Mxx>(Dimensions<2>()), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Mxx>(2), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like(m22), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like(m2x_2), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like(mx2_2), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like(mxx_22), M22::Identity()));

  static_assert(identity_matrix<decltype(make_identity_matrix_like<M22>())>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<Mxx>(Dimensions<2>()))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<Mxx>(2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(m22))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(m2x_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(mx2_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(mxx_22))>);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<M22>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Mxx>(Dimensions<2>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Mxx>(2)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_identity_matrix_like<Mxx>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m22)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m2x_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(mx2_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(mxx_22)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_identity_matrix_like(mxx_22)), 2);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<M22>()), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Mxx>(Dimensions<2>())), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<Mxx>(2)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_identity_matrix_like<Mxx>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m22)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(m2x_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(mx2_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(mxx_22)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_identity_matrix_like(mxx_22)), 2);

  constant_adapter<M34, double, 5> c534 {};
  constant_adapter<M3x, double, 5> c530_4 {4};
  constant_adapter<Mx4, double, 5> c504_3 {3};
  constant_adapter<Mxx, double, 5> c500_34 {3, 4};

  constant_adapter<M33, double, 5> c533 {};
  constant_adapter<M3x, double, 5> c530_3 {3};
  constant_adapter<Mx3, double, 5> c503_3 {3};
  constant_adapter<Mxx, double, 5> c500_33 {3, 3};

  constant_adapter<M31, double, 5> c531 {};
  constant_adapter<M3x, double, 5> c530_1 {1};
  constant_adapter<Mx1, double, 5> c501_3 {3};
  constant_adapter<Mxx, double, 5> c500_31 {3, 1};

  static_assert(identity_matrix<decltype(make_identity_matrix_like<C533>())>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<C500>(Dimensions<3>()))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<C500>(3))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c533))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c530_3))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c503_3))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c500_33))>);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C533>()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(Dimensions<3>())), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(3)), 0> == dynamic_size); EXPECT_EQ(get_pattern_collection<0>(make_identity_matrix_like<C500>(3)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c533)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c530_3)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c503_3)), 0> == 3); EXPECT_EQ(get_pattern_collection<0>(make_identity_matrix_like(c503_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c500_33)), 0> == dynamic_size); EXPECT_EQ(get_pattern_collection<0>(make_identity_matrix_like(c500_33)), 3);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C533>()), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(3)), 1> == dynamic_size);  EXPECT_EQ(get_pattern_collection<1>(make_identity_matrix_like<C500>(3)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c533)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c530_3)), 1> == 3); EXPECT_EQ(get_pattern_collection<1>(make_identity_matrix_like(c530_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c503_3)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c500_33)), 1> == dynamic_size); EXPECT_EQ(get_pattern_collection<1>(make_identity_matrix_like(c500_33)), 3);

  ZA23 z23 {Dimensions<2>(), Dimensions<3>()};
  ZA20 z20_3 {Dimensions<2>(), 3};
  ZA03 z03_2 {2, Dimensions<3>()};
  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z20_2 {Dimensions<2>(), 2};
  ZA02 z02_2 {2, Dimensions<2>()};
  ZA00 z00_23 {2, 3};
  ZA00 z00_22 {2, 2};

  static_assert(identity_matrix<decltype(make_identity_matrix_like<ZA22>())>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<ZA00>(Dimensions<2>()))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<ZA00>(2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z22))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z20_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z02_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z00_22))>);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA22>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA00>(Dimensions<2>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA00>(2)), 0> == dynamic_size); EXPECT_EQ(get_pattern_collection<0>(make_identity_matrix_like<ZA00>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z22)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z20_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z02_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z00_22)), 0> == dynamic_size); EXPECT_EQ(get_pattern_collection<0>(make_identity_matrix_like(z00_22)), 2);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA22>()), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA00>(Dimensions<2>())), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA00>(2)), 1> == dynamic_size);  EXPECT_EQ(get_pattern_collection<1>(make_identity_matrix_like<ZA00>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z22)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z20_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z02_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z00_22)), 1> == dynamic_size); EXPECT_EQ(get_pattern_collection<1>(make_identity_matrix_like(z00_22)), 2);

  EXPECT_TRUE(is_near(make_identity_matrix_like<constant_adapter<M22, double, 3>>(), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<zero_adapter<M22>>(), M22::Identity()));

  EXPECT_TRUE(is_near(make_identity_matrix_like<D3>(), M33::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Dx>(3), M33::Identity(3, 3)));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Dx>(3), M33::Identity()));

  EXPECT_TRUE(is_near(make_zero(d3), M33::Zero()));
  EXPECT_TRUE(is_near(make_zero(dx_3), M33::Zero(3, 3)));
  EXPECT_TRUE(is_near(make_zero<D3>(), M33::Zero()));
  EXPECT_TRUE(is_near(make_zero(dx_3), M33::Zero()));

  EXPECT_TRUE(is_near(make_diagonal_adapter(c531), make_dense_object_from<M33>(5, 0, 0, 0, 5, 0, 0, 0, 5)));
  EXPECT_TRUE(is_near(make_diagonal_adapter(c531, Dimensions<3>{}), make_dense_object_from<M33>(5, 0, 0, 0, 5, 0, 0, 0, 5)));
  EXPECT_TRUE(is_near(make_diagonal_adapter(c531, Dimensions{3}), make_dense_object_from<M33>(5, 0, 0, 0, 5, 0, 0, 0, 5)));
  EXPECT_TRUE(is_near(make_diagonal_adapter(c531, Dimensions<4>{}), make_dense_object_from<M44>(5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0)));
  EXPECT_TRUE(is_near(make_diagonal_adapter(c531, Dimensions{4}), make_dense_object_from<M44>(5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0)));
  EXPECT_TRUE(is_near(make_diagonal_adapter(c531, Dimensions<3>{}, Dimensions<4>{}), make_dense_object_from<M34>(5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0)));
  EXPECT_TRUE(is_near(make_diagonal_adapter(c531, Dimensions{3}, Dimensions{4}), make_dense_object_from<M34>(5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0)));
}

TEST(adapters, Diagonal_overloads)
{
  EXPECT_TRUE(is_near(make_dense_object_from(d3), m33));
  EXPECT_TRUE(is_near(make_dense_object_from(dx_3), m33));

  EXPECT_EQ(get_pattern_collection<0>(d3), 3);
  EXPECT_EQ(get_pattern_collection<0>(dx_3), 3);

  EXPECT_EQ(get_pattern_collection<1>(d3), 3);
  EXPECT_EQ(get_pattern_collection<1>(dx_3), 3);

  EXPECT_TRUE(is_near(cholesky_square(d3), D3 {1, 4, 9}));
  EXPECT_TRUE(is_near(cholesky_square(dx_3), D3 {1, 4, 9}));
  EXPECT_TRUE(is_near(cholesky_factor(D3 {1., 4, 9}), m33));
  EXPECT_TRUE(is_near(cholesky_factor(Dx {1., 4, 9}), m33));
  EXPECT_TRUE(is_near(cholesky_square(M11 {4}), M11 {16}));
  EXPECT_TRUE(is_near(cholesky_factor(M11 {4}), M11 {2}));
  EXPECT_TRUE(is_near(cholesky_square((Mxx {1,1} << 4).finished()), M11 {16}));
  EXPECT_TRUE(is_near(cholesky_factor((Mxx {1,1} << 4).finished()), M11 {2}));
  EXPECT_TRUE(is_near(cholesky_square(M22::Identity() * 0.1), D2 {0.01, 0.01}));
  EXPECT_TRUE(is_near(cholesky_factor(M22::Identity() * 0.01), D2 {0.1, 0.1}));
  EXPECT_TRUE(is_near(cholesky_square(Mxx::Identity(2, 2) * 0.1), D2 {0.01, 0.01}));
  EXPECT_TRUE(is_near(cholesky_factor(Mxx::Identity(2, 2) * 0.01), D2 {0.1, 0.1}));
  EXPECT_TRUE(is_near(cholesky_square(diagonal_adapter<M11> {9.}), M11 {81}));
  EXPECT_TRUE(is_near(cholesky_square(Dx {M11 {9}}), M11 {81}));
  EXPECT_TRUE(is_near(cholesky_factor(diagonal_adapter<M11> {9.}), M11 {3}));
  EXPECT_TRUE(is_near(cholesky_factor(Dx {M11 {9}}), M11 {3}));

  auto m21 = M21 {1, 4};
  auto m2x_1 = M2x {m21};
  auto mx1_2 = Mx1 {m21};
  auto mxx_21 = Mxx {m21};

  auto m22_1004 = make_dense_object_from<M22>(1, 0, 0, 4);

  EXPECT_TRUE(is_near(to_diagonal(m21), m22_1004));
  EXPECT_TRUE(is_near(to_diagonal(m2x_1), m22_1004)); static_assert(not has_dynamic_dimensions<decltype(to_diagonal(m2x_1))>);
  EXPECT_TRUE(is_near(to_diagonal(mx1_2), m22_1004)); static_assert(has_dynamic_dimensions<decltype(to_diagonal(mx1_2))>);
  EXPECT_TRUE(is_near(to_diagonal(mxx_21), m22_1004)); static_assert(has_dynamic_dimensions<decltype(to_diagonal(mxx_21))>);

  auto z11 {M11::Identity() - M11::Identity()};

  auto z21 {(M22::Identity() - M22::Identity()).diagonal()};
  auto z01_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 1> {z11, 2, 1};
  auto z20_1 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 1};
  auto z00_21 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 1};

  static_assert(zero<decltype(to_diagonal(z21))>);
  static_assert(zero<decltype(to_diagonal(z20_1))>); static_assert(not has_dynamic_dimensions<decltype(to_diagonal(z20_1))>);
  static_assert(zero<decltype(to_diagonal(z01_2))>); static_assert(has_dynamic_dimensions<decltype(to_diagonal(z01_2))>);
  static_assert(zero<decltype(to_diagonal(z00_21))>); static_assert(has_dynamic_dimensions<decltype(to_diagonal(z00_21))>);

  EXPECT_TRUE(is_near(diagonal_of(d3), m31));
  EXPECT_TRUE(is_near(diagonal_of(dx_3), m31));

  EXPECT_TRUE(is_near(transpose(d3), d3));
  EXPECT_TRUE(is_near(transpose(diagonal_adapter<M31> {cdouble(1,2), cdouble(2,3), 3}), diagonal_adapter<M31> {cdouble(1,2), cdouble(2,3), 3}));

  EXPECT_TRUE(is_near(adjoint(d3), d3));
  EXPECT_TRUE(is_near(adjoint(diagonal_adapter<M31> {cdouble(1,2), cdouble(2,3), 3}), diagonal_adapter<M31> {cdouble(1,-2), cdouble(2,-3), 3}));

  EXPECT_NEAR(determinant(diagonal_adapter<M31> {2., 3, 4}), 24, 1e-6);

  EXPECT_NEAR(trace(diagonal_adapter<M31> {2., 3, 4}), 9, 1e-6);
  //
  EXPECT_TRUE(is_near(solve(d3, make_dense_object_from<M31>(4., 10, 18)),
    make_eigen_matrix(4., 5, 6)));
  EXPECT_TRUE(is_near(average_reduce<1>(d3), make_eigen_matrix(1., 2, 3)));
  EXPECT_TRUE(is_near(average_reduce<0>(d3), make_eigen_matrix(1., 2, 3)));
  EXPECT_TRUE(is_near(LQ_decomposition(d3), d3));
  EXPECT_TRUE(is_near(QR_decomposition(d3), d3));
  EXPECT_TRUE(is_near(LQ_decomposition(diagonal_adapter<M31> {cdouble(1,2), cdouble(2,3), 3}), diagonal_adapter<M31> {cdouble(1,2), cdouble(2,3), 3}));
  EXPECT_TRUE(is_near(QR_decomposition(diagonal_adapter<M31> {cdouble(1,2), cdouble(2,3), 3}), diagonal_adapter<M31> {cdouble(1,2), cdouble(2,3), 3}));
  EXPECT_TRUE(is_near(LQ_decomposition(M1by1 {4}), diagonal_adapter<M11> {4.}));
  EXPECT_TRUE(is_near(QR_decomposition(M1by1 {4}), diagonal_adapter<M11> {4.}));

  using N = std::normal_distribution<double>;

  D2 d3b = make_zero<D2>(Dimensions<2>{}, Dimensions<2>{});
  Dx dx_2 {make_zero<Dx>(2, 2)};
  Dx dx_3 {make_zero<Dx>(3, 3)};
  for (int i=0; i<100; i++)
  {
    d3b = (d3b * i + randomize<D2>(N {1.0, 0.3}, 2.0)) / (i + 1);
    dx_2 = (dx_2 * i + randomize<D2>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
    dx_3 = (dx_3 * i + randomize<Dx>(3, 3, N {1.0, 0.3})) / (i + 1);
  }
  D2 d2_offset = {1, 2};
  D3 d0_3_offset = {1, 1, 1};
  EXPECT_TRUE(is_near(d3b, d2_offset, 0.1));
  EXPECT_FALSE(is_near(d3b, d2_offset, 1e-6));
  EXPECT_TRUE(is_near(dx_2, d2_offset, 0.1));
  EXPECT_FALSE(is_near(dx_2, d2_offset, 1e-6));
  EXPECT_TRUE(is_near(dx_3, d0_3_offset, 0.1));
  EXPECT_FALSE(is_near(dx_3, d0_3_offset, 1e-6));
}


TEST(adapters, diagonal_contract)
{
  auto m33dd = make_dense_object_from<M33>(10, 0, 0, 0, 18, 0, 0, 0, 28);

  EXPECT_TRUE(is_near(contract(dm3a, dm3b), m33dd)); static_assert(diagonal_matrix<decltype(contract(dm3a, dm3b))>);
  EXPECT_TRUE(is_near(contract(dm3a, dm0_3b), m33dd)); static_assert(dimension_size_of_index_is<decltype(contract(dm3a, dm0_3b)), 0, 3>);
  EXPECT_TRUE(is_near(contract(dm3a, dw3b), m33dd)); static_assert(diagonal_matrix<decltype(contract(dm3a, dw3b))>);
  EXPECT_TRUE(is_near(contract(dm0_3a, dm3b), m33dd)); static_assert(dimension_size_of_index_is<decltype(contract(dm0_3a, dm3b)), 0, 3>);
  EXPECT_TRUE(is_near(contract(dm0_3a, dm0_3b), m33dd)); static_assert(diagonal_matrix<decltype(contract(dm0_3a, dm0_3b))>);
  EXPECT_TRUE(is_near(contract(dm0_3a, dw3b), m33dd)); static_assert(dimension_size_of_index_is<decltype(contract(dm0_3a, dw3b)), 0, 3>);
  EXPECT_TRUE(is_near(contract(dw3a, dm3b), m33dd)); static_assert(diagonal_matrix<decltype(contract(dw3a, dm3b))>);
  EXPECT_TRUE(is_near(contract(dw3a, dm0_3b), m33dd)); static_assert(dimension_size_of_index_is<decltype(contract(dw3a, dm0_3b)), 0, 3>);
  EXPECT_TRUE(is_near(contract(dw3a, dw3b), m33dd)); static_assert(diagonal_matrix<decltype(contract(dw3a, dw3b))>);
}


TEST(adapters, Diagonal_blocks)
{
  EXPECT_TRUE(is_near(concatenate_diagonal(d3, D2 {4, 5}), d5));
  EXPECT_TRUE(is_near(concatenate_diagonal(d3, Dx {4, 5}), d5));
  EXPECT_TRUE(is_near(concatenate_diagonal(dx_3, D2 {4, 5}), d5));
  EXPECT_TRUE(is_near(concatenate_diagonal(dx_3, Dx {4, 5}), d5));

  auto m_1234vert = make_eigen_matrix<4,2>(1., 0, 0, 2, 3, 0, 0, 4);
  EXPECT_TRUE(is_near(concatenate_vertical(D2 {1, 2}, D2 {3, 4}), m_1234vert));
  EXPECT_TRUE(is_near(concatenate_vertical(D2 {1, 2}, Dx {3, 4}), m_1234vert));
  EXPECT_TRUE(is_near(concatenate_vertical(Dx {1, 2}, D2 {3, 4}), m_1234vert));
  EXPECT_TRUE(is_near(concatenate_vertical(Dx {1, 2}, Dx {3, 4}), m_1234vert));

  auto m_1234horiz = make_eigen_matrix<2,4>(1., 0, 3, 0, 0, 2, 0, 4);
  EXPECT_TRUE(is_near(concatenate_horizontal(D2 {1, 2}, D2 {3, 4}), m_1234horiz));
  EXPECT_TRUE(is_near(concatenate_horizontal(D2 {1, 2}, Dx {3, 4}), m_1234horiz));
  EXPECT_TRUE(is_near(concatenate_horizontal(Dx {1, 2}, D2 {3, 4}), m_1234horiz));
  EXPECT_TRUE(is_near(concatenate_horizontal(Dx {1, 2}, Dx {3, 4}), m_1234horiz));

  EXPECT_TRUE(is_near(split_diagonal(d5), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal(dx_5), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(d5), std::tuple {d3, D2 {4, 5}}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(d5_const), std::tuple {d3, D2 {4, 5}}));
  //EXPECT_TRUE(is_near(split_diagonal<3, 2>(dx_5), std::tuple {d3, D2 {4, 5}}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(d5), std::tuple {D2 {1, 2}, D2 {3, 4}}));
  //EXPECT_TRUE(is_near(split_diagonal<2, 2>(dx_5), std::tuple {D2 {1, 2}, D2 {3, 4}}));

  EXPECT_TRUE(is_near(split_vertical(d5), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical(dx_5), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(d5),
    std::tuple {make_dense_object_from<M35>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0),
               make_dense_object_from<M25>(
                 0, 0, 0, 4, 0,
                 0, 0, 0, 0, 5)}));
  /*EXPECT_TRUE(is_near(split_vertical<3, 2>(dx_5),
    std::tuple {make_dense_object_from<M35>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0),
                make_dense_object_from<M25>(
                  0, 0, 0, 4, 0,
                  0, 0, 0, 0, 5)}));*/
  EXPECT_TRUE(is_near(split_vertical<2, 2>(d5),
    std::tuple {make_dense_object_from<M25>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0),
               make_dense_object_from<M25>(
                 0, 0, 3, 0, 0,
                 0, 0, 0, 4, 0)}));
  /*EXPECT_TRUE(is_near(split_vertical<2, 2>(dx_5),
    std::tuple {make_dense_object_from<M25>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0),
                make_dense_object_from<M25>(
                  0, 0, 3, 0, 0,
                  0, 0, 0, 4, 0)}));*/

  EXPECT_TRUE(is_near(split_horizontal(d5), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(dx_5), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(d5),
    std::tuple {make_dense_object_from<M53>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      0, 0, 0,
      0, 0, 0),
               make_dense_object_from<M52>(
                 0, 0,
                 0, 0,
                 0, 0,
                 4, 0,
                 0, 5)}));
  /*EXPECT_TRUE(is_near(split_horizontal<3, 2>(dx_5),
    std::tuple {make_dense_object_from<M53>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      0, 0, 0,
      0, 0, 0),
                make_dense_object_from<M52>(
                  0, 0,
                  0, 0,
                  0, 0,
                  4, 0,
                  0, 5)}));*/
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(d5),
    std::tuple {make_dense_object_from<M52>(
      1, 0,
      0, 2,
      0, 0,
      0, 0,
      0, 0),
               make_dense_object_from<M52>(
                 0, 0,
                 0, 0,
                 3, 0,
                 0, 4,
                 0, 0)}));
  /*EXPECT_TRUE(is_near(split_horizontal<2, 2>(dx_5),
    std::tuple {make_dense_object_from<M52>(
      1, 0,
      0, 2,
      0, 0,
      0, 0,
      0, 0),
                make_dense_object_from<M52>(
                  0, 0,
                  0, 0,
                  3, 0,
                  0, 4,
                  0, 0)}));*/

  EXPECT_TRUE(is_near(column(d3, 2), make_eigen_matrix(0., 0, 3)));
  EXPECT_TRUE(is_near(column<1>(d3), make_eigen_matrix(0., 2, 0)));
  
  EXPECT_TRUE(is_near(row(d3, 2), eigen_matrix_t<double, 1, 3>(0., 0, 3)));
  EXPECT_TRUE(is_near(row<1>(d3), eigen_matrix_t<double, 1, 3>(0., 2, 0)));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, d3),
    make_dense_object_from<M33>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, d3),
    make_dense_object_from<M33>(
      1, 1, 2,
      0, 3, 2,
      0, 1, 5)));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, d3),
    make_dense_object_from<M33>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }, d3),
    make_dense_object_from<M33>(
      1, 0, 0,
      1, 3, 1,
      2, 2, 5)));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, d3),
    make_dense_object_from<M33>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, d3),
    make_dense_object_from<M33>(
      1, 1, 2,
      1, 4, 3,
      2, 3, 7)));
}

TEST(adapters, Diagonal_arithmetic)
{
  auto d3a = d3;
  auto d3b = diagonal_adapter<M31> {4., 5, 6};
  auto i = M33::Identity();
  auto z = zero_adapter<eigen_matrix_t<double, 3, 3>> {};
  EXPECT_TRUE(is_near(d3a + d3b, diagonal_adapter<M31> {5., 7, 9})); static_assert(internal::diagonal_expr<decltype(d3a + d3b)>);
  EXPECT_TRUE(is_near(d3a + i, diagonal_adapter<M31> {2., 3, 4})); static_assert(internal::diagonal_expr<decltype(d3a + i)>);
  EXPECT_TRUE(is_near(d3a + z, d3a)); static_assert(internal::diagonal_expr<decltype(d3a + z)>);
  EXPECT_TRUE(is_near(i + i, diagonal_adapter<M31> {2., 2, 2})); static_assert(diagonal_matrix<decltype(i + i)>);
  EXPECT_TRUE(is_near(i + z, i)); static_assert(identity_matrix<decltype(i + z)>);
  EXPECT_TRUE(is_near(z + i, i)); static_assert(identity_matrix<decltype(z + i)>);
  EXPECT_TRUE(is_near(z + z, z)); static_assert(zero<decltype(z + z)>);

  EXPECT_TRUE(is_near(d3a - d3b, diagonal_adapter<M31> {-3., -3, -3})); static_assert(diagonal_matrix<decltype(d3a - d3b)>);
  EXPECT_TRUE(is_near(d3a - i, diagonal_adapter<M31> {0., 1, 2})); static_assert(diagonal_matrix<decltype(d3a - i)>);
  EXPECT_TRUE(is_near(d3a - z, d3a)); static_assert(diagonal_matrix<decltype(d3a - z)>);
  EXPECT_TRUE(is_near(i - i, z)); static_assert(zero<decltype(i - i)>);
  EXPECT_TRUE(is_near(i * (i - i), z)); static_assert(zero<decltype(i - i)>);
  EXPECT_TRUE(is_near(z * (i - i), z)); static_assert(zero<decltype(z * (i - i))>);
  EXPECT_TRUE(is_near(i - z, i)); static_assert(identity_matrix<decltype(i - z)>);
  EXPECT_TRUE(is_near(z - i, -i)); static_assert(diagonal_matrix<decltype(z - i)>);
  EXPECT_TRUE(is_near(z - z, z)); static_assert(zero<decltype(z - z)>);

  EXPECT_TRUE(is_near(d3a * 4, diagonal_adapter<M31> {4., 8, 12})); static_assert(internal::diagonal_expr<decltype(d3a * 4)>);
  EXPECT_TRUE(is_near(4 * d3a, diagonal_adapter<M31> {4., 8, 12})); static_assert(internal::diagonal_expr<decltype(4 * d3a)>);
  EXPECT_TRUE(is_near(d3a / 2, diagonal_adapter<M31> {0.5, 1, 1.5})); static_assert(diagonal_matrix<decltype(d3a / 2)>);
  static_assert(diagonal_matrix<decltype(d3a / 0)>);
  EXPECT_TRUE(is_near(i * 4, diagonal_adapter<M31> {4., 4, 4})); static_assert(diagonal_matrix<decltype(i * 4)>);
  EXPECT_TRUE(is_near(4 * i, diagonal_adapter<M31> {4., 4, 4})); static_assert(diagonal_matrix<decltype(4 * i)>);
  static_assert(diagonal_matrix<decltype(i * 0)>);
  static_assert(not zero<decltype(i * 0)>);
  EXPECT_TRUE(is_near(i / 2, diagonal_adapter<M31> {0.5, 0.5, 0.5})); static_assert(diagonal_matrix<decltype(i / 2)>);
  EXPECT_TRUE(is_near(z * 4, z)); static_assert(zero<decltype(z * 4)>);
  EXPECT_TRUE(is_near(4 * z, z)); static_assert(zero<decltype(4 * z)>);
  EXPECT_TRUE(is_near(z / 4, z)); static_assert(zero<decltype(z / 4)>);
  static_assert(zero<decltype(z / 0)>); // This is not technically true, but it's indeterminate at compile time.
  EXPECT_TRUE(is_near((i - i) * 4, z)); static_assert(zero<decltype((i - i) * 4)>);
  EXPECT_TRUE(is_near((i - i) / 4, z)); static_assert(not zero<decltype((i - i) / 4)>);
  static_assert(diagonal_matrix<decltype(i / 0)>);

  EXPECT_TRUE(is_near(-d3a, diagonal_adapter<M31> {-1., -2, -3})); static_assert(internal::diagonal_expr<decltype(-d3a)>);
  EXPECT_TRUE(is_near(-i, diagonal_adapter<M31> {-1., -1, -1})); static_assert(diagonal_matrix<decltype(-i)>);
  EXPECT_TRUE(is_near(-z, z)); static_assert(zero<decltype(z)>);

  EXPECT_TRUE(is_near(d3a * d3b, diagonal_adapter {4., 10, 18})); static_assert(internal::diagonal_expr<decltype(d3a * d3b)>);
  EXPECT_TRUE(is_near(d3a * constant_adapter<eigen_matrix_t<double, 3, 3>, 4> {}, make_dense_object_from<M33>(4, 4, 4, 8, 8, 8, 12, 12, 12)));
  EXPECT_TRUE(is_near(constant_adapter<eigen_matrix_t<double, 3, 3>, 4> {} * d3a, make_dense_object_from<M33>(4, 8, 12, 4, 8, 12, 4, 8, 12)));
  EXPECT_TRUE(is_near(d3a * i, d3a)); static_assert(internal::diagonal_expr<decltype(d3a * i)>);
  EXPECT_TRUE(is_near(i * d3a, d3a)); static_assert(internal::diagonal_expr<decltype(i * d3a)>);
  EXPECT_TRUE(is_near(d3a * z, z)); static_assert(zero<decltype(d3a * z)>);
  EXPECT_TRUE(is_near(z * d3a, z)); static_assert(zero<decltype(z * d3a)>);
  EXPECT_TRUE(is_near(i * i, i)); static_assert(identity_matrix<decltype(i * i)>);
  EXPECT_TRUE(is_near(z * z, z)); static_assert(zero<decltype(z * z)>);

  EXPECT_TRUE(is_near(d3a * HermitianAdapter<M33, HermitianAdapterType::lower> {
    1., 2, 3,
    2, 4, 5,
    3, 5, 6}, make_eigen_matrix<3,3>(
      1., 2, 3,
      4, 8, 10,
      9, 15, 18)));
  EXPECT_TRUE(is_near(HermitianAdapter<M33, HermitianAdapterType::upper> {
    1., 2, 3,
    2, 4, 5,
    3, 5, 6} * d3, make_eigen_matrix<3,3>(
    1., 4, 9,
    2, 8, 15,
    3, 10, 18)));
  EXPECT_TRUE(is_near(d3 * TriangularAdapter<M33, triangle_type::lower> {
    1., 0, 0,
    2, 3, 0,
    4, 5, 6}, make_eigen_matrix<3,3>(
    1., 0, 0,
    4, 6, 0,
    12, 15, 18)));
  EXPECT_TRUE(is_near(TriangularAdapter<M33, triangle_type::upper> {
    1., 0, 0,
    2, 3, 0,
    4, 5, 6} * d3, make_eigen_matrix<3,3>(
    1., 0, 0,
    2, 6, 0,
    4, 10, 18)));
  EXPECT_TRUE(is_near(d3 * make_dense_object_from<M33>(
    1, 2, 3,
    4, 5, 6,
    7, 8, 9), make_eigen_matrix<3,3>(
    1., 2, 3,
    8, 10, 12,
    21, 24, 27)));
  EXPECT_TRUE(is_near(make_dense_object_from<M33>(
    1, 2, 3,
    4, 5, 6,
    7, 8, 9) * d3, make_eigen_matrix<3,3>(
    1., 4, 9,
    4, 10, 18,
    7, 16, 27)));
}

TEST(adapters, Diagonal_references)
{
  diagonal_adapter<const M31> n {4, 5, 6};
  diagonal_adapter<M31> x = d3;
  diagonal_adapter<M31&> x_l {x};
  EXPECT_TRUE(is_near(x_l, d3));
  diagonal_adapter x_l2 {x_l};
  static_assert(std::is_lvalue_reference_v<nested_object_of_t<decltype(x_l2)>>);
  EXPECT_TRUE(is_near(x_l, d3));
  diagonal_adapter<const M31&> x_lc = x_l;
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
  EXPECT_TRUE(is_near(diagonal_adapter<M31&> {d3}.nested_object(), (M31 {} << 1, 2, 3).finished() ));

  M31 p; p << 10, 11, 12;
  M31 q; q << 13, 14, 15;
  diagonal_adapter yl {p};
  static_assert(std::is_lvalue_reference_v<nested_object_of_t<decltype(yl)>>);
  EXPECT_TRUE(is_near(diagonal_of(yl), p));
  diagonal_adapter yr {(M31 {} << 13, 14, 15).finished() * 1.0};
  static_assert(not std::is_reference_v<nested_object_of_t<decltype(yr)>>);
  EXPECT_TRUE(is_near(diagonal_of(yr), q));
  yl = diagonal_adapter {q};
  EXPECT_TRUE(is_near(p, q));
  p = (M31 {} << 16, 17, 18).finished();
  EXPECT_TRUE(is_near(diagonal_of(yl), p));
}
