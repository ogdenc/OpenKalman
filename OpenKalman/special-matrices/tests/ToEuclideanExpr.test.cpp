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
 * \brief Tests relating to Eigen3::ToEuclideanExpr.
 */

#include "special-matrices.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


using numbers::pi;


namespace
{
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M42 = eigen_matrix_t<double, 4, 2>;

  using M30 = eigen_matrix_t<double, 3, dynamic_size>;
  using M20 = eigen_matrix_t<double, 2, 3>;
  using M03 = eigen_matrix_t<double, dynamic_size, 3>;
  using M02 = eigen_matrix_t<double, dynamic_size, 2>;
  using M00 = eigen_matrix_t<double, dynamic_size, dynamic_size>;

  using Car = TypedIndex<Axis, angle::Radians>;
  using Cra = TypedIndex<angle::Radians, Axis>;
  using Cara = TypedIndex<Axis, angle::Radians, Axis>;

  using To23 = ToEuclideanExpr<Car, M23>;
  using To32 = ToEuclideanExpr<Cara, M32>;
  using To02 = ToEuclideanExpr<DynamicTypedIndex, M02>;

  auto dara = DynamicTypedIndex {Cara {}};

  template<typename...Args>
  inline auto mat2(Args...args) { return MatrixTraits<M23>::make(args...); }

  template<typename...Args>
  inline auto mat3(Args...args) { return MatrixTraits<M32>::make(args...); }

  template<typename...Args>
  inline auto mat4(Args...args) { return MatrixTraits<M42>::make(args...); }
  
  template<typename C, typename T> using To = ToEuclideanExpr<C, T>;
  
}


TEST(eigen3, ToEuclideanExpr_static_checks)
{
  static_assert(writable<To<Cara, M32>>);
  static_assert(writable<To<Cara, M32&>>);
  static_assert(not writable<To<Cara, const M32>>);
  static_assert(not writable<To<Cara, const M32&>>);
  
  static_assert(modifiable<To<Cara, M32>, M42>);
  static_assert(modifiable<To<Cara, M32>, M42>);
  static_assert(not modifiable<To<Cara, M32>, M32>);
  static_assert(modifiable<To<Cara, M32>, To<Cara, M32>>);
  static_assert(modifiable<To<Cara, Eigen::Block<M32, 3, 1, true>>, To<Cara, eigen_matrix_t<double, 3, 1>>>);
  static_assert(modifiable<To<Cara, M32>, const To<Cara, M32>>);
  static_assert(modifiable<To<Cara, M32>, To<Cara, const M32>>);
  static_assert(not modifiable<To<Cara, const M32>, To<Cara, M32>>);
  static_assert(not modifiable<To<Dimensions<3>, M32>, To<Dimensions<4>, M42>>);
  static_assert(modifiable<To<Cara, M32>&, To<Cara, M32>>);
  static_assert(modifiable<To<Cara, M32&>, To<Cara, M32>>);
  static_assert(not modifiable<To<Cara, M32&>, M32>);
  static_assert(not modifiable<const To<Cara, M32>&, To<Cara, M32>>);
  static_assert(not modifiable<To<Cara, const M32&>, To<Cara, M32>>);
  static_assert(not modifiable<To<Cara, const M32>&, To<Cara, M32>>);
}


TEST(eigen3, ToEuclideanExpr_class)
{
  M42 m;
  m << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4;
  To32 d1;
  d1 << 1, 2, pi/6, pi/3, 3, 4;
  EXPECT_TRUE(is_near(d1.nested_matrix(), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(d1, m));
  //
  To32 d2 {(M32() << 1, 2, pi/6, pi/3, 3, 4).finished()};
  EXPECT_TRUE(is_near(d2, m));
  To32 d3 = d2;
  EXPECT_TRUE(is_near(d3, m));
  To32 d4 = To32{1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d4, m));
  To32 d5 {make_zero_matrix_like<M32>()};
  EXPECT_TRUE(is_near(d5, mat4(0, 0, 1, 1, 0, 0, 0, 0)));
  To32 d6 {ZeroMatrix<eigen_matrix_t<double, 3, 2>>()};
  EXPECT_TRUE(is_near(d6, mat4(0, 0, 1, 1, 0, 0, 0, 0)));
  To32 d7 = To32(ZeroMatrix<eigen_matrix_t<double, 3, 2>>());
  EXPECT_TRUE(is_near(d7, mat4(0, 0, 1, 1, 0, 0, 0, 0)));
  To32 d8 {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d8, m));
  EXPECT_TRUE(is_near(To32(ZeroMatrix<eigen_matrix_t<double, 3, 2>>()), mat4(0, 0, 1, 1, 0, 0, 0, 0)));
  //
  d5 = d1;
  EXPECT_TRUE(is_near(d5, m));
  d6 = To32 {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d5, m));
  d7 = m;
  EXPECT_TRUE(is_near(d7, m));
  d7 = M42::Zero();
  d7 = {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d7, m));
  //
  d1 += d2;
  EXPECT_TRUE(is_near(d1, mat4(2, 4, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 6, 8)));
  d1 -= To32 {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d1, mat4(1, 2, 1, 1, 0, 0, 3, 4)));
  d1 *= 3;
  EXPECT_TRUE(is_near(d1, mat4(3, 6, 1, 1, 0, 0, 9, 12)));
  d1 /= 3;
  EXPECT_TRUE(is_near(d1, mat4(1, 2, 1, 1, 0, 0, 3, 4)));
  d1 += mat4(1, 2, 1, 1, 0, 0, 3, 4);
  EXPECT_TRUE(is_near(d1, mat4(2, 4, 1, 1, 0, 0, 6, 8)));
  d1 -= mat4(1, 2, 1, 1, 0, 0, 3, 4);
  EXPECT_TRUE(is_near(d1, mat4(1, 2, 1, 1, 0, 0, 3, 4)));

  EXPECT_EQ(To32::rows(), 4);
  EXPECT_EQ(To32::cols(), 2);
  EXPECT_TRUE(is_near(make_zero_matrix_like<To32>(), M42::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<ToEuclideanExpr<Dimensions<2>, eigen_matrix_t<double, 2, 2>>>(), eigen_matrix_t<double, 2, 2>::Identity()));
}


TEST(eigen3, ToEuclideanExpr_subscripts)
{
  EXPECT_NEAR(get_element(To32 {1, 2, pi/6, pi/3, 3, 4}, 1, 1), 0.5, 1e-8);
  EXPECT_NEAR(get_element(To32 {1, 2, pi/6, pi/3, 3, 4}, 1, 0), std::sqrt(3)/2, 1e-8);

  ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 1>> e1 = {3, pi/4};
  EXPECT_EQ(e1[0], 3);
  EXPECT_NEAR(e1(1), std::sqrt(2.)/2, 1e-6);
  EXPECT_NEAR(e1(2), std::sqrt(2.)/2, 1e-6);
  ToEuclideanExpr<TypedIndex<Dimensions<2>>, eigen_matrix_t<double, 2, 2>> e2 = {1, 2, 3, 4};
  e2(0,0) = 5;
  EXPECT_EQ(e2(0, 0), 5);
  e2(0,1) = 6;
  EXPECT_EQ(e2(0, 1), 6);
  e2(1,0) = 7;
  EXPECT_EQ(e2(1, 0), 7);
  e2(1,1) = 8;
  EXPECT_EQ(e2(1, 1), 8);
  EXPECT_TRUE(is_near(e2, make_eigen_matrix<double, 2, 2>(5, 6, 7, 8)));
  EXPECT_NEAR((ToEuclideanExpr<Cara, eigen_matrix_t<double, 3, 1>>{1, pi/6, 3})(1), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((ToEuclideanExpr<Cara, eigen_matrix_t<double, 3, 1>>{1, pi/6, 3})(2), 0.5, 1e-6);
  EXPECT_NEAR((To32 {1, 2, pi/6, pi/3, 3, 4})(0, 0), 1, 1e-6);
  EXPECT_NEAR((To32 {1, 2, pi/6, pi/3, 3, 4})(1, 0), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((To32 {1, 2, pi/6, pi/3, 3, 4})(1, 1), 0.5, 1e-6);
  EXPECT_NEAR((To32 {1, 2, pi/6, pi/3, 3, 4})(2, 0), 0.5, 1e-6);
}


TEST(eigen3, ToEuclideanExpr_traits)
{
  static_assert(to_euclidean_expr<decltype(To32 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(typed_matrix_nestable<decltype(To32 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not from_euclidean_expr<decltype(To32 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not native_eigen_matrix<decltype(To32 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not eigen_matrix<decltype(To32 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not identity_matrix<decltype(To32 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not zero_matrix<decltype(To32 {1, 2, pi/6, pi/3, 3, 4})>);
  // MatrixTraits
  EXPECT_TRUE(is_near(MatrixTraits<To32>::make(make_eigen_matrix<double, 3, 2>(1, 2, pi/6, pi/3, 3, 4)),
    mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(MatrixTraits<To32>::make(1, 2, pi/6, pi/3, 3, 4),
    mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(make_zero_matrix_like<To32>(), eigen_matrix_t<double, 4, 2>::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<ToEuclideanExpr<Dimensions<2>, eigen_matrix_t<double, 2, 2>>>(), eigen_matrix_t<double, 2, 2>::Identity()));
}


TEST(eigen3, ToEuclideanExpr_overloads)
{
  M23 m23; m23 << 1, 2, 3, 4, 5, 6;
  M03 m03_2 {2,3}; m03_2 << 1, 2, 3, 4, 5, 6;
  M20 m20_3 {2,3}; m20_3 << 1, 2, 3, 4, 5, 6;
  M00 m00_23 {2,3}; m00_23 << 1, 2, 3, 4, 5, 6;

  EXPECT_TRUE(is_near(nested_matrix(To32 {1, 2, pi/6, pi/3, 3, 4}), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(To32 {1, 2, pi/6, pi/3, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(make_self_contained(To32 {1, 2, pi/6, pi/3, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));

  EXPECT_TRUE(is_near(to_euclidean<Dimensions<2>>(m23), m23));
  EXPECT_TRUE(is_near(to_euclidean<Dimensions<2>>(m20_3), m23));
  EXPECT_TRUE(is_near(to_euclidean<Dimensions<2>>(m03_2), m23));
  EXPECT_TRUE(is_near(to_euclidean<Dimensions<2>>(m00_23), m23));

  auto m33_to_ra = make_dense_writable_matrix_from<M33>(std::cos(1.), std::cos(2.), std::cos(3.), std::sin(1.), std::sin(2.), std::sin(3.), 4, 5, 6);

  EXPECT_TRUE(is_near(to_euclidean<TypedIndex<angle::Radians, Axis>>(m23), m33_to_ra));
  //EXPECT_TRUE(is_near(to_euclidean<TypedIndex<angle::Radians, Axis>>(m20_3), m33_to_ra)); \todo
  //EXPECT_TRUE(is_near(to_euclidean<TypedIndex<angle::Radians, Axis>>(m03_2), m33_to_ra));
  //EXPECT_TRUE(is_near(to_euclidean<TypedIndex<angle::Radians, Axis>>(m00_23), m33_to_ra));

  ConstantMatrix<eigen_matrix_t<double, 3, 4>, 5> c534 {};
  ConstantMatrix<eigen_matrix_t<double, 3, dynamic_size>, 5> c530_4 {4};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, 4>, 5> c504_3 {3};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5> c500_34 {3, 4};

  EXPECT_TRUE(is_near(to_euclidean<Dimensions<3>>(c534), c534));
  EXPECT_TRUE(is_near(to_euclidean<Dimensions<3>>(c530_4), c534));
  EXPECT_TRUE(is_near(to_euclidean<Dimensions<3>>(c504_3), c534));
  EXPECT_TRUE(is_near(to_euclidean<Dimensions<3>>(c500_34), c534));
  //EXPECT_TRUE(is_near(to_euclidean(DynamicTypedIndex {Dimensions<3>{}}, c534), c534));
  //EXPECT_TRUE(is_near(to_euclidean(DynamicTypedIndex {Dimensions<3>{}}, c530_4), c534));
  //EXPECT_TRUE(is_near(to_euclidean(DynamicTypedIndex {Dimensions<3>{}}, c504_3), c534));
  //EXPECT_TRUE(is_near(to_euclidean(DynamicTypedIndex {Dimensions<3>{}}, c500_34), c534));

  auto m44_to_raa = make_dense_writable_matrix_from<M44>(std::cos(5.), std::cos(5.), std::cos(5.), std::cos(5.),
    std::sin(5.), std::sin(5.), std::sin(5.), std::sin(5), 5, 5, 5, 5, 5, 5, 5, 5);

  EXPECT_TRUE(is_near(to_euclidean<TypedIndex<angle::Radians, Axis, Axis>>(c534), m44_to_raa));
  EXPECT_TRUE(is_near(to_euclidean<TypedIndex<angle::Radians, Axis, Axis>>(c530_4), m44_to_raa));
  EXPECT_TRUE(is_near(to_euclidean<TypedIndex<angle::Radians, Axis, Axis>>(c504_3), m44_to_raa));
  EXPECT_TRUE(is_near(to_euclidean<TypedIndex<angle::Radians, Axis, Axis>>(c500_34), m44_to_raa));
  //EXPECT_TRUE(is_near(to_euclidean(DynamicTypedIndex {TypedIndex<angle::Radians, Axis, Axis>{}}, c534), m44_to_raa));
  //EXPECT_TRUE(is_near(to_euclidean(DynamicTypedIndex {TypedIndex<angle::Radians, Axis, Axis>{}}, c530_4), m44_to_raa));
  //EXPECT_TRUE(is_near(to_euclidean(DynamicTypedIndex {TypedIndex<angle::Radians, Axis, Axis>{}}, c504_3), m44_to_raa));
  //EXPECT_TRUE(is_near(to_euclidean(DynamicTypedIndex {TypedIndex<angle::Radians, Axis, Axis>{}}, c500_34), m44_to_raa));

  ZeroMatrix<eigen_matrix_t<double, 2, 3>> z23;
  ZeroMatrix<eigen_matrix_t<double, 2, dynamic_size>> z20_3 {3};
  ZeroMatrix<eigen_matrix_t<double, dynamic_size, 3>> z03_2 {2};
  ZeroMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>> z00_23 {2, 3};

  EXPECT_TRUE(is_near(to_euclidean<Dimensions<2>>(z23), z23));
  EXPECT_TRUE(is_near(to_euclidean<Dimensions<2>>(z20_3), z23));
  EXPECT_TRUE(is_near(to_euclidean<Dimensions<2>>(z03_2), z23));
  EXPECT_TRUE(is_near(to_euclidean<Dimensions<2>>(z00_23), z23));

  auto z33 = ZeroMatrix<eigen_matrix_t<double, 3, 3>> {};

  EXPECT_TRUE(is_near(to_euclidean<Axis, TypedIndex<angle::Radians>>(z23), z33));
  EXPECT_TRUE(is_near(to_euclidean<Axis, TypedIndex<angle::Radians>>(z20_3), z33));
  EXPECT_TRUE(is_near(to_euclidean<Axis, TypedIndex<angle::Radians>>(z03_2), z33));
  EXPECT_TRUE(is_near(to_euclidean<Axis, TypedIndex<angle::Radians>>(z00_23), z33));

  EXPECT_TRUE(is_near(from_euclidean(To32 {1, 2, 2*pi + pi/6, -4*pi + pi/3, 3, 4}), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(from_euclidean<Cara>(To32 {1, 2, 2*pi + pi/6, -4*pi + pi/3, 3, 4}), mat3(1, 2, pi/6, pi/3, 3, 4)));

  EXPECT_TRUE(is_near(to_diagonal(ToEuclideanExpr<Cara, eigen_matrix_t<double, 3, 1>>{1, pi/6, 3}), DiagonalMatrix {1., std::sqrt(3)/2, 0.5, 3}));

  EXPECT_TRUE(is_near(diagonal_of(To23 {1, 2, 3, pi/6, pi/3, pi/4}), make_eigen_matrix<double, 3, 1>(1, 0.5, std::sqrt(2.)/2)));
  EXPECT_TRUE(is_near(transpose(To32 {1, 2, pi/6, pi/3, 3, 4}), make_eigen_matrix<double, 2, 4>(1, std::sqrt(3)/2, 0.5, 3, 2, 0.5, std::sqrt(3)/2, 4)));
  EXPECT_TRUE(is_near(adjoint(To32 {1, 2, pi/6, pi/3, 3, 4}), make_eigen_matrix<double, 2, 4>(1, std::sqrt(3)/2, 0.5, 3, 2, 0.5, std::sqrt(3)/2, 4)));
  EXPECT_NEAR(determinant(To23 {1, 2, 3, pi/6, pi/3, pi/4}), 0.25 * (3 * std::sqrt(2) - 3 * std::sqrt(6.) + 6), 1e-6);
  EXPECT_NEAR(trace(To23 {1, 2, 3, pi/6, pi/3, pi/4}), 1.5 + std::sqrt(2.)/2, 1e-6);
  EXPECT_TRUE(is_near(solve(
    To23 {1, 2, 3, pi/6, pi/3, pi/4},
    make_eigen_matrix<double, 3, 1>(14, std::sqrt(3)/2 + 1 + std::sqrt(2)*3/2, 0.5 + std::sqrt(3) + std::sqrt(2)*3/2)),
    make_eigen_matrix<double, 3, 1>(1, 2, 3)));
  EXPECT_TRUE(is_near(average_reduce<1>(To32 {1, 2, pi/6, pi/3, 3, 4}),
    make_eigen_matrix<double, 4, 1>(1.5, 0.5*(std::sqrt(3)/2 + 0.5), 0.5*(std::sqrt(3)/2 + 0.5), 3.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(To32 {1, 2, pi/6, pi/3, 3, 4}),
    make_eigen_matrix<double, 1, 2>(4.5/4 + std::sqrt(3)/8, 6.5/4 + std::sqrt(3)/8)));
  EXPECT_TRUE(is_near(LQ_decomposition(To23 {1, 2, 3, pi/6, pi/3, pi/4}),
    LQ_decomposition(make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2))));
  EXPECT_TRUE(is_near(QR_decomposition(To23 {1, 2, 3, pi/6, pi/3, pi/4}),
    QR_decomposition(make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2))));

  using N = std::normal_distribution<double>;
  auto m = make_dense_writable_matrix_from(make_zero_matrix_like<eigen_matrix_t<double, 3, 2>>());
  for (int i=0; i<100; i++)
  {
    m = (m * i + from_euclidean(randomize<To32>(N {1.0, 0.3}))) / (i + 1);
  }
  auto offset = eigen_matrix_t<double, 3, 2>::Constant(1);
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-6));
}


TEST(eigen3, ToEuclideanExpr_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4},
    ToEuclideanExpr<Cra, eigen_matrix_t<double, 2, 3>> {pi/4, pi/3, pi/6, 4, 5, 6}),
    make_eigen_matrix<double, 6, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6)));
  EXPECT_TRUE(is_near(concatenate_horizontal(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4},
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {4., 5, 6, pi/4, pi/3, pi/6}),
    make_eigen_matrix<double, 3, 6>(
      1, 2, 3, 4, 5, 6,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2, std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2, std::sqrt(2)/2, std::sqrt(3)/2, 0.5)));
  EXPECT_TRUE(is_near(split_vertical(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 2>> {1., 2, pi/6, pi/3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<3, 3>(
    ToEuclideanExpr<TypedIndex<Axis, angle::Radians, angle::Radians, Axis>, eigen_matrix_t<double, 4, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4,
      pi/4, pi/3, pi/6,
      4, 5, 6}),
    std::tuple {make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
               make_eigen_matrix<double, 3, 3>(
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                 4, 5, 6)}));
  auto a1 = ToEuclideanExpr<TypedIndex<Polar<>, angle::Radians, Axis>, eigen_matrix_t<double, 4, 3>> {
    1., 2, 3,
    pi/6, pi/3, pi/4,
    pi/4, pi/3, pi/6,
    4, 5, 6};
  EXPECT_TRUE(is_near(split_vertical<3, 3>(a1),
    std::tuple {make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
    make_eigen_matrix<double, 3, 3>(
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                 4, 5, 6)}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(
    ToEuclideanExpr<TypedIndex<Axis, angle::Radians, angle::Radians, Axis>, eigen_matrix_t<double, 4, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4,
      pi/4, pi/3, pi/6,
      4, 5, 6}),
    std::tuple {
      make_eigen_matrix<double, 3, 3>(
        1, 2, 3,
        std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
        0.5, std::sqrt(3)/2, std::sqrt(2)/2),
      make_eigen_matrix<double, 2, 3>(
        std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
        std::sqrt(2)/2, std::sqrt(3)/2, 0.5)}));
  EXPECT_TRUE(is_near(split_vertical<Car, Cra>(
    ToEuclideanExpr<TypedIndex<Axis, angle::Radians, angle::Radians, Axis>, eigen_matrix_t<double, 4, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4,
      pi/4, pi/3, pi/6,
      4, 5, 6}),
    std::tuple {make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
               make_eigen_matrix<double, 3, 3>(
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                 4, 5, 6)}));
  EXPECT_TRUE(is_near(split_vertical<Car, TypedIndex<angle::Radians>>(
    ToEuclideanExpr<TypedIndex<Axis, angle::Radians, angle::Radians, Axis>, eigen_matrix_t<double, 4, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4,
      pi/4, pi/3, pi/6,
      4, 5, 6}),
    std::tuple {
    make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
    make_eigen_matrix<double, 2, 3>(
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5)}));

  EXPECT_TRUE(is_near(split_horizontal(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 2>> {1., 2, pi/6, pi/3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<3, 3>(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 6>> {
      1., 2, 3, 4, 5, 6,
      pi/6, pi/3, pi/4, pi/4, pi/3, pi/6}),
    std::tuple {make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
               make_eigen_matrix<double, 3, 3>(
                 4, 5, 6,
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5)}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(
    ToEuclideanExpr<Polar<>, eigen_matrix_t<double, 2, 6>> {
      1., 2, 3, 4, 5, 6,
      pi/6, pi/3, pi/4, pi/4, pi/3, pi/6}),
    std::tuple {
    make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
    make_eigen_matrix<double, 3, 2>(
      4, 5,
      std::sqrt(2)/2, 0.5,
      std::sqrt(2)/2, std::sqrt(3)/2)}));

  EXPECT_TRUE(is_near(split_diagonal<Axis, angle::Radians>(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4}),
    std::tuple {make_eigen_matrix<double, 1, 1>(1),
               make_eigen_matrix<double, 2, 2>(
                 0.5, std::sqrt(3)/2,
                 std::sqrt(3)/2, 0.5)}));
  EXPECT_TRUE(is_near(split_diagonal<1, 2>(
    ToEuclideanExpr<Polar<>, eigen_matrix_t<double, 2, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4}),
    std::tuple {make_eigen_matrix<double, 1, 1>(1),
               make_eigen_matrix<double, 2, 2>(
                 0.5, std::sqrt(3)/2,
                 std::sqrt(3)/2, 0.5)}));

  EXPECT_TRUE(is_near(column(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}, 2),
    make_eigen_matrix<double, 3, 1>(3, std::sqrt(2)/2, std::sqrt(2)/2)));
  EXPECT_TRUE(is_near(column<1>(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 3, 1>(2, 0.5, std::sqrt(3)/2)));

  EXPECT_TRUE(is_near(row(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}, 2),
    make_eigen_matrix<double, 1, 3>(0.5, std::sqrt(3)/2, std::sqrt(2)/2)));
  EXPECT_TRUE(is_near(row<1>(
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 1, 3>(std::sqrt(3)/2, 0.5, std::sqrt(2)/2)));


  // \todo Add tests for eigen3-matrix-overloads versions of apply_columnwise and apply_rowwise that start with an native_eigen_matrix and return a euclidean_expr


  auto b = ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4};
  EXPECT_TRUE(is_near(apply_columnwise([](auto& col){ col *= 3; }, b),
    make_eigen_matrix<double, 3, 3>(
      3, 6, 9,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2)));
  b = ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4};
  EXPECT_TRUE(is_near(apply_columnwise([](auto& col, std::size_t i){ col *= i + 1; }, b),
    make_eigen_matrix<double, 3, 3>(
      1, 4, 9,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col * 2); },
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 3, 3>(
      2, 4, 6,
      std::sqrt(3), 1, std::sqrt(2),
      1, std::sqrt(3), std::sqrt(2))));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col * i); },
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 3, 3>(
      0, 2, 6,
      0, 0.5, std::sqrt(2),
      0, std::sqrt(3)/2, std::sqrt(2))));
  EXPECT_TRUE(is_near(apply_columnwise<3>([](){ return ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 1>> {1., pi/6}; }),
    make_eigen_matrix<double, 3, 3>(
      1, 1, 1,
      std::sqrt(3)/2, std::sqrt(3)/2, std::sqrt(3)/2,
      0.5, 0.5, 0.5)));
  EXPECT_TRUE(is_near(apply_columnwise<3>([](std::size_t i){ return make_self_contained(ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 1>> {1., pi/6} * (i + 1)); }),
    make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, std::sqrt(3), std::sqrt(3)*3/2,
      0.5, 1, 1.5)));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row * 2); },
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 3, 3>(
      2, 4, 6,
      std::sqrt(3), 1, std::sqrt(2),
      1, std::sqrt(3), std::sqrt(2))));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row * i); },
    ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 3, 3>(
      0, 0, 0,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      1.0, std::sqrt(3), std::sqrt(2))));
  EXPECT_TRUE(is_near(apply_rowwise<2>([](){ return ToEuclideanExpr<Axis, eigen_matrix_t<double, 1, 2>> {1, 2}; }),
    make_eigen_matrix<double, 2, 2>(1, 2, 1, 2)));
  EXPECT_TRUE(is_near(apply_rowwise<3>([](std::size_t i){ return make_self_contained(ToEuclideanExpr<Axis, eigen_matrix_t<double, 1, 2>> {1, 2} * (i + 1)); }),
    make_eigen_matrix<double, 3, 2>(1, 2, 2, 4, 3, 6)));

  //
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x * 3; },
      ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 3, 3>(
      3, 6, 9,
      std::sqrt(3)*3/2, 1.5, std::sqrt(2)*3/2,
      1.5, std::sqrt(3)*3/2, std::sqrt(2)*3/2)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x * (j + 1); },
      ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 3, 3>(
      1, 4, 9,
      std::sqrt(3)/2, 1, std::sqrt(2)*3/2,
      0.5, std::sqrt(3), std::sqrt(2)*3/2)));
}


TEST(eigen3, ToEuclideanExpr_arithmetic)
{
  EXPECT_TRUE(is_near(To32 {1, 2, pi/6, pi/3, 3, 4} + To32 {1, 2, pi/6, pi/3, 3, 4}, mat4(2, 4, std::sqrt(3), 1, 1, std::sqrt(3), 6, 8)));
  EXPECT_TRUE(is_near(To32 {1, 2, pi/6, pi/3, 3, 4} - To32 {1, 2, pi/6, pi/3, 3, 4}, M42::Zero()));
  EXPECT_TRUE(is_near(To32 {1, 2, pi/6, pi/3, 3, 4} * 2, mat4(2, 4, std::sqrt(3), 1, 1, std::sqrt(3), 6, 8)));
  EXPECT_TRUE(is_near(2 * To32 {1, 2, pi/6, pi/3, 3, 4}, mat4(2, 4, std::sqrt(3), 1, 1, std::sqrt(3), 6, 8)));
  EXPECT_TRUE(is_near(To32 {1, 2, pi/6, pi/3, 3, 4} / 2, mat4(0.5, 1, std::sqrt(3)/4, 0.25, 0.25, std::sqrt(3)/4, 1.5, 2)));
  EXPECT_TRUE(is_near(-To32 {1, 2, pi/6, pi/3, 3, 4}, mat4(-1, -2, -std::sqrt(3)/2, -0.5, -0.5, -std::sqrt(3)/2, -3, -4)));
  EXPECT_TRUE(is_near(To32 {1, 2, pi/6, pi/3, 3, 4} * DiagonalMatrix {1., 2}, mat4(1, 4, std::sqrt(3)/2, 1, 0.5, std::sqrt(3), 3, 8)));
}


TEST(eigen3, ToEuclideanExpr_references)
{
  M22 m, n;
  m << pi/6, pi/4, 1, 2;
  n << pi/4, pi/3, 3, 4;
  M32 me, ne;
  me << std::sqrt(3)/2, std::sqrt(2)/2, 0.5, std::sqrt(2)/2, 1, 2;
  ne << std::sqrt(2)/2, 0.5, std::sqrt(2)/2, std::sqrt(3)/2, 3, 4;
  using To = ToEuclideanExpr<Cra, M22>;
  To x = To {m};
  ToEuclideanExpr<Cra, M22&> x_lvalue = x;
  EXPECT_TRUE(is_near(x_lvalue, me));
  x = To {n};
  EXPECT_TRUE(is_near(x_lvalue, ne));
  x_lvalue = To {m};
  EXPECT_TRUE(is_near(x, me));
}
