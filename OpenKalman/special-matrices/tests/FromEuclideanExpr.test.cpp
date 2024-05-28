/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021 Christopher Lee Ogden <ogden@gatech.edu>
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
  using M42 = eigen_matrix_t<double, 4, 2>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M11 = eigen_matrix_t<double, 1, 1>;

  using M3x = eigen_matrix_t<double, 3, dynamic_size_v>;
  using M2x = eigen_matrix_t<double, 2, dynamic_size_v>;
  using Mx3 = eigen_matrix_t<double, dynamic_size_v, 3>;
  using Mx2 = eigen_matrix_t<double, dynamic_size_v, 2>;
  using Mxx = eigen_matrix_t<double, dynamic_size_v, dynamic_size_v>;

  using Car = FixedDescriptor<Axis, angle::Radians>;
  using Cra = FixedDescriptor<angle::Radians, Axis>;
  using Cara = FixedDescriptor<Axis, angle::Radians, Axis>;

  auto dara = DynamicTypedIndex {Cara {}};

  using From32 = FromEuclideanExpr<Car, M32>;
  using From42 = FromEuclideanExpr<Cara, M42>;
  using From02 = FromEuclideanExpr<DynamicTypedIndex<double>, Mx2>;
  using FromTo32 = FromEuclideanExpr<Cara, ToEuclideanExpr<Cara, M32>>;
  using FromTo02 = FromEuclideanExpr<DynamicTypedIndex<double>, ToEuclideanExpr<DynamicTypedIndex<double>, Mx2>>;

  template<typename...Args>
  inline auto mat3(Args...args) { return make_dense_writable_matrix_from<M32>(args...); }

  template<typename...Args>
  inline auto mat4(Args...args) { return make_dense_writable_matrix_from<M42>(args...); }
  
  template<typename C, typename T> using From = FromEuclideanExpr<C, T>;
}


TEST(special_matrices, FromEuclideanExpr_static_checks)
{
  static_assert(writable<From<Cara, M42>>);
  static_assert(writable<From<Cara, M42&>>);
  static_assert(not writable<From<Cara, const M42>>);
  static_assert(not writable<From<Cara, const M42&>>);
  
  static_assert(modifiable<From<Cara, M42>, M32>);
  static_assert(modifiable<From<Cara, M42>, M32>);
  static_assert(not modifiable<From<Cara, M42>, M42>);
  static_assert(modifiable<From<Cara, M42>, From<Cara, M42>>);
  static_assert(modifiable<From<Cara, M42>, const From<Cara, M42>>);
  static_assert(modifiable<From<Cara, M42>, From<Cara, const M42>>);
  static_assert(not modifiable<From<Cara, const M42>, From<Cara, M42>>);
  static_assert(not modifiable<From<Dimensions<3>, M32>, From<Dimensions<4>, M42>>);
  static_assert(modifiable<From<Cara, M42>&, From<Cara, M42>>);
  static_assert(modifiable<From<Cara, M42&>, From<Cara, M42>>);
  static_assert(not modifiable<From<Cara, M42&>, M42>);
  static_assert(not modifiable<const From<Cara, M42>&, From<Cara, M42>>);
  static_assert(not modifiable<From<Cara, const M42&>, From<Cara, M42>>);
  static_assert(not modifiable<From<Cara, const M42>&, From<Cara, M42>>);
}


TEST(special_matrices, FromEuclideanExpr_class)
{
  M32 m;
  m << 1, 2, pi/6, pi/3, 3, 4;

  From42 d1;
  d1 << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4;
  EXPECT_TRUE(is_near(d1.nested_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(d1, m));

  /*From02 d02_4 {dara, Mx2 {4, 2}};
  d02_4 << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4;
  EXPECT_TRUE(is_near(d02_4.nested_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(d02_4, m));

  From02 d02_4a {dara, 4, 2};
  d02_4a << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4;
  EXPECT_TRUE(is_near(d02_4a.nested_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(d02_4a, m));*/

  FromTo32 d1b;
  d1b << 1, 2, pi/6, pi/3, 3, 4;
  EXPECT_TRUE(is_near(d1b.nested_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(d1b, mat3(1, 2, pi/6, pi/3, 3, 4)));

  /*FromTo02 ft02_3 {dara, Mx2 {3, 2}};
  ft02_3 << 1, 2, pi/6, pi/3, 3, 4;
  EXPECT_TRUE(is_near(ft02_3.nested_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(ft02_3, mat3(1, 2, pi/6, pi/3, 3, 4)));

  FromTo02 ft02_3a {dara, 3, 2};
  ft02_3a << 1, 2, pi/6, pi/3, 3, 4;
  EXPECT_TRUE(is_near(ft02_3a.nested_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(ft02_3a, mat3(1, 2, pi/6, pi/3, 3, 4)));*/

  From42 d2 {(M42() << 1, 2, std::sqrt(3.)/2, 0.5, 0.5, std::sqrt(3.)/2, 3, 4).finished()};
  EXPECT_TRUE(is_near(d2, m));
  From42 d3 = d2;
  EXPECT_TRUE(is_near(d3, m));
  From42 d4 = From42{1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d4, m));
  From42 d5 {make_zero_matrix_like<M42>()};
  EXPECT_TRUE(is_near(d5, mat3(0, 0, 0, 0, 0, 0)));
  From42 d6 {ZeroMatrix<eigen_matrix_t<double, 4, 2>>()};
  EXPECT_TRUE(is_near(d6, mat3(0, 0, 0, 0, 0, 0)));
  From42 d7 = From42(ZeroMatrix<eigen_matrix_t<double, 4, 2>>());
  EXPECT_TRUE(is_near(d7, mat3(0, 0, 0, 0, 0, 0)));
  From42 d8 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d8, m));
  EXPECT_TRUE(is_near(From42(ZeroMatrix<eigen_matrix_t<double, 4, 2>>()), mat3(0, 0, 0, 0, 0, 0)));
  FromTo32 d9 {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d9.nested_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(d9, mat3(1, 2, pi/6, pi/3, 3, 4)));
  //
  d5 = d1;
  EXPECT_TRUE(is_near(d5, m));
  d6 = From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d5, m));
  d7 = m;
  EXPECT_TRUE(is_near(d7, m));
  d7 = M32::Zero();
  d7 = {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d7, m));
  //d8 = M32::Zero();
  //d8 = d02_4;
  //EXPECT_TRUE(is_near(d8, m));
  d9 = M32::Zero();
  d9 = {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d9, m));
  //
  d1 += d2;
  EXPECT_TRUE(is_near(d1, m + m));
  d1 -= m;
  EXPECT_TRUE(is_near(d1, m));
  d1 += m;
  EXPECT_TRUE(is_near(d1, m + m));
  d1 -= From42 {1, 2, std::sqrt(3.)/2, 0.5, 0.5, std::sqrt(3.)/2, 3, 4};
  EXPECT_TRUE(is_near(d1, m));
  d1 *= 3;
  EXPECT_TRUE(is_near(d1, m * 3));
  d1 /= 3;
  EXPECT_TRUE(is_near(d1, m));

  EXPECT_EQ(From42::rows(), 3);
  EXPECT_EQ(From42::cols(), 2);
  EXPECT_TRUE(is_near(make_zero_matrix_like<From42>(), M32::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<FromEuclideanExpr<Dimensions<2>, eigen_matrix_t<double, 2, 2>>>(), eigen_matrix_t<double, 2, 2>::Identity()));

  FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 1>> e1 = {3, std::sqrt(2.)/2, std::sqrt(2.)/2};

  EXPECT_EQ(e1[0], 3);
  EXPECT_NEAR(e1(1), pi/4, 1e-6);
  EXPECT_EQ(d1(0, 1), 2);
  EXPECT_EQ(d1(2, 1), 4);
}


TEST(special_matrices, FromEuclideanExpr_subscripts)
{
  auto el = FromTo32 {1, 2, pi/6, pi/3, 3, 4};
  set_element(el, pi/2, 1, 0);
  EXPECT_NEAR(get_element(el, 1, 0), pi/2, 1e-8);
  set_element(el, 3.1, 2, 0);
  EXPECT_NEAR(get_element(el, 2, 0), 3.1, 1e-8);
  EXPECT_NEAR(get_element(From32 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}, 1, 1), pi/3, 1e-8);

  FromEuclideanExpr<FixedDescriptor<Dimensions<2>>, eigen_matrix_t<double, 2, 2>> e2 = {1, 2, 3, 4};
  e2(0,0) = 5;
  EXPECT_EQ(e2(0, 0), 5);
  e2(0,1) = 6;
  EXPECT_EQ(e2(0, 1), 6);
  e2(1,0) = 7;
  EXPECT_EQ(e2(1, 0), 7);
  e2(1,1) = 8;
  EXPECT_EQ(e2(1, 1), 8);
  EXPECT_TRUE(is_near(e2, make_eigen_matrix<double, 2, 2>(5, 6, 7, 8)));
  EXPECT_NEAR((FromEuclideanExpr<Cara, eigen_matrix_t<double, 4, 1>>{1., std::sqrt(3)/2, 0.5, 3})(1), pi/6, 1e-6);
  EXPECT_NEAR((FromEuclideanExpr<Cara, eigen_matrix_t<double, 4, 1>>{1., std::sqrt(3)/2, 0.5, 3})(2), 3, 1e-6);
  EXPECT_NEAR((From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})(0, 0), 1, 1e-6);
  EXPECT_NEAR((From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})(1, 0), pi/6, 1e-6);
  EXPECT_NEAR((From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})(1, 1), pi/3, 1e-6);
  EXPECT_NEAR((From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})(2, 0), 3, 1e-6);
}


TEST(special_matrices, FromEuclideanExpr_traits)
{
  static_assert(from_euclidean_expr<decltype(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(typed_matrix_nestable<decltype(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(not to_euclidean_expr<decltype(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(not native_eigen_matrix<decltype(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(not identity_matrix<decltype(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(not zero_matrix<decltype(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);

  EXPECT_TRUE(is_near(make_zero_matrix_like<From42>(), eigen_matrix_t<double, 3, 2>::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<FromEuclideanExpr<Dimensions<2>, eigen_matrix_t<double, 2, 2>>>(), eigen_matrix_t<double, 2, 2>::Identity()));
}


TEST(special_matrices, FromEuclideanExpr_properties)
{
  static_assert(wrappable<M23>);
  EXPECT_TRUE(get_wrappable(M23 {}));
  static_assert(not wrappable<From32>);
  EXPECT_TRUE(not_wrappable(From32 {}));
}


TEST(special_matrices, FromEuclideanExpr_overloads)
{
  M23 m23; m23 << 1, 2, 3, 4, 5, 6;
  Mx3 mx3_2 {2,3}; mx3_2 << 1, 2, 3, 4, 5, 6;
  M2x m2x_3 {2,3}; m2x_3 << 1, 2, 3, 4, 5, 6;
  Mxx mxx_23 {2,3}; mxx_23 << 1, 2, 3, 4, 5, 6;

  M32 m32; m32 << 1, 4, 2, 5, 3, 6;
  Mx2 mx2_3 {3,2}; mx2_3 << 1, 4, 2, 5, 3, 6;
  M3x m3x_2 {3,2}; m3x_2 << 1, 4, 3, 5, 3, 6;
  M3x mxx_32 {3,2}; mxx_32 << 1, 4, 3, 5, 3, 6;

  EXPECT_TRUE(is_near(nested_matrix(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(make_self_contained(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat3(1, 2, pi/6, pi/3, 3, 4)));

  EXPECT_TRUE(is_near(from_euclidean<Dimensions<2>>(m23), m23));
  EXPECT_TRUE(is_near(from_euclidean<Dimensions<2>>(m2x_3), m23));
  EXPECT_TRUE(is_near(from_euclidean<Dimensions<2>>(mx3_2), m23));
  EXPECT_TRUE(is_near(from_euclidean<Dimensions<2>>(mxx_23), m23));

  auto m22_from_ra = make_dense_writable_matrix_from<M22>(std::atan2(2.,1.), std::atan2(5.,4.), 3, 6);

  EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians, Axis>>(m32), m22_from_ra));
  //EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians, Axis>>(m3x_2), m22_from_ra));
  //EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians, Axis>>(mx2_3), m22_from_ra));
  //EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians, Axis>>(mxx_32), m22_from_ra));

  EXPECT_TRUE(is_near(to_euclidean(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(to_euclidean<Cara>(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));

  EXPECT_TRUE(is_near(wrap_angles<Dimensions<2>>(m23), m23));
  EXPECT_TRUE(is_near(wrap_angles<Dimensions<2>>(m2x_3), m23));
  EXPECT_TRUE(is_near(wrap_angles<Dimensions<2>>(mx3_2), m23));
  EXPECT_TRUE(is_near(wrap_angles<Dimensions<2>>(mxx_23), m23));

  auto m23_wrap_ar = make_dense_writable_matrix_from<M23>(1, 2, 3, 4-pi, 5-pi, 6-pi);

  ConstantAdapter<eigen_matrix_t<double, 3, 4>, 5> c534 {};
  ConstantAdapter<eigen_matrix_t<double, 3, dynamic_size_v>, 5> c530_4 {4};
  ConstantAdapter<eigen_matrix_t<double, dynamic_size_v, 4>, 5> c504_3 {3};
  ConstantAdapter<eigen_matrix_t<double, dynamic_size_v, dynamic_size_v>, 5> c500_34 {3, 4};

  EXPECT_TRUE(is_near(from_euclidean<Dimensions<3>>(c534), c534));
  EXPECT_TRUE(is_near(from_euclidean<Dimensions<3>>(c530_4), c534));
  EXPECT_TRUE(is_near(from_euclidean<Dimensions<3>>(c504_3), c534));
  EXPECT_TRUE(is_near(from_euclidean<Dimensions<3>>(c500_34), c534));
  //EXPECT_TRUE(is_near(from_euclidean(DynamicTypedIndex {Dimensions<3>{}}, c534), c534));
  //EXPECT_TRUE(is_near(from_euclidean(DynamicTypedIndex {Dimensions<3>{}}, c530_4), c534));
  //EXPECT_TRUE(is_near(from_euclidean(DynamicTypedIndex {Dimensions<3>{}}, c504_3), c534));
  //EXPECT_TRUE(is_near(from_euclidean(DynamicTypedIndex {Dimensions<3>{}}, c500_34), c534));

  auto m24_from_ra = make_dense_writable_matrix_from<M24>(pi/4, pi/4, pi/4, pi/4, 5, 5, 5, 5)

  EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians, Axis>>(c534), m24_from_ra));
  EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians, Axis>>(c530_4), m24_from_ra));
  EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians, Axis>>(c504_3), m24_from_ra));
  EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians, Axis>>(c500_34), m24_from_ra));
  //EXPECT_TRUE(is_near(from_euclidean(DynamicTypedIndex {FixedDescriptor<angle::Radians, Axis>{}}, c534), m24_from_ra));
  //EXPECT_TRUE(is_near(from_euclidean(DynamicTypedIndex {FixedDescriptor<angle::Radians, Axis>{}}, c530_4), m24_from_ra));
  //EXPECT_TRUE(is_near(from_euclidean(DynamicTypedIndex {FixedDescriptor<angle::Radians, Axis>{}}, c504_3), m24_from_ra));
  //EXPECT_TRUE(is_near(from_euclidean(DynamicTypedIndex {FixedDescriptor<angle::Radians, Axis>{}}, c500_34), m24_from_ra));

  ZeroMatrix<eigen_matrix_t<double, 2, 3>> z23;
  ZeroMatrix<eigen_matrix_t<double, 2, dynamic_size_v>> z20_3 {3};
  ZeroMatrix<eigen_matrix_t<double, dynamic_size_v, 3>> z03_2 {2};
  ZeroMatrix<eigen_matrix_t<double, dynamic_size_v, dynamic_size_v>> z00_23 {2, 3};

  EXPECT_TRUE(is_near(from_euclidean<Dimensions<2>>(z23), z23));
  EXPECT_TRUE(is_near(from_euclidean<Dimensions<2>>(z20_3), z23));
  EXPECT_TRUE(is_near(from_euclidean<Dimensions<2>>(z03_2), z23));
  EXPECT_TRUE(is_near(from_euclidean<Dimensions<2>>(z00_23), z23));

  ZeroMatrix<eigen_matrix_t<double, 1, 3>> z13;

  EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians>>(z23), z13));
  EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians>>(z20_3), z13));
  EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians>>(z03_2), z13));
  EXPECT_TRUE(is_near(from_euclidean<FixedDescriptor<angle::Radians>>(z00_23), z13));

  EXPECT_TRUE(is_near(wrap_angles<FixedDescriptor<Axis, angle::Radians>>(m23), m23_wrap_ar));
  //EXPECT_TRUE(is_near(wrap_angles<FixedDescriptor<Axis, angle::Radians>>(m2x_3), m23_wrap_ar));
  //EXPECT_TRUE(is_near(wrap_angles<FixedDescriptor<Axis, angle::Radians>>(mx3_2), m23_wrap_ar));
  //EXPECT_TRUE(is_near(wrap_angles<FixedDescriptor<Axis, angle::Radians>>(mxx_23), m23_wrap_ar));

  EXPECT_TRUE(is_near(wrap_angles<Dimensions<3>>(c534), c534));
  EXPECT_TRUE(is_near(wrap_angles<Dimensions<3>>(c530_4), c534));
  EXPECT_TRUE(is_near(wrap_angles<Dimensions<3>>(c504_3), c534));
  EXPECT_TRUE(is_near(wrap_angles<Dimensions<3>>(c500_34), c534));
  //EXPECT_TRUE(is_near(wrap_angles(DynamicTypedIndex {Dimensions<3>{}}, c534), c534));
  //EXPECT_TRUE(is_near(wrap_angles(DynamicTypedIndex {Dimensions<3>{}}, c530_4), c534));
  //EXPECT_TRUE(is_near(wrap_angles(DynamicTypedIndex {Dimensions<3>{}}, c504_3), c534));
  //EXPECT_TRUE(is_near(wrap_angles(DynamicTypedIndex {Dimensions<3>{}}, c500_34), c534));

  auto m34_wrap_ara = make_dense_writable_matrix_from<M34>(5,5,5,5,5-pi,5-pi,5-pi,5-pi,5,5,5,5);

  EXPECT_TRUE(is_near(wrap_angles<Axis, FixedDescriptor<angle::Radians, Axis, Axis>>(c534), m34_wrap_ara));
  EXPECT_TRUE(is_near(wrap_angles<Axis, FixedDescriptor<angle::Radians, Axis, Axis>>(c530_4), m34_wrap_ara));
  EXPECT_TRUE(is_near(wrap_angles<Axis, FixedDescriptor<angle::Radians, Axis, Axis>>(c504_3), m34_wrap_ara));
  EXPECT_TRUE(is_near(wrap_angles<Axis, FixedDescriptor<angle::Radians, Axis, Axis>>(c500_34), m34_wrap_ara));
  //EXPECT_TRUE(is_near(wrap_angles(DynamicTypedIndex {FixedDescriptor<angle::Radians, Axis, Axis>{}}, c534), m34_wrap_ara));
  //EXPECT_TRUE(is_near(wrap_angles(DynamicTypedIndex {FixedDescriptor<angle::Radians, Axis, Axis>{}}, c530_4), m34_wrap_ara));
  //EXPECT_TRUE(is_near(wrap_angles(DynamicTypedIndex {FixedDescriptor<angle::Radians, Axis, Axis>{}}, c504_3), m34_wrap_ara));
  //EXPECT_TRUE(is_near(wrap_angles(DynamicTypedIndex {FixedDescriptor<angle::Radians, Axis, Axis>{}}, c500_34), m34_wrap_ara));

  EXPECT_TRUE(is_near(wrap_angles<Dimensions<2>>(z23), z23));
  EXPECT_TRUE(is_near(wrap_angles<Dimensions<2>>(z20_3), z23));
  EXPECT_TRUE(is_near(wrap_angles<Dimensions<2>>(z03_2), z23));
  EXPECT_TRUE(is_near(wrap_angles<Dimensions<2>>(z00_23), z23));

  EXPECT_TRUE(is_near(wrap_angles<Axis, FixedDescriptor<angle::Radians>>(z23), z23));
  EXPECT_TRUE(is_near(wrap_angles<Axis, FixedDescriptor<angle::Radians>>(z20_3), z23));
  EXPECT_TRUE(is_near(wrap_angles<Axis, FixedDescriptor<angle::Radians>>(z03_2), z23));
  EXPECT_TRUE(is_near(wrap_angles<Axis, FixedDescriptor<angle::Radians>>(z00_23), z23));

  EXPECT_TRUE(is_near(to_diagonal(FromEuclideanExpr<Cara, eigen_matrix_t<double, 4, 1>>{1., std::sqrt(3)/2, 0.5, 3}), DiagonalMatrix {1, pi/6, 3}));
  EXPECT_TRUE(is_near(diagonal_of(From32 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}), make_eigen_matrix<double, 2, 1>(1, pi/3)));
  EXPECT_TRUE(is_near(transpose(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), make_eigen_matrix<double, 2, 3>(1, pi/6, 3, 2, pi/3, 4)));
  EXPECT_TRUE(is_near(adjoint(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), make_eigen_matrix<double, 2, 3>(1, pi/6, 3, 2, pi/3, 4)));
  EXPECT_NEAR(determinant(From32 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}), 0.0, 1e-6);
  EXPECT_NEAR(trace(From32 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}), 1 + pi/3, 1e-6);
  EXPECT_TRUE(is_near(solve(
    From32 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2},
    make_eigen_matrix<double, 2, 1>(5, pi*5/6)),
    make_eigen_matrix<double, 2, 1>(1, 2)));
  EXPECT_TRUE(is_near(average_reduce<1>(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), make_eigen_matrix<double, 3, 1>(1.5, pi/4, 3.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), make_eigen_matrix<double, 1, 2>(4./3 + pi/18, 2 + pi/9)));
  EXPECT_TRUE(is_near(LQ_decomposition(From32 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}),
    LQ_decomposition(make_eigen_matrix<double, 2, 2>(1, 2, pi/6, pi/3))));
  EXPECT_TRUE(is_near(QR_decomposition(From32 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}),
    QR_decomposition(make_eigen_matrix<double, 2, 2>(1, 2, pi/6, pi/3))));

  using N = std::normal_distribution<double>;
  auto m = make_dense_writable_matrix_from(make_zero_matrix_like<eigen_matrix_t<double, 4, 2>>());
  for (int i=0; i<100; i++)
  {
    m = (m * i + to_euclidean(randomize<From42>(N {1.0, 0.3}))) / (i + 1);
  }
  auto offset = eigen_matrix_t<double, 4, 2>::Constant(1);
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-6));

  for (int i=0; i<100; i++)
  {
    m = (m * i + to_euclidean(randomize<From42>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3}))) / (i + 1);
  }
  auto offset2 = to_euclidean(From42 {1., 1., 2., 2., 3., 3., 4., 4.});
  EXPECT_TRUE(is_near(m, offset2, 0.1));
  EXPECT_FALSE(is_near(m, offset2, 1e-6));

  for (int i=0; i<100; i++)
  {
    m = (m * i + to_euclidean(randomize<From42>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3},
      N {5.0, 0.3}, 6.0, N {7.0, 0.3}, N {8.0, 0.3}))) / (i + 1);
  }
  auto offset3 = to_euclidean(From42 {1., 2., 3., 4., 5., 6., 7., 8.});
  EXPECT_TRUE(is_near(m, offset3, 0.1));
  EXPECT_FALSE(is_near(m, offset3, 1e-6));
}


TEST(special_matrices, FromEuclideanExpr_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {1, 2, 3,
                                                                               std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                               0.5, std::sqrt(3)/2, std::sqrt(2)/2},
    FromEuclideanExpr<Cra, eigen_matrix_t<double, 3, 3>> {std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                                                                               std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                                                                               4, 5, 6}),
    make_eigen_matrix<double, 4, 3>(
      1., 2, 3,
      pi/6, pi/3, pi/4,
      pi/4, pi/3, pi/6,
      4, 5, 6)));
  EXPECT_TRUE(is_near(concatenate_horizontal(
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {1, 2, 3,
                                                                               std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                               0.5, std::sqrt(3)/2, std::sqrt(2)/2},
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {4, 5, 6,
                                                                               std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                                                                               std::sqrt(2)/2, std::sqrt(3)/2, 0.5}),
    make_eigen_matrix<double, 2, 6>(
      1, 2, 3, 4, 5, 6,
      pi/6, pi/3, pi/4, pi/4, pi/3, pi/6)));
  EXPECT_TRUE(is_near(split_vertical(FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 2>> {
      1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    FromEuclideanExpr<FixedDescriptor<Axis, angle::Radians, angle::Radians, Axis>, eigen_matrix_t<double, 6, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6}),
    std::tuple {make_eigen_matrix<double, 2, 3>(1., 2, 3, pi/6, pi/3, pi/4),
               make_eigen_matrix<double, 2, 3>(pi/4, pi/3, pi/6, 4, 5, 6)}));
  EXPECT_TRUE(is_near(split_vertical<2, 1>(
    FromEuclideanExpr<FixedDescriptor<Axis, angle::Radians, angle::Radians, Axis>, eigen_matrix_t<double, 6, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6}),
    std::tuple {make_eigen_matrix<double, 2, 3>(1., 2, 3, pi/6, pi/3, pi/4),
               make_eigen_matrix<double, 1, 3>(pi/4, pi/3, pi/6)}));
  EXPECT_TRUE(is_near(split_vertical<Car, FixedDescriptor<angle::Radians, Axis>>(
    FromEuclideanExpr<FixedDescriptor<Axis, angle::Radians, angle::Radians, Axis>, eigen_matrix_t<double, 6, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6}),
    std::tuple {make_eigen_matrix<double, 2, 3>(1., 2, 3, pi/6, pi/3, pi/4),
               make_eigen_matrix<double, 2, 3>(pi/4, pi/3, pi/6, 4, 5, 6)}));
  EXPECT_TRUE(is_near(
    split_vertical<Car, FixedDescriptor<angle::Radians, Axis>>(
      from_euclidean<FixedDescriptor<Axis, angle::Radians, angle::Radians, Axis>>(
        to_euclidean<FixedDescriptor<Axis, angle::Radians, angle::Radians, Axis>>(
          make_eigen_matrix<double, 4, 3>(1., 2, 3, pi/6, pi/3, pi/4, pi/4, pi/3, pi/6, 4, 5, 6)
      ))),
    std::tuple {
      make_eigen_matrix<double, 2, 3>(1., 2, 3, pi/6, pi/3, pi/4),
      make_eigen_matrix<double, 2, 3>(pi/4, pi/3, pi/6, 4, 5, 6)
      }));
  EXPECT_TRUE(is_near(split_vertical<Car, FixedDescriptor<angle::Radians>>(
    FromEuclideanExpr<FixedDescriptor<Axis, angle::Radians, angle::Radians, Axis>, eigen_matrix_t<double, 6, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6}),
    std::tuple {make_eigen_matrix<double, 2, 3>(1., 2, 3, pi/6, pi/3, pi/4),
               make_eigen_matrix<double, 1, 3>(pi/4, pi/3, pi/6)}));
  EXPECT_TRUE(is_near(split_horizontal(FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 2>> {
    1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<3, 3>(
    FromEuclideanExpr<Polar<>, const eigen_matrix_t<double, 3, 6>> {
      1, 2, 3, 4, 5, 6,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2, std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2, std::sqrt(2)/2, std::sqrt(3)/2, 0.5}),
    std::tuple {make_eigen_matrix<double, 2, 3>(1., 2, 3, pi/6, pi/3, pi/4),
               make_eigen_matrix<double, 2, 3>(4, 5, 6, pi/4, pi/3, pi/6)}));

  auto a1 = FromEuclideanExpr<Polar<>, const eigen_matrix_t<double, 3, 6>> {
    1, 2, 3, 4, 5, 6,
    std::sqrt(3)/2, 0.5, std::sqrt(2)/2, std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
    0.5, std::sqrt(3)/2, std::sqrt(2)/2, std::sqrt(2)/2, std::sqrt(3)/2, 0.5};

  EXPECT_TRUE(is_near(split_horizontal<3, 3>(a1),
    std::tuple {make_eigen_matrix<double, 2, 3>(1., 2, 3, pi/6, pi/3, pi/4),
               make_eigen_matrix<double, 2, 3>(4, 5, 6, pi/4, pi/3, pi/6)}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(
    FromEuclideanExpr<Polar<>, eigen_matrix_t<double, 3, 6>> {
      1, 2, 3, 4, 5, 6,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2, std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2, std::sqrt(2)/2, std::sqrt(3)/2, 0.5}),
    std::tuple {make_eigen_matrix<double, 2, 3>(1., 2, 3, pi/6, pi/3, pi/4),
               make_eigen_matrix<double, 2, 2>(4, 5, pi/4, pi/3)}));

  EXPECT_TRUE(is_near(split_diagonal<Axis, angle::Radians>(
    FromEuclideanExpr<Car, const eigen_matrix_t<double, 3, 2>> {
      1, 2,
      std::sqrt(3)/2, 0.5,
      0.5, std::sqrt(3)/2}),
    std::tuple {make_eigen_matrix<double, 1, 1>(1),
               make_eigen_matrix<double, 1, 1>(pi/6)}));
  EXPECT_TRUE(is_near(split_diagonal<1, 1>(
    FromEuclideanExpr<Polar<>, const eigen_matrix_t<double, 3, 2>> {
      1, 2,
      std::sqrt(3)/2, 0.5,
      0.5, std::sqrt(3)/2}),
    std::tuple {make_eigen_matrix<double, 1, 1>(1),
               make_eigen_matrix<double, 1, 1>(pi/6)}));

  EXPECT_TRUE(is_near(column(
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2}, 2),
    make_eigen_matrix<double, 2, 1>(3, pi/4)));
  EXPECT_TRUE(is_near(column<1>(
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2}),
    make_eigen_matrix<double, 2, 1>(2, pi/3)));

  EXPECT_TRUE(is_near(row(
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2}, 1),
    make_eigen_matrix<double, 1, 3>(pi/6, pi/3, pi/4)));
  EXPECT_TRUE(is_near(row<0>(
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2}),
    make_eigen_matrix<double, 1, 3>(1, 2, 3)));

  auto b = FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {
    1, 2, 3,
    std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
    0.5, std::sqrt(3)/2, std::sqrt(2)/2};

  EXPECT_TRUE(is_near(b, make_eigen_matrix<double, 2, 3>(1, 2, 3, pi/6, pi/3, pi/4)));
  EXPECT_TRUE(is_near(apply_columnwise([](auto& col){ col *= 3; }),
    make_eigen_matrix<double, 2, 3>(
      3, 6, 9,
      pi/2, pi, pi*3/4), b));

  b = FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {
    1, 2, 3,
    std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
    0.5, std::sqrt(3)/2, std::sqrt(2)/2};

  EXPECT_TRUE(is_near(apply_columnwise([](auto& col, std::size_t i){ col *= i + 1; }),
    make_eigen_matrix<double, 2, 3>(
      1, 4, 9,
      pi/6, pi*2/3, pi*3/4), b));

  auto f2 = [](const auto& col){ return make_self_contained(col + col); };

  EXPECT_TRUE(is_near(apply_columnwise(f2,
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2}),
    make_eigen_matrix<double, 2, 3>(
      2, 4, 6,
      pi/3, pi*2/3, pi/2)));
  EXPECT_TRUE(is_near(apply_columnwise(f2,
    FromEuclideanExpr<Car, ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>>>
      {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 2, 3>(2., 4, 6, pi/3, pi*2/3, pi/2)));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col * i); },
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {1, 2, 3,
                                                          std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                          0.5, std::sqrt(3)/2, std::sqrt(2)/2}),
    make_eigen_matrix<double, 2, 3>(
      0, 2, 6,
      0, pi/3, pi/2)));
  EXPECT_TRUE(is_near(apply_columnwise<3>(
    [](){ return FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 1>> {1., std::sqrt(3)/2, 0.5}; }),
    make_eigen_matrix<double, 2, 3>(
      1, 1, 1,
      pi/6, pi/6, pi/6)));
  EXPECT_TRUE(is_near(apply_columnwise<3>(
    [](std::size_t i){ return make_self_contained(FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 1>> {1., std::sqrt(3)/2, 0.5} * (i + 1)); }),
    make_eigen_matrix<double, 2, 3>(
      1, 2, 3,
      pi/6, pi/3, pi/2)));


  EXPECT_TRUE(is_near(apply_rowwise(f2,
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2}),
    make_eigen_matrix<double, 2, 3>(
      2, 4, 6,
      pi/3, pi*2/3, pi/2)));
  EXPECT_TRUE(is_near(apply_rowwise(f2,
    FromEuclideanExpr<Car, ToEuclideanExpr<Car, eigen_matrix_t<double, 2, 3>>>
      {1., 2, 3, pi/6, pi/3, pi/4}),
    make_eigen_matrix<double, 2, 3>(2., 4, 6, pi/3, pi*2/3, pi/2)));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row * (i + 1)); },
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {1, 2, 3,
                                                          std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                          0.5, std::sqrt(3)/2, std::sqrt(2)/2}),
    make_eigen_matrix<double, 2, 3>(
      1, 2, 3,
      pi/3, pi*2/3, pi/2)));
  EXPECT_TRUE(is_near(apply_rowwise<3>(
    [](){ return FromEuclideanExpr<Axis, eigen_matrix_t<double, 1, 3>> {1, 2, 3}; }),
    make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      1, 2, 3,
      1, 2, 3)));
  EXPECT_TRUE(is_near(apply_rowwise<3>(
    [](std::size_t i){ return make_self_contained(FromEuclideanExpr<Axis, eigen_matrix_t<double, 1, 3>> {1, 2, 3} * (i + 1)); }),
    make_eigen_matrix<double, 3, 3>(
      1, 2, 3,
      2, 4, 6,
      3, 6, 9)));

  //
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x * 3; },
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {1, 2, 3,
                                                          std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                          0.5, std::sqrt(3)/2, std::sqrt(2)/2}),
    make_eigen_matrix<double, 2, 3>(
      3, 6, 9,
      pi/2, pi, pi*3/4)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x * (j + 1); },
    FromEuclideanExpr<Car, eigen_matrix_t<double, 3, 3>> {1, 2, 3,
                                                          std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                          0.5, std::sqrt(3)/2, std::sqrt(2)/2}),
    make_eigen_matrix<double, 2, 3>(
      1, 4, 9,
      pi/6, pi*2/3, pi*3/4)));
}


TEST(special_matrices, FromEuclideanExpr_arithmetic)
{
  EXPECT_TRUE(is_near(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} + From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}, mat3(2, 4, pi/3, pi*2/3, 6, 8)));
  EXPECT_TRUE(is_near(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} - From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}, M32::Zero()));
  EXPECT_TRUE(is_near(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} * 2, mat3(2, 4, pi/3, pi*2/3, 6, 8)));
  EXPECT_TRUE(is_near(2 * From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}, mat3(2, 4, pi/3, pi*2/3, 6, 8)));
  EXPECT_TRUE(is_near(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} / 2, mat3(0.5, 1, pi/12, pi/6, 1.5, 2)));
  EXPECT_TRUE(is_near(-From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}, mat3(-1, -2, -pi/6, -pi/3, -3, -4)));
  EXPECT_TRUE(is_near(From42 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} * DiagonalMatrix {1., 2}, mat3(1, 4, pi/6, pi*2/3, 3, 8)));
  using To3 = ToEuclideanExpr<Cara, M32>;
  using FromTo32 = FromEuclideanExpr<Cara, To3>;
  EXPECT_TRUE(is_near(FromTo32(To3 {1, 2, pi/6 + 2*pi, pi/3 - 6*pi, 3, 4}) + FromTo32(To3 {1, 2, pi/6, pi/3, 3, 4}), mat3(2, 4, pi/3, pi*2/3, 6, 8)));
  EXPECT_TRUE(is_near(FromTo32(To3 {2, 4, pi/3, pi*2/3, 6, 8}) - FromTo32(To3 {1, 2, pi/6 + 2*pi, pi/3 - 6*pi, 3, 4}), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(-FromTo32(To3 {1, 2, pi/6 + 2*pi, pi/3 - 6*pi, 3, 4}), mat3(-1, -2, -pi/6, -pi/3, -3, -4)));
}


TEST(special_matrices, FromEuclideanExpr_references)
{
  M22 m, n;
  m << pi/6, pi/4, 1, 2;
  n << pi/4, pi/3, 3, 4;
  M32 me, ne;
  me << std::sqrt(3)/2, std::sqrt(2)/2, 0.5, std::sqrt(2)/2, 1, 2;
  ne << std::sqrt(2)/2, 0.5, std::sqrt(2)/2, std::sqrt(3)/2, 3, 4;
  using From = FromEuclideanExpr<Cra, M32>;
  From x = From {me};
  FromEuclideanExpr<Cra, M32&> x_lvalue = x;
  EXPECT_TRUE(is_near(x_lvalue, m));
  x = From {ne};
  EXPECT_TRUE(is_near(x_lvalue, n));
  x_lvalue = From {me};
  EXPECT_TRUE(is_near(x, m));
}


TEST(special_matrices, Wrap_angle)
{
  using R = FromEuclideanExpr<angle::Radians, ToEuclideanExpr<angle::Radians, M11>>;
  R x0 {pi/4};
  EXPECT_NEAR(get_element(x0, 0, 0), pi/4, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), pi/4, 1e-6);
  set_element(x0, 5*pi/4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0), -3*pi/4, 1e-6);
  set_element(x0, -7*pi/6, 0);
  EXPECT_NEAR(get_element(x0, 0), 5*pi/6, 1e-6);
}


TEST(special_matrices, Wrap_distance)
{
  using R = FromEuclideanExpr<Distance, ToEuclideanExpr<Distance, M11>>;
  R x0 {-5};
  EXPECT_TRUE(is_near(x0 + R {1.2}, eigen_matrix_t<double, 1, 1> {6.2}));
  EXPECT_TRUE(is_near(R {R {1.1} - 3. * R {1}}, R {1.9}));
  EXPECT_TRUE(is_near(R {1.2} + R {-3}, R {4.2}));
  EXPECT_NEAR(get_element(x0, 0, 0), 5, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), 5., 1e-6);
  set_element(x0, 4, 0);
  EXPECT_NEAR(get_element(x0, 0), 4., 1e-6);
  set_element(x0, -3, 0);
  EXPECT_NEAR(get_element(x0, 0), 3., 1e-6);
}


TEST(special_matrices, Wrap_inclination)
{
  using R = FromEuclideanExpr<inclination::Radians, ToEuclideanExpr<inclination::Radians, M11>>;
  R x0 {pi/2};
  EXPECT_NEAR(get_element(x0, 0, 0), pi/2, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), pi/2, 1e-6);
  set_element(x0, pi/4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0, 0), pi/4, 1e-6);
  set_element(x0, 3*pi/4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0, 0), pi/4, 1e-6);
}


TEST(special_matrices, Wrap_polar)
{
  using C1 = Polar<Distance, angle::Radians>;
  using P = FromEuclideanExpr<C1, ToEuclideanExpr<C1, eigen_matrix_t<double, 2, 1>>>;
  P x0 {2, pi/4};
  EXPECT_NEAR(get_element(x0, 0, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), pi/4, 1e-6);
  set_element(x0, -1.5, 0);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -3*pi/4, 1e-6);
  set_element(x0, 7*pi/6, 1);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -5*pi/6, 1e-6);

  using C2 = Polar<angle::Radians, Distance>;
  using Q = FromEuclideanExpr<C2, ToEuclideanExpr<C2, eigen_matrix_t<double, 2, 1>>>;
  Q x1 {pi/4, 2};
  EXPECT_NEAR(get_element(x1, 1, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), pi/4, 1e-6);
  set_element(x1, -1.5, 1);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -3*pi/4, 1e-6);
  set_element(x1, 7*pi/6, 0);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -5*pi/6, 1e-6);
}


TEST(special_matrices, Wrap_spherical)
{
  using C1 = Spherical<Distance, angle::Radians, inclination::Radians>;
  using S = FromEuclideanExpr<C1, ToEuclideanExpr<C1, eigen_matrix_t<double, 3, 1>>>;
  S x0 {2, pi/4, -pi/4};
  EXPECT_NEAR(get_element(x0, 0, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), pi/4, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), -pi/4, 1e-6);
  set_element(x0, -1.5, 0);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -3*pi/4, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), pi/4, 1e-6);
  set_element(x0, 7*pi/6, 1);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -5*pi/6, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), pi/4, 1e-6);
  set_element(x0, 3*pi/4, 2);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), pi/6, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), pi/4, 1e-6);

  using C2 = Spherical<angle::Radians, Distance, inclination::Radians>;
  using T = FromEuclideanExpr<C2, ToEuclideanExpr<C2, eigen_matrix_t<double, 3, 1>>>;
  T x1 {pi/4, 2, -pi/4};
  EXPECT_NEAR(get_element(x1, 1, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), pi/4, 1e-6);
  EXPECT_NEAR(get_element(x1, 2), -pi/4, 1e-6);
  set_element(x1, -1.5, 1);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -3*pi/4, 1e-6);
  EXPECT_NEAR(get_element(x1, 2), pi/4, 1e-6);
  set_element(x1, 7*pi/6, 0);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -5*pi/6, 1e-6);
  EXPECT_NEAR(get_element(x1, 2), pi/4, 1e-6);
  set_element(x1, 3*pi/4, 2);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), pi/6, 1e-6);
  EXPECT_NEAR(get_element(x1, 2), pi/4, 1e-6);

  using C3 = Spherical<angle::Radians, inclination::Radians, Distance>;
  using U = FromEuclideanExpr<C3, ToEuclideanExpr<C3, eigen_matrix_t<double, 3, 1>>>;
  U x2 {pi/4, -pi/4, 2};
  EXPECT_NEAR(get_element(x2, 2, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x2, 0), pi/4, 1e-6);
  EXPECT_NEAR(get_element(x2, 1), -pi/4, 1e-6);
  set_element(x2, -1.5, 2);
  EXPECT_NEAR(get_element(x2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x2, 0), -3*pi/4, 1e-6);
  EXPECT_NEAR(get_element(x2, 1), pi/4, 1e-6);
  set_element(x2, 7*pi/6, 0);
  EXPECT_NEAR(get_element(x2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x2, 0), -5*pi/6, 1e-6);
  EXPECT_NEAR(get_element(x2, 1), pi/4, 1e-6);
  set_element(x2, 3*pi/4, 1);
  EXPECT_NEAR(get_element(x2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x2, 0), pi/6, 1e-6);
  EXPECT_NEAR(get_element(x2, 1), pi/4, 1e-6);
}
