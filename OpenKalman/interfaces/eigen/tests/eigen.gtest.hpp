/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Tests for the Eigen3 interface.
 *
 * \file
 * \brief Header file for Eigen3 tests.
 */

#ifndef EIGEN3_GTEST_HPP
#define EIGEN3_GTEST_HPP

#include "interfaces/eigen/eigen.hpp"
#include "basics/tests/tests.hpp"
#include <complex>

namespace OpenKalman::test
{

#ifdef __cpp_concepts
  template<Eigen3::eigen_dense_general Arg1, Eigen3::eigen_dense_general Arg2, typename Err> requires
    std::is_arithmetic_v<Err> or Eigen3::eigen_dense_general<Err>
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<Eigen3::eigen_dense_general<Arg1> and Eigen3::eigen_dense_general<Arg2> and
    (std::is_arithmetic_v<Err> or Eigen3::eigen_dense_general<Err>)>>
#endif
    : ::testing::AssertionResult
  {

  private:

    static ::testing::AssertionResult
    compare(const Arg1& arg1, const Arg2& arg2, const Err& err)
    {
      if constexpr (std::is_arithmetic_v<Err>)
      {
        if (arg1.matrix().isApprox(arg2.matrix(), err) or (arg1.matrix().isMuchSmallerThan(1., err) and
          arg2.matrix().isMuchSmallerThan(1., err)))
        {
          return ::testing::AssertionSuccess();
        }
      }
      else
      {
        if (((arg1.array() - arg2.array()).abs() - err).maxCoeff() <= 0)
        {
          return ::testing::AssertionSuccess();
        }
      }

      return ::testing::AssertionFailure() << std::endl << arg1 << std::endl << "is not near" << std::endl <<
        arg2 << std::endl;
    }

  public:

    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {compare(arg1, arg2, err)} {};

   };


#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2, typename Err> requires
    (Eigen3::eigen_general<Arg1> and not Eigen3::eigen_general<Arg2>) or
    (not Eigen3::eigen_general<Arg1> and Eigen3::eigen_general<Arg2>)
  struct TestComparison<Arg1, Arg2, Err>
#else
  template<typename Arg1, typename Arg2, typename Err>
  struct TestComparison<Arg1, Arg2, Err, std::enable_if_t<indexible<Arg1> and indexible<Arg2> and
    ((Eigen3::eigen_general<Arg1> and not Eigen3::eigen_general<Arg2>) or
    (not Eigen3::eigen_general<Arg1> and Eigen3::eigen_general<Arg2>))>>
#endif
    : ::testing::AssertionResult
  {
  private:

    using A = std::conditional_t<Eigen3::eigen_general<Arg1>, Arg1, Arg2>;

  public:
    TestComparison(const Arg1& arg1, const Arg2& arg2, const Err& err)
      : ::testing::AssertionResult {is_near(to_native_matrix<A>(arg1), to_native_matrix<A>(arg2), err)} {};

  };

  using Mxx = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using Mx0 = eigen_matrix_t<double, dynamic_size, 0>;
  using Mx1 = eigen_matrix_t<double, dynamic_size, 1>;
  using Mx2 = eigen_matrix_t<double, dynamic_size, 2>;
  using Mx3 = eigen_matrix_t<double, dynamic_size, 3>;
  using Mx4 = eigen_matrix_t<double, dynamic_size, 4>;
  using Mx5 = eigen_matrix_t<double, dynamic_size, 5>;

  using M0x = eigen_matrix_t<double, 0, dynamic_size>;
  using M00 = eigen_matrix_t<double, 0, 0>;
  using M01 = eigen_matrix_t<double, 0, 1>;
  using M02 = eigen_matrix_t<double, 0, 2>;
  using M03 = eigen_matrix_t<double, 0, 3>;
  using M04 = eigen_matrix_t<double, 0, 4>;

  using M1x = eigen_matrix_t<double, 1, dynamic_size>;
  using M10 = eigen_matrix_t<double, 1, 0>;
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M12 = eigen_matrix_t<double, 1, 2>;
  using M13 = eigen_matrix_t<double, 1, 3>;
  using M14 = eigen_matrix_t<double, 1, 4>;

  using M2x = eigen_matrix_t<double, 2, dynamic_size>;
  using M20 = eigen_matrix_t<double, 2, 0>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M24 = eigen_matrix_t<double, 2, 4>;

  using M3x = eigen_matrix_t<double, 3, dynamic_size>;
  using M30 = eigen_matrix_t<double, 3, 0>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;

  using M4x = eigen_matrix_t<double, 4, dynamic_size>;
  using M40 = eigen_matrix_t<double, 4, 0>;
  using M41 = eigen_matrix_t<double, 4, 1>;
  using M42 = eigen_matrix_t<double, 4, 2>;
  using M43 = eigen_matrix_t<double, 4, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M45 = eigen_matrix_t<double, 4, 5>;

  using M5x = eigen_matrix_t<double, 5, dynamic_size>;
  using M50 = eigen_matrix_t<double, 5, 0>;
  using M51 = eigen_matrix_t<double, 5, 1>;
  using M52 = eigen_matrix_t<double, 5, 2>;
  using M53 = eigen_matrix_t<double, 5, 3>;
  using M54 = eigen_matrix_t<double, 5, 4>;
  using M55 = eigen_matrix_t<double, 5, 5>;

  using cdouble = std::complex<double>;

  using CM11 = eigen_matrix_t<cdouble, 1, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM33 = eigen_matrix_t<cdouble, 3, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;

  using CM2x = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using CMx2 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using CMxx = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;

  using A11 = Eigen::Array<double, 1, 1>; static_assert(one_by_one_matrix<A11>);
  using A1x = Eigen::Array<double, 1, Eigen::Dynamic>;
  using Ax1 = Eigen::Array<double, Eigen::Dynamic, 1>;
  using Axx = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

  using A22 = Eigen::Array<double, 2, 2>;
  using A2x = Eigen::Array<double, 2, Eigen::Dynamic>;
  using Ax2 = Eigen::Array<double, Eigen::Dynamic, 2>;
  using Axx = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

  using CA22 = Eigen::Array<cdouble, 2, 2>;

  using I11 = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, A11>; static_assert(constant_coefficient_v<I11> == 1);
  static_assert(one_by_one_matrix<I11, Likelihood::maybe>);
  using I1x = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, A1x>; static_assert(constant_coefficient_v<I1x> == 1);
  using Ix1 = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, Ax1>; static_assert(constant_coefficient_v<Ix1> == 1);
  using Ixx = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, Axx>; static_assert(constant_coefficient_v<Ixx> == 1);
  using I22 = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, A22>;
  using I2x = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, A2x>;

  using Z11 = decltype(std::declval<I11>() - std::declval<I11>());
  using Z22 = decltype(std::declval<I22>() - std::declval<I22>());
  using Z21 = Eigen::Replicate<Z11, 2, 1>;
  using Z23 = Eigen::Replicate<Z11, 2, 3>;
  using Z12 = Eigen::Replicate<Z11, 1, 2>;
  using Z2x = Eigen::Replicate<Z11, 2, Eigen::Dynamic>;
  using Zx2 = Eigen::Replicate<Z11, Eigen::Dynamic, 2>;
  using Zxx = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic>;
  using Z1x = Eigen::Replicate<Z11, 1, Eigen::Dynamic>;
  using Zx1 = Eigen::Replicate<Z11, Eigen::Dynamic, 1>;

  using C11_1 = I11;
  using C22_1 = Eigen::Replicate<C11_1, 2, 2>;
  using C21_1 = Eigen::Replicate<C11_1, 2, 1>;
  using C12_1 = Eigen::Replicate<C11_1, 1, 2>;
  using C2x_1 = Eigen::Replicate<C11_1, 2, Eigen::Dynamic>;
  using C1x_1 = Eigen::Replicate<C11_1, 1, Eigen::Dynamic>;
  using Cx1_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 1>;
  using Cx2_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 2>;
  using Cxx_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_2 = decltype(std::declval<I11>() + std::declval<I11>());
  using C22_2 = Eigen::Replicate<C11_2, 2, 2>;
  using C21_2 = Eigen::Replicate<C11_2, 2, 1>;
  using C12_2 = Eigen::Replicate<C11_2, 1, 2>;
  using C2x_2 = Eigen::Replicate<C11_2, 2, Eigen::Dynamic>;
  using Cx2_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 2>;
  using C1x_2 = Eigen::Replicate<C11_2, 1, Eigen::Dynamic>;
  using Cx1_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 1>;
  using Cxx_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_3 = decltype(std::declval<I11>() + std::declval<I11>() + std::declval<I11>());
  using C21_3 = Eigen::Replicate<C11_3, 2, 1>;
  using C2x_3 = Eigen::Replicate<C11_3, 2, Eigen::Dynamic>;
  using Cx1_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, 1>;
  using Cxx_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_m1 = decltype(-std::declval<I11>());
  using C21_m1 = Eigen::Replicate<C11_m1, 2, 1>;
  using C2x_m1 = Eigen::Replicate<C11_m1, 2, Eigen::Dynamic>;
  using Cx1_m1 = Eigen::Replicate<C11_m1, Eigen::Dynamic, 1>;
  using Cxx_m1 = Eigen::Replicate<C11_m1, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_m2 = decltype(-(std::declval<I11>() + std::declval<I11>()));
  using C22_m2 = Eigen::Replicate<C11_m2, 2, 2>;
  using C21_m2 = Eigen::Replicate<C11_m2, 2, 1>;
  using C2x_m2 = Eigen::Replicate<C11_m2, 2, Eigen::Dynamic>;
  using Cx1_m2 = Eigen::Replicate<C11_m2, Eigen::Dynamic, 1>;
  using Cxx_m2 = Eigen::Replicate<C11_m2, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_1cx = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<cdouble>, Eigen::Array<cdouble, 1, 1>>;
  using C11_2cx = decltype(std::declval<C11_1cx>() + std::declval<C11_1cx>());

  using B11 = Eigen::Array<bool, 1, 1>;
  using B22 = Eigen::Array<bool, 2, 2>;

  using B11_true = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<bool>, B11>;
  using B11_false = decltype(not std::declval<B11_true>());
  using B22_true = decltype(std::declval<B11_true>().replicate<2,2>());
  using B22_false = decltype(std::declval<B11_false>().replicate<2,2>());
  using BI22 = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<bool>, B22>;

  using Cd22_2 = decltype(std::declval<I22>() + std::declval<I22>());
  using Cd2x_2 = Eigen::Replicate<Cd22_2, 1, Eigen::Dynamic>;
  using Cdx2_2 = Eigen::Replicate<Cd22_2, Eigen::Dynamic, 1>;
  using Cdxx_2 = Eigen::Replicate<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>;

  using Cd22_3 = decltype(std::declval<I22>() + std::declval<I22>() + std::declval<I22>());
  using Cd2x_3 = Eigen::Replicate<Cd22_3, 1, Eigen::Dynamic>;
  using Cdx2_3 = Eigen::Replicate<Cd22_3, Eigen::Dynamic, 1>;
  using Cdxx_3 = Eigen::Replicate<Cd22_3, Eigen::Dynamic, Eigen::Dynamic>;

  using Cd22_m1 = decltype(-std::declval<I22>());
  using Cd2x_m1 = Eigen::Replicate<Cd22_m1, 1, Eigen::Dynamic>;
  using Cdx2_m1 = Eigen::Replicate<Cd22_m1, Eigen::Dynamic, 1>;
  using Cdxx_m1 = Eigen::Replicate<Cd22_m1, Eigen::Dynamic, Eigen::Dynamic>;

  using Cd22_m2 = decltype(-std::declval<Cd22_2>());
  using Cd2x_m2 = Eigen::Replicate<Cd22_m2, 1, Eigen::Dynamic>;
  using Cdx2_m2 = Eigen::Replicate<Cd22_m2, Eigen::Dynamic, 1>;
  using Cdxx_m2 = Eigen::Replicate<Cd22_m2, Eigen::Dynamic, Eigen::Dynamic>;

  using DM2 = Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 2>>;
  using DMx = Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>;

  using DW21 = Eigen3::EigenWrapper<Eigen::DiagonalWrapper<M21>>;
  using DW2x = Eigen3::EigenWrapper<Eigen::DiagonalWrapper<M2x>>;
  using DWx1 = Eigen3::EigenWrapper<Eigen::DiagonalWrapper<Mx1>>;
  using DWxx = Eigen3::EigenWrapper<Eigen::DiagonalWrapper<Mxx>>;

  using Salv22 = Eigen3::EigenWrapper<Eigen::SelfAdjointView<M22, Eigen::Lower>>;
  using Salv2x = Eigen3::EigenWrapper<Eigen::SelfAdjointView<M2x, Eigen::Lower>>;
  using Salvx2 = Eigen3::EigenWrapper<Eigen::SelfAdjointView<Mx2, Eigen::Lower>>;
  using Salvxx = Eigen3::EigenWrapper<Eigen::SelfAdjointView<Mxx, Eigen::Lower>>;

  using Sauv22 = Eigen3::EigenWrapper<Eigen::SelfAdjointView<M22, Eigen::Upper>>;
  using Sauv2x = Eigen3::EigenWrapper<Eigen::SelfAdjointView<M2x, Eigen::Upper>>;
  using Sauvx2 = Eigen3::EigenWrapper<Eigen::SelfAdjointView<Mx2, Eigen::Upper>>;
  using Sauvxx = Eigen3::EigenWrapper<Eigen::SelfAdjointView<Mxx, Eigen::Upper>>;

  using Sadv22 = Eigen3::EigenWrapper<Eigen::SelfAdjointView<M22::IdentityReturnType, Eigen::Lower>>;
  using Sadv2x = Eigen3::EigenWrapper<Eigen::SelfAdjointView<M2x::IdentityReturnType, Eigen::Lower>>;
  using Sadvx2 = Eigen3::EigenWrapper<Eigen::SelfAdjointView<Mx2::IdentityReturnType, Eigen::Lower>>;
  using Sadvxx = Eigen3::EigenWrapper<Eigen::SelfAdjointView<Mxx::IdentityReturnType, Eigen::Lower>>;

  using Tlv22 = Eigen3::EigenWrapper<Eigen::TriangularView<M22, Eigen::Lower>>;
  using Tlv2x = Eigen3::EigenWrapper<Eigen::TriangularView<M2x, Eigen::Lower>>;
  using Tlvx2 = Eigen3::EigenWrapper<Eigen::TriangularView<Mx2, Eigen::Lower>>;
  using Tlvxx = Eigen3::EigenWrapper<Eigen::TriangularView<Mxx, Eigen::Lower>>;

  using Tuv22 = Eigen3::EigenWrapper<Eigen::TriangularView<M22, Eigen::Upper>>;
  using Tuv2x = Eigen3::EigenWrapper<Eigen::TriangularView<M2x, Eigen::Upper>>;
  using Tuvx2 = Eigen3::EigenWrapper<Eigen::TriangularView<Mx2, Eigen::Upper>>;
  using Tuvxx = Eigen3::EigenWrapper<Eigen::TriangularView<Mxx, Eigen::Upper>>;

} // namespace OpenKalman::test

#endif //EIGEN3_GTEST_HPP
