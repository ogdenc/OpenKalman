/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to native Eigen3 types.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_HPP
#define OPENKALMAN_EIGEN3_TRAITS_HPP

#include <type_traits>


namespace OpenKalman
{
  using namespace OpenKalman::internal;
  using namespace OpenKalman::Eigen3;
  using namespace OpenKalman::Eigen3::internal;


  namespace internal::detail
  {
    // T is self-contained and Eigen stores it by value rather than by reference.
    template<typename T>
#ifdef __cpp_concepts
    concept stores =
#else
    static constexpr bool stores =
#endif
      self_contained<T> and ((Eigen::internal::traits<T>::Flags & Eigen::NestByRefBit) == 0);
  }


  // ------- //
  //  Array  //
  // ------- //

  template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  struct is_native_eigen_matrix<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : std::false_type {};


  template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  struct is_self_contained<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : std::true_type {};


  // -------------- //
  //  ArrayWrapper  //
  // -------------- //

  template<typename XprType>
  struct is_native_eigen_matrix<Eigen::ArrayWrapper<XprType>>
    : std::false_type {};


  template<typename XprType>
  struct is_self_contained<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<detail::stores<XprType>> {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix XprType>
  struct constant_diagonal_coefficient<Eigen::ArrayWrapper<XprType>>
#else
  template<typename XprType>
  struct constant_diagonal_coefficient<Eigen::ArrayWrapper<XprType>, std::enable_if_t<
    constant_diagonal_matrix<XprType>>>
#endif
    : constant_diagonal_coefficient<std::decay_t<XprType>> {};


#ifdef __cpp_concepts
  template<constant_matrix XprType>
  struct constant_coefficient<Eigen::ArrayWrapper<XprType>>
#else
  template<typename XprType>
  struct constant_coefficient<Eigen::ArrayWrapper<XprType>, std::enable_if_t<constant_matrix<XprType>>>
#endif
    : constant_coefficient<std::decay_t<XprType>> {};


  template<typename XprType>
  struct is_diagonal_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<diagonal_matrix<XprType>> {};


  template<typename XprType>
  struct is_lower_self_adjoint_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<lower_self_adjoint_matrix<XprType>> {};


  template<typename XprType>
  struct is_upper_self_adjoint_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<upper_self_adjoint_matrix<XprType>> {};


  template<typename XprType>
  struct is_lower_triangular_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<lower_triangular_matrix<XprType>> {};


  template<typename XprType>
  struct is_upper_triangular_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<upper_triangular_matrix<XprType>> {};


  // ------- //
  //  Block  //
  // ------- //

  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct is_native_eigen_matrix<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    : std::bool_constant<native_eigen_matrix<XprType>> {};


  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct is_self_contained<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    : std::bool_constant<detail::stores<XprType>> {};


  /// A block taken from a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix XprType, int BlockRows, int BlockCols, bool InnerPanel> requires constant_matrix<XprType>
  struct constant_coefficient<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
#else
  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct constant_coefficient<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>,
    std::enable_if_t<constant_matrix<XprType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<XprType>> {};


  // --------------- //
  //  CwiseBinaryOp  //
  // --------------- //

  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct is_native_eigen_matrix<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    : std::bool_constant<native_eigen_matrix<LhsType> and native_eigen_matrix<RhsType>> {};


  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct is_self_contained<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    : std::bool_constant<detail::stores<LhsType> and detail::stores<RhsType>> {};


  // --- constant_diagonal_coefficient --- //

  /// The sum of two constant-diagonal matrices is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> + constant_diagonal_coefficient_v<Arg2>,
      typename Eigen::internal::scalar_sum_op<Scalar1, Scalar2>::result_type> {};


  /// The difference between two constant-diagonal matrices is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> - constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_difference_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise product of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_product_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise conjugate product of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise min of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> <= constant_diagonal_coefficient_v<Arg2> ?
        constant_diagonal_coefficient_v<Arg1> : constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_min_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise max of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> >= constant_diagonal_coefficient_v<Arg2> ?
        constant_diagonal_coefficient_v<Arg1> : constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_max_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise hypotenuse of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<Scalar, Scalar>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<Scalar, Scalar>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<
#if __cpp_nontype_template_args >= 201911L
      OpenKalman::internal::constexpr_sqrt(
        constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg1> +
        constant_diagonal_coefficient_v<Arg2> * constant_diagonal_coefficient_v<Arg2>)> {};
#else
      OpenKalman::internal::constexpr_sqrt(static_cast<short>(
        constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg1> +
        constant_diagonal_coefficient_v<Arg2> * constant_diagonal_coefficient_v<Arg2>))> {};
#endif


  // --- constant_coefficient --- //

  /// The sum of two constant matrices is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> + constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_sum_op<Scalar1, Scalar2>::result_type> {};


  /// The difference between two constant matrices is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> - constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_difference_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise product of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_product_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise conjugate product of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise min of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> <= constant_coefficient_v<Arg2> ?
        constant_coefficient_v<Arg1> : constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_min_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise max of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> >= constant_coefficient_v<Arg2> ?
        constant_coefficient_v<Arg1> : constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_max_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise hypotenuse of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<Scalar, Scalar>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<Scalar, Scalar>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<
#if __cpp_nontype_template_args >= 201911L
        OpenKalman::internal::constexpr_sqrt(constant_coefficient_v<Arg1> * constant_coefficient_v<Arg1> +
          constant_coefficient_v<Arg2> * constant_coefficient_v<Arg2>)> {};
#else
        OpenKalman::internal::constexpr_sqrt(
          static_cast<short>(constant_coefficient_v<Arg1> * constant_coefficient_v<Arg1> +
          constant_coefficient_v<Arg2> * constant_coefficient_v<Arg2>))> {};
#endif


  /// The coefficient-wise power of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_pow_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_pow_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<
        OpenKalman::internal::constexpr_pow(constant_coefficient_v<Arg1>, constant_coefficient_v<Arg2>),
        typename Eigen::internal::scalar_pow_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise quotient of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> / constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise AND of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<bool {constant_coefficient_v<Arg1>} and bool {constant_coefficient_v<Arg2>}> {};


  /// The coefficient-wise OR of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_or_op, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_or_op, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<bool {constant_coefficient_v<Arg1>} or bool {constant_coefficient_v<Arg2>}> {};


  /// The coefficient-wise XOR of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_xor_op, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_xor_op, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<bool {constant_coefficient_v<Arg1>} xor bool {constant_coefficient_v<Arg2>}> {};


  // --- is_diagonal_matrix --- //

  /// A diagonal matrix times a scalar (or vice versa) is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> or diagonal_matrix<Arg2>> {};


  /// A diagonal matrix divided by a scalar is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1>> {};


  /// The sum of two diagonal matrices is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>> {};


  /// The difference between two diagonal matrices is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>> {};


  // --- is_lower_self_adjoint_matrix and is_upper_self_adjoint_matrix --- //

  /// The sum of two self-adjoint matrices is lower-self-adjoint if at least one of the matrices is lower-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_lower_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<self_adjoint_matrix<Arg1> and self_adjoint_matrix<Arg2> and
      (lower_self_adjoint_matrix<Arg1> or lower_self_adjoint_matrix<Arg2>)> {};


  /// The sum of two upper-self-adjoint matrices is upper-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_upper_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>> {};


  /// The difference between two self-adjoint matrices is lower-self-adjoint if at least one is lower-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_lower_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<self_adjoint_matrix<Arg1> and self_adjoint_matrix<Arg2> and
      (lower_self_adjoint_matrix<Arg1> or lower_self_adjoint_matrix<Arg2>)> {};


  /// The difference between two upper-self-adjoint matrices is upper-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_upper_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>> {};


  // ---------------- //
  //  CwiseNullaryOp  //
  // ---------------- //

  template<typename UnaryOp, typename PlainObjectType>
  struct is_native_eigen_matrix<Eigen::CwiseNullaryOp<UnaryOp, PlainObjectType>>
    : std::true_type {};


  template<typename UnaryOp, typename PlainObjectType>
  struct is_self_contained<Eigen::CwiseNullaryOp<UnaryOp, PlainObjectType>>
    : std::true_type {};


#ifdef __cpp_concepts
  template<typename Scalar, square_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>,
      std::enable_if_t<square_matrix<Arg>>>
#endif
    : constant_coefficient_type<short {1}, Scalar> {};


  /// A constant square matrix is lower-self-adjoint if it is not complex.
  template<typename Scalar, typename PlainObjectType>
  struct is_lower_self_adjoint_matrix<
    Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>>
    : std::bool_constant<square_matrix<PlainObjectType> and not complex_number<Scalar>> {};


  /// A constant square matrix is upper-self-adjoint if it is not complex.
  template<typename Scalar, typename PlainObjectType>
  struct is_upper_self_adjoint_matrix<
    Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>>
    : std::bool_constant<square_matrix<PlainObjectType> and not complex_number<Scalar>> {};


  // ---------------- //
  //  CwiseTernaryOp  //
  // ---------------- //

  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct is_native_eigen_matrix<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
    : std::true_type {};


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct is_self_contained<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
    : std::bool_constant<detail::stores<Arg1> and detail::stores<Arg2> and detail::stores<Arg3>> {};


  // -------------- //
  //  CwiseUnaryOp  //
  // -------------- //

  template<typename UnaryOp, typename XprType>
  struct is_native_eigen_matrix<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
    : std::true_type {};


  template<typename UnaryOp, typename XprType>
  struct is_self_contained<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
    : std::bool_constant<detail::stores<XprType>> {};


  /// The negation of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type< - constant_diagonal_coefficient_v<Arg>> {};


  /// The conjugate of a constant-diagonal matrix is constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<
#if __cpp_nontype_template_args >= 201911L
      complex_number<Scalar> ?
#ifdef __cpp_lib_constexpr_complex
      std::conj(constant_diagonal_coefficient_v<Arg>) :
#else
      std::complex(std::real(constant_diagonal_coefficient_v<Arg>), -std::imag(constant_diagonal_coefficient_v<Arg>)) :
#endif
#endif
      constant_diagonal_coefficient_v<Arg>> {};


  /// The real part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::real(constant_diagonal_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg>> {};
#endif


  /// The imaginary part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::imag(constant_diagonal_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<short {0}, Scalar> {};
#endif


  /// The coefficient-wise absolute value of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg> >= 0 ?
        constant_diagonal_coefficient_v<Arg> : -constant_diagonal_coefficient_v<Arg>> {};


  /// The coefficient-wise squared absolute value of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs2_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs2_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg> * constant_diagonal_coefficient_v<Arg>> {};


  /// The coefficient-wise square root a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_sqrt_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_sqrt_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<OpenKalman::internal::constexpr_sqrt(constant_diagonal_coefficient_v<Arg>)> {};


  /// The coefficient-wise inverse of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_inverse_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_inverse_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_t<Arg> {1} / constant_diagonal_coefficient_v<Arg>> {};


  /// The coefficient-wise square of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_square_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_square_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg> * constant_diagonal_coefficient_v<Arg>> {};


  /// The coefficient-wise cube of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cube_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cube_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg> * constant_diagonal_coefficient_v<Arg> *
        constant_diagonal_coefficient_v<Arg>> {};


  /// The negation of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<-constant_coefficient_v<Arg>> {};


  /// The conjugate of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<
#if __cpp_nontype_template_args >= 201911L
      complex_number<Scalar> ?
#ifdef __cpp_lib_constexpr_complex
      std::conj(constant_coefficient_v<Arg>) :
#else
      std::complex(std::real(constant_coefficient_v<Arg>), -std::imag(constant_coefficient_v<Arg>)) :
#endif
#endif
      constant_coefficient_v<Arg>> {};


  /// The real part of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::real(constant_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<constant_coefficient_v<Arg>> {};
#endif


  /// The imaginary part of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::imag(constant_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<short {0}, Scalar> {};
#endif


  /// The absolute value of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg> >= 0 ?
      constant_coefficient_v<Arg> : -constant_coefficient_v<Arg>> {};


  /// The squared absolute value of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs2_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs2_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg> * constant_coefficient_v<Arg>> {};


  /// The square root a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_sqrt_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_sqrt_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<OpenKalman::internal::constexpr_sqrt(constant_coefficient_v<Arg>)> {};


  /// The inverse of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_inverse_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_inverse_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_coefficient_t<Arg> {1} / constant_coefficient_v<Arg>> {};


  /// The square of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_square_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_square_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg> * constant_coefficient_v<Arg>> {};


  /// The cube of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cube_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cube_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<
      constant_coefficient_v<Arg> * constant_coefficient_v<Arg> * constant_coefficient_v<Arg>> {};


  /// The logical not of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_boolean_not_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_boolean_not_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<not bool {constant_coefficient_v<Arg>}> {};


  /// The negation of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  /// The conjugate of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  /// Any unary operation on a lower-self-adjoint matrix is also lower-self-adjoint.
  template<typename UnaryOp, typename Arg>
  struct is_lower_self_adjoint_matrix<Eigen::CwiseUnaryOp<UnaryOp, Arg>>
    : std::bool_constant<lower_self_adjoint_matrix<Arg>> {};


  /// Any unary operation on an upper-self-adjoint matrix is also upper-self-adjoint.
  template<typename UnaryOp, typename Arg>
  struct is_upper_self_adjoint_matrix<Eigen::CwiseUnaryOp<UnaryOp, Arg>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  // ---------------- //
  //  CwiseUnaryView  //
  // ---------------- //

  template<typename ViewOp, typename MatrixType>
  struct is_native_eigen_matrix<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
    : std::bool_constant<native_eigen_matrix<MatrixType>> {};


  template<typename ViewOp, typename MatrixType>
  struct is_self_contained<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
    : std::false_type {};


  /// The real part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::real(constant_diagonal_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg>> {};
#endif


  /// The imaginary part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::imag(constant_diagonal_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<short {0}, Scalar> {};
#endif


  /// The real part of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::real(constant_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<constant_coefficient_v<Arg>> {};
#endif


  /// The imaginary part of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::imag(constant_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<short {0}, Scalar> {};
#endif


  /// The real part of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  /// The imaginary part of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  /// The real part of a lower-self-adjoint matrix is also lower-self-adjoint.
  template<typename Scalar, typename Arg>
  struct is_lower_self_adjoint_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<lower_self_adjoint_matrix<Arg>> {};


  /// The real part of an upper-self-adjoint matrix is also upper-self-adjoint.
  template<typename Scalar, typename Arg>
  struct is_upper_self_adjoint_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg>> {};


  /// The real part of a lower-triangular matrix is also lower-triangular.
  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  /// The imaginary part of a lower-triangular matrix is also lower-triangular.
  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  /// The real part of an upper-triangular matrix is also upper-triangular.
  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  /// The imaginary part of an upper-triangular matrix is also upper-triangular.
  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  // ---------- //
  //  Diagonal  //
  // ---------- //

  template<typename MatrixType, int DiagIndex>
  struct is_native_eigen_matrix<Eigen::Diagonal<MatrixType, DiagIndex>>
    : std::bool_constant<native_eigen_matrix<MatrixType>> {};


  template<typename MatrixType, int DiagIndex>
  struct is_self_contained<Eigen::Diagonal<MatrixType, DiagIndex>>
    : std::bool_constant<detail::stores<MatrixType>> {};


  /// The diagonal of a constant-diagonal matrix is constant.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, int DiagIndex>
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
  template<typename MatrixType, int DiagIndex>
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<MatrixType>> {};


  /// The diagonal of a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix MatrixType, int DiagIndex> requires (not constant_diagonal_matrix<MatrixType>)
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
  template<typename MatrixType, int DiagIndex>
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
    constant_matrix<MatrixType> and (not constant_diagonal_matrix<MatrixType>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


  // ---------------- //
  //  DiagonalMatrix  //
  // ---------------- //

  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct is_native_eigen_matrix<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : std::false_type {};


  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct is_self_contained<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : std::true_type {};


  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct is_diagonal_matrix<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : std::true_type {};


  // ----------------- //
  //  DiagonalWrapper  //
  // ----------------- //

  template<typename DiagonalVectorType>
  struct is_native_eigen_matrix<Eigen::DiagonalWrapper<DiagonalVectorType>>
    : std::false_type {};


  template<typename DiagVectorType>
  struct is_self_contained<Eigen::DiagonalWrapper<DiagVectorType>>
    : std::false_type {};


  /// A diagonal wrapper is constant-diagonal if its nested vector is constant.
#ifdef __cpp_concepts
  template<constant_matrix DiagonalVectorType>
  struct constant_diagonal_coefficient<Eigen::DiagonalWrapper<DiagonalVectorType>>
#else
  template<typename DiagonalVectorType>
  struct constant_diagonal_coefficient<Eigen::DiagonalWrapper<DiagonalVectorType>, std::enable_if_t<
    constant_matrix<DiagonalVectorType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<DiagonalVectorType>> {};


  template<typename DiagonalVectorType>
  struct is_diagonal_matrix<Eigen::DiagonalWrapper<DiagonalVectorType>>
    : std::true_type {};


  // ------------------------- //
  //  IndexedView (Eigen 3.4)  //
  // ------------------------- //


  // --------- //
  //  Inverse  //
  // --------- //

  template<typename XprType>
  struct is_native_eigen_matrix<Eigen::Inverse<XprType>>
    : std::false_type {};


  template<typename XprType>
  struct is_self_contained<Eigen::Inverse<XprType>>
    : std::bool_constant<detail::stores<XprType>> {};


  // ----- //
  //  Map  //
  // ----- //

  template<typename PlainObjectType, int MapOptions, typename StrideType>
  struct is_native_eigen_matrix<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
    : std::bool_constant<native_eigen_matrix<PlainObjectType>> {};


  template<typename PlainObjectType, int MapOptions, typename StrideType>
  struct is_self_contained<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
    : std::false_type {};


  // -------- //
  //  Matrix  //
  // -------- //

  template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
  struct is_native_eigen_matrix<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
    : std::true_type {};


  template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
  struct is_self_contained<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
    : std::true_type {};


  // --------------- //
  //  MatrixWrapper  //
  // --------------- //

  template<typename XprType>
  struct is_native_eigen_matrix<Eigen::MatrixWrapper<XprType>>
    : std::true_type {};


  template<typename XprType>
  struct is_self_contained<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<detail::stores<XprType>> {};


  /// A matrix wrapper is constant-diagonal if its nested expression is constant-diagonal.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix XprType>
  struct constant_diagonal_coefficient<Eigen::MatrixWrapper<XprType>>
#else
  template<typename XprType>
  struct constant_diagonal_coefficient<Eigen::MatrixWrapper<XprType>, std::enable_if_t<
    constant_diagonal_matrix<XprType>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<XprType>> {};


  /// A matrix wrapper is constant if its nested expression is constant.
#ifdef __cpp_concepts
  template<constant_matrix XprType>
  struct constant_coefficient<Eigen::MatrixWrapper<XprType>>
#else
  template<typename XprType>
  struct constant_coefficient<Eigen::MatrixWrapper<XprType>, std::enable_if_t<constant_matrix<XprType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<XprType>> {};


  template<typename XprType>
  struct is_diagonal_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<diagonal_matrix<XprType>> {};


  template<typename XprType>
  struct is_lower_self_adjoint_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<lower_self_adjoint_matrix<XprType>> {};


  template<typename XprType>
  struct is_upper_self_adjoint_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<upper_self_adjoint_matrix<XprType>> {};


  template<typename XprType>
  struct is_lower_triangular_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<lower_triangular_matrix<XprType>> {};


  template<typename XprType>
  struct is_upper_triangular_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<upper_triangular_matrix<XprType>> {};


  // ------------------ //
  //  PartialReduxExpr  //
  // ------------------ //

  template<typename MatrixType, typename MemberOp, int Direction>
  struct is_native_eigen_matrix<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
    : std::bool_constant<native_eigen_matrix<MatrixType>> {};


  template<typename MatrixType, typename MemberOp, int Direction>
  struct is_self_contained<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
    : std::bool_constant<detail::stores<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, int p, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_lpnorm<p, Scalar>, Direction>>
#else
  template<typename MatrixType, int p, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_lpnorm<p, Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_squaredNorm<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_squaredNorm<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_norm<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_norm<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_stableNorm<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_stableNorm<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_hypotNorm<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_hypotNorm<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  requires (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
    (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_sum<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_sum<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType> and
      (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
      (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType> *
      (Direction == Eigen::Vertical ? MatrixTraits<MatrixType>::rows : MatrixTraits<MatrixType>::columns),
      std::common_type_t<typename MatrixTraits<MatrixType>::Scalar, Scalar>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_mean<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_mean<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_minCoeff<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_minCoeff<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_maxCoeff<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_maxCoeff<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_all<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_all<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_any<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_any<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Index, int Direction>
  requires (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
    (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_count<Index>, Direction>>
#else
  template<typename MatrixType, typename Index, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_count<Index>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType> and
      (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
      (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)>>
#endif
    : constant_coefficient_type<
      Direction == Eigen::Vertical ? MatrixTraits<MatrixType>::rows : MatrixTraits<MatrixType>::columns,
      typename MatrixTraits<MatrixType>::Scalar> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  requires (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
    (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_prod<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_prod<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType> and
      (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
      (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)>>
#endif
    : constant_coefficient_type<OpenKalman::internal::constexpr_pow(constant_coefficient_v<MatrixType>,
      (Direction == Eigen::Vertical ? MatrixTraits<MatrixType>::rows : MatrixTraits<MatrixType>::columns)),
      std::common_type_t<typename MatrixTraits<MatrixType>::Scalar, Scalar>> {};


  // ------------------- //
  //  PermutationMatrix  //
  // ------------------- //

  template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex>
  struct is_native_eigen_matrix<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>>
    : std::false_type {};


  template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex>
  struct is_self_contained<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>>
    : std::true_type {};


  // -------------------- //
  //  PermutationWrapper  //
  // -------------------- //

  template<typename IndicesType>
  struct is_native_eigen_matrix<Eigen::PermutationWrapper<IndicesType>>
    : std::false_type {};


  template<typename IndicesType>
  struct is_self_contained<Eigen::PermutationWrapper<IndicesType>>
    : std::true_type {};


  // --------- //
  //  Product  //
  // --------- //

  template<typename LhsType, typename RhsType, int Option>
  struct is_native_eigen_matrix<Eigen::Product<LhsType, RhsType, Option>>
    : std::true_type {};


  template<typename LhsType, typename RhsType, int Option>
  struct is_self_contained<Eigen::Product<LhsType, RhsType, Option>>
    : std::bool_constant<detail::stores<LhsType> and detail::stores<RhsType>> {};


  /// A constant-diagonal matrix times another constant-diagonal matrix is constant-diagonal.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>> {};


  /// A constant-diagonal matrix times a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix Arg1, constant_matrix Arg2> requires (not constant_diagonal_matrix<Arg2>)
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    constant_diagonal_matrix<Arg1> and constant_matrix<Arg2> and (not constant_diagonal_matrix<Arg2>)>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> * constant_coefficient_v<Arg2>> {};


  /// A constant matrix times a constant-diagonal matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_diagonal_matrix Arg2> requires (not constant_diagonal_matrix<Arg1>)
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    constant_matrix<Arg1> and constant_diagonal_matrix<Arg2> and (not constant_diagonal_matrix<Arg1>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>> {};


  /// The product of two constant matrices is constant if the rows of A or columns of B is known at compile time.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2> requires (not constant_diagonal_matrix<Arg1>) and
    (not constant_diagonal_matrix<Arg2>) and ((not dynamic_columns<Arg1>) or (not dynamic_rows<Arg2>))
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    constant_matrix<Arg1> and constant_matrix<Arg2> and (not constant_diagonal_matrix<Arg1>) and
    (not constant_diagonal_matrix<Arg2>) and ((not dynamic_columns<Arg1>) or (not dynamic_rows<Arg2>))>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2> *
      int {dynamic_rows<Arg2> ? MatrixTraits<Arg1>::columns : MatrixTraits<Arg2>::rows},
      typename Eigen::Product<Arg1, Arg2>::Scalar> {};


  /// The product of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::Product<Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>> {};


  /// A constant-diagonal matrix times a lower-self-adjoint matrix (or vice versa) is lower-self-adjoint.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (constant_diagonal_matrix<Arg1> and lower_self_adjoint_matrix<Arg2>) or
    (lower_self_adjoint_matrix<Arg1> and constant_diagonal_matrix<Arg2>)
  struct is_lower_self_adjoint_matrix<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_lower_self_adjoint_matrix<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (constant_diagonal_matrix<Arg1> and lower_self_adjoint_matrix<Arg2>) or
    (lower_self_adjoint_matrix<Arg1> and constant_diagonal_matrix<Arg2>)>>
#endif
    : std::true_type {};


  /// A constant-diagonal matrix times an upper-self-adjoint matrix (or vice versa) is upper-self-adjoint.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (constant_diagonal_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>) or
    (upper_self_adjoint_matrix<Arg1> and constant_diagonal_matrix<Arg2>)
  struct is_upper_self_adjoint_matrix<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_upper_self_adjoint_matrix<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (constant_diagonal_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>) or
    (upper_self_adjoint_matrix<Arg1> and constant_diagonal_matrix<Arg2>)>>
#endif
    : std::true_type {};


  /// A constant-diagonal matrix times an upper-triangular matrix (or vice versa) is upper-triangular.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (constant_diagonal_matrix<Arg1> and upper_triangular_matrix<Arg2>) or
    (upper_triangular_matrix<Arg1> and constant_diagonal_matrix<Arg2>)
  struct is_upper_triangular_matrix<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_upper_triangular_matrix<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (constant_diagonal_matrix<Arg1> and upper_triangular_matrix<Arg2>) or
    (upper_triangular_matrix<Arg1> and constant_diagonal_matrix<Arg2>)>>
#endif
    : std::true_type {};


  /// A constant-diagonal matrix times an lower-triangular matrix (or vice versa) is lower-triangular.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (constant_diagonal_matrix<Arg1> and lower_triangular_matrix<Arg2>) or
    (lower_triangular_matrix<Arg1> and constant_diagonal_matrix<Arg2>)
  struct is_lower_triangular_matrix<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_lower_triangular_matrix<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (constant_diagonal_matrix<Arg1> and lower_triangular_matrix<Arg2>) or
    (lower_triangular_matrix<Arg1> and constant_diagonal_matrix<Arg2>)>>
#endif
    : std::true_type {};


  // ----- //
  //  Ref  //
  // ----- //

  template<typename PlainObjectType, int Options, typename StrideType>
  struct is_native_eigen_matrix<Eigen::Ref<PlainObjectType, Options, StrideType>>
    : std::bool_constant<native_eigen_matrix<PlainObjectType>> {};


  template<typename PlainObjectType, int Options, typename StrideType>
  struct is_self_contained<Eigen::Ref<PlainObjectType, Options, StrideType>>
    : std::false_type {};


  // ----------- //
  //  Replicate  //
  // ----------- //

  template<typename MatrixType, int RowFactor, int ColFactor>
  struct is_native_eigen_matrix<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    : std::bool_constant<native_eigen_matrix<MatrixType>> {};


  template<typename MatrixType, int RowFactor, int ColFactor>
  struct is_self_contained<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    : std::bool_constant<detail::stores<MatrixType>> {};


  /// A replication of a constant-diagonal matrix is constant-diagonal if it is replicated only once.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType>
  struct constant_diagonal_coefficient<Eigen::Replicate<MatrixType, 1, 1>>
#else
  template<typename MatrixType>
  struct constant_diagonal_coefficient<Eigen::Replicate<MatrixType, 1, 1>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<MatrixType>> {};


  /// A replication of a constant-diagonal matrix is constant if it is one-by-one.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, int RowFactor, int ColFactor> requires one_by_one_matrix<MatrixType>
  struct constant_diagonal_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
#else
  template<typename MatrixType, int RowFactor, int ColFactor>
  struct constant_diagonal_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType> and one_by_one_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


  /// A replication of a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix MatrixType, int RowFactor, int ColFactor>
  struct constant_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
#else
  template<typename MatrixType, int RowFactor, int ColFactor>
  struct constant_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, std::enable_if_t<
    constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


  template<typename MatrixType>
  struct is_diagonal_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<diagonal_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_lower_self_adjoint_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<lower_self_adjoint_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_upper_self_adjoint_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<upper_self_adjoint_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_lower_triangular_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<lower_triangular_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_upper_triangular_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<upper_triangular_matrix<MatrixType>> {};


  // --------- //
  //  Reverse  //
  // --------- //

  template<typename MatrixType, int Direction>
  struct is_native_eigen_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<native_eigen_matrix<MatrixType>> {};


  template<typename MatrixType, int Direction>
  struct is_self_contained<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<detail::stores<MatrixType>> {};


  /// The double reverse of a constant-diagonal matrix, or any reverse of a one-by-one matrix, is constant-diagonal.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, int Direction>
  requires (Direction == Eigen::BothDirections) or one_by_one_matrix<MatrixType>
  struct constant_coefficient<Eigen::Reverse<MatrixType, Direction>>
#else
  template<typename MatrixType, int Direction>
  struct constant_coefficient<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType> and (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<MatrixType>> {};


  /// The reverse of a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix MatrixType, int Direction>
  struct constant_coefficient<Eigen::Reverse<MatrixType, Direction>>
#else
  template<typename MatrixType, int Direction>
  struct constant_coefficient<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


  template<typename MatrixType, int Direction>
  struct is_diagonal_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<diagonal_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  template<typename MatrixType, int Direction>
  struct is_lower_self_adjoint_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<upper_self_adjoint_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  template<typename MatrixType, int Direction>
  struct is_upper_self_adjoint_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<lower_self_adjoint_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  template<typename MatrixType, int Direction>
  struct is_lower_triangular_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<upper_triangular_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  template<typename MatrixType, int Direction>
  struct is_upper_triangular_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<lower_triangular_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  // -------- //
  //  Select  //
  // -------- //

  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_native_eigen_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    : std::bool_constant<native_eigen_matrix<ThenMatrixType>> {};


  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_self_contained<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    : std::bool_constant<detail::stores<ConditionMatrixType> and detail::stores<ThenMatrixType> and
      detail::stores<ElseMatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, constant_diagonal_matrix ThenMatrixType, typename ElseMatrixType>
  requires constant_coefficient_v<ConditionMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType> and constant_diagonal_matrix<ThenMatrixType> and
      constant_coefficient_v<ConditionMatrixType>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<ThenMatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, constant_diagonal_matrix ElseMatrixType>
  requires (not constant_coefficient_v<ConditionMatrixType>)
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<(not constant_matrix<ConditionMatrixType>) and constant_diagonal_matrix<ElseMatrixType> and
      (not constant_coefficient_v<ConditionMatrixType>)>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<ElseMatrixType>> {};


#ifdef __cpp_concepts
  template<typename ConditionMatrixType, constant_diagonal_matrix ThenMatrixType,
    constant_diagonal_matrix ElseMatrixType> requires (not constant_matrix<ConditionMatrixType>) and
    (constant_diagonal_coefficient_v<ThenMatrixType> == constant_diagonal_coefficient_v<ElseMatrixType>)
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_diagonal_matrix<ThenMatrixType> and constant_diagonal_matrix<ElseMatrixType> and
      (not constant_matrix<ConditionMatrixType>) and
      (constant_diagonal_coefficient_v<ThenMatrixType> == constant_diagonal_coefficient_v<ElseMatrixType>)>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<ThenMatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, constant_matrix ThenMatrixType, typename ElseMatrixType>
  requires constant_coefficient_v<ConditionMatrixType> and (not constant_diagonal_matrix<ThenMatrixType>)
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType> and constant_matrix<ThenMatrixType> and
      (constant_coefficient_v<ConditionMatrixType>) and (not constant_diagonal_matrix<ThenMatrixType>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<ThenMatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, constant_matrix ElseMatrixType>
  requires (not constant_coefficient_v<ConditionMatrixType>) and (not constant_diagonal_matrix<ElseMatrixType>)
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType> and constant_matrix<ElseMatrixType> and
      (not constant_coefficient_v<ConditionMatrixType>) and (not constant_diagonal_matrix<ElseMatrixType>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<ElseMatrixType>> {};


#ifdef __cpp_concepts
  template<typename ConditionMatrixType, constant_matrix ThenMatrixType, constant_matrix ElseMatrixType>
  requires (not constant_matrix<ConditionMatrixType>) and
    (constant_coefficient_v<ThenMatrixType> == constant_coefficient_v<ElseMatrixType>) and
    (not constant_diagonal_matrix<ThenMatrixType> or not constant_diagonal_matrix<ElseMatrixType>)
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ThenMatrixType> and constant_matrix<ElseMatrixType> and
      (not constant_matrix<ConditionMatrixType>) and
      (constant_coefficient_v<ThenMatrixType> == constant_coefficient_v<ElseMatrixType>) and
      (not constant_diagonal_matrix<ThenMatrixType> or not constant_diagonal_matrix<ElseMatrixType>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<ThenMatrixType>> {};


  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_diagonal_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    : std::bool_constant<constant_matrix<ConditionMatrixType> and
      ((diagonal_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (diagonal_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>))> {};


  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_lower_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    : std::bool_constant<constant_matrix<ConditionMatrixType> and
      ((lower_self_adjoint_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (lower_self_adjoint_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>))> {};


  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_upper_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    : std::bool_constant<constant_matrix<ConditionMatrixType> and
      ((upper_self_adjoint_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (upper_self_adjoint_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>))> {};


  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_lower_triangular_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    : std::bool_constant<constant_matrix<ConditionMatrixType> and
      ((lower_triangular_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (lower_triangular_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>))> {};


  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_upper_triangular_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    : std::bool_constant<constant_matrix<ConditionMatrixType> and
      ((upper_triangular_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (upper_triangular_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>))> {};


  // ----------------- //
  //  SelfAdjointView  //
  // ----------------- //

  template<typename MatrixType, unsigned int UpLo>
  struct is_native_eigen_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::false_type {};


  template<typename MatrixType, unsigned int UpLo>
  struct is_self_contained<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::false_type {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, unsigned int UpLo>
  struct constant_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
  template<typename MatrixType, unsigned int UpLo>
  struct constant_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, unsigned int UpLo>
  struct constant_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
  template<typename MatrixType, unsigned int UpLo>
  struct constant_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


  template<typename MatrixType, unsigned int UpLo>
  struct is_diagonal_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::bool_constant<diagonal_matrix<MatrixType>> {};


  template<typename MatrixType, unsigned int UpLo>
  struct is_upper_self_adjoint_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::bool_constant<(UpLo & Eigen::Upper) != 0> {};


  template<typename MatrixType, unsigned int UpLo>
  struct is_lower_self_adjoint_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::bool_constant<(UpLo & Eigen::Lower) != 0> {};


  // ------- //
  //  Solve  //
  // ------- //

  template<typename Decomposition, typename RhsType>
  struct is_native_eigen_matrix<Eigen::Solve<Decomposition, RhsType>>
    : std::bool_constant<native_eigen_matrix<RhsType>> {};


  template<typename Decomposition, typename RhsType>
  struct is_self_contained<Eigen::Solve<Decomposition, RhsType>>
    : std::bool_constant<detail::stores<Decomposition> and detail::stores<RhsType>> {};


  // ----------- //
  //  Transpose  //
  // ----------- //

  template<typename MatrixType>
  struct is_native_eigen_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<native_eigen_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_self_contained<Eigen::Transpose<MatrixType>>
    : std::bool_constant<detail::stores<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType>
  struct constant_diagonal_coefficient<Eigen::Transpose<MatrixType>>
#else
  template<typename MatrixType>
  struct constant_diagonal_coefficient<Eigen::Transpose<MatrixType>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType>
  struct constant_coefficient<Eigen::Transpose<MatrixType>>
#else
  template<typename MatrixType>
  struct constant_coefficient<Eigen::Transpose<MatrixType>, std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


  template<typename MatrixType>
  struct is_diagonal_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<diagonal_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_lower_self_adjoint_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<upper_self_adjoint_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_upper_self_adjoint_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<lower_self_adjoint_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_lower_triangular_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<upper_triangular_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_upper_triangular_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<lower_triangular_matrix<MatrixType>> {};


  // ---------------- //
  //  TriangularView  //
  // ---------------- //

  template<typename MatrixType, unsigned int Mode>
  struct is_native_eigen_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::false_type {};


  template<typename MatrixType, unsigned int Mode>
  struct is_self_contained<Eigen::TriangularView<MatrixType, Mode>>
    : std::false_type {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, unsigned int Mode>
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>>
#else
  template<typename MatrixType, unsigned int Mode>
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<zero_matrix MatrixType, unsigned int Mode>
  struct constant_coefficient<Eigen::TriangularView<MatrixType, Mode>>
#else
  template<typename MatrixType, unsigned int Mode>
  struct constant_coefficient<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<zero_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


  template<typename MatrixType, unsigned int Mode>
  struct is_diagonal_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<diagonal_matrix<MatrixType>> {};


  template<typename MatrixType, unsigned int Mode>
  struct is_lower_triangular_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<lower_triangular_matrix<MatrixType> or (Mode & Eigen::Lower) != 0> {};


  template<typename MatrixType, unsigned int Mode>
  struct is_upper_triangular_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<upper_triangular_matrix<MatrixType> or (Mode & Eigen::Upper) != 0> {};


  // ------------- //
  //  VectorBlock  //
  // ------------- //

  template<typename VectorType, int Size>
  struct is_native_eigen_matrix<Eigen::VectorBlock<VectorType, Size>>
    : std::bool_constant<native_eigen_matrix<VectorType>> {};


  template<typename VectorType, int Size>
  struct is_self_contained<Eigen::VectorBlock<VectorType, Size>>
    : std::bool_constant<detail::stores<VectorType>> {};


  /// A segment taken from a constant vector is constant.
#ifdef __cpp_concepts
  template<constant_matrix VectorType, int Size>
  struct constant_coefficient<Eigen::VectorBlock<VectorType, Size>>
#else
    template<typename VectorType, int Size>
    struct constant_coefficient<Eigen::VectorBlock<VectorType, Size>, std::enable_if_t<constant_matrix<VectorType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<VectorType>> {};


  // -------------- //
  //  VectorWiseOp  //
  // -------------- //

  // A vectorwise operation on a zero vector is zero.
  template<typename ExpressionType, int Direction>
  struct is_native_eigen_matrix<Eigen::VectorwiseOp<ExpressionType, Direction>>
    : std::false_type {};


  template<typename ExpressionType, int Direction>
  struct is_self_contained<Eigen::VectorwiseOp<ExpressionType, Direction>>
    : std::bool_constant<detail::stores<ExpressionType>> {};


  /// A vectorwise operation on a constant vector is constant.
#ifdef __cpp_concepts
  template<constant_matrix ExpressionType, int Direction>
  struct constant_coefficient<Eigen::VectorwiseOp<ExpressionType, Direction>>
#else
  template<typename ExpressionType, int Direction>
  struct constant_coefficient<Eigen::VectorwiseOp<ExpressionType, Direction>,
    std::enable_if_t<constant_matrix<ExpressionType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<ExpressionType>> {};


} // namespace OpenKalman::internal

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
