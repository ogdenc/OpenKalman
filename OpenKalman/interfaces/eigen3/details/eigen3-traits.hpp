/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_HPP
#define OPENKALMAN_EIGEN3_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{

  // ------------------------------------------------------- //
  //    Concepts / type traits for new Eigen matrix types    //
  // ------------------------------------------------------- //
  namespace Eigen3
  {

    namespace detail
    {
      template<typename T>
      struct is_eigen_self_adjoint_expr : OpenKalman::internal::class_trait<is_eigen_self_adjoint_expr, T> {};

      template<typename BaseMatrix, TriangleType storage_triangle>
      struct is_eigen_self_adjoint_expr<SelfAdjointMatrix<BaseMatrix, storage_triangle>> : std::true_type {};
    }

    /**
     * Type T is a self-adjoint matrix based on the Eigen library.
     *
     * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<T>::value;
#else
    inline constexpr bool eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<T>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_eigen_triangular_expr : OpenKalman::internal::class_trait<is_eigen_triangular_expr, T> {};

      template<typename BaseMatrix, TriangleType triangle_type>
      struct is_eigen_triangular_expr<TriangularMatrix<BaseMatrix, triangle_type>> : std::true_type {};
    }

    /**
     * A triangular matrix based on the Eigen library.
     *
     * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_triangular_expr = detail::is_eigen_triangular_expr<T>::value;
#else
    inline constexpr bool eigen_triangular_expr = detail::is_eigen_triangular_expr<T>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_eigen_diagonal_expr : OpenKalman::internal::class_trait<is_eigen_diagonal_expr, T> {};

      template<typename BaseMatrix>
      struct is_eigen_diagonal_expr<DiagonalMatrix<BaseMatrix>> : std::true_type {};
    }

    /**
     * A diagonal matrix based on the Eigen library.
     *
     * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_diagonal_expr = detail::is_eigen_diagonal_expr<T>::value;
#else
    inline constexpr bool eigen_diagonal_expr = detail::is_eigen_diagonal_expr<T>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_eigen_zero_expr : OpenKalman::internal::class_trait<is_eigen_zero_expr, T> {};

      template<typename BaseMatrix>
      struct is_eigen_zero_expr<ZeroMatrix<BaseMatrix>> : std::true_type {};
    }

    /**
     * A zero matrix based on the Eigen library. (All coefficients are zero.)
     *
     * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_zero_expr = detail::is_eigen_zero_expr<T>::value;
#else
    inline constexpr bool eigen_zero_expr = detail::is_eigen_zero_expr<T>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_from_euclidean_expr : OpenKalman::internal::class_trait<is_from_euclidean_expr, T> {};

      template<typename Coefficients, typename BaseMatrix>
      struct is_from_euclidean_expr<FromEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};
    }

    /**
     * An expression converting each column vector in a Euclidean Eigen matrix from Euclidean space.
     *
     * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept from_euclidean_expr = detail::is_from_euclidean_expr<T>::value;
#else
    inline constexpr bool from_euclidean_expr = detail::is_from_euclidean_expr<T>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_to_euclidean_expr : OpenKalman::internal::class_trait<is_to_euclidean_expr, T> {};

      template<typename Coefficients, typename BaseMatrix>
      struct is_to_euclidean_expr<ToEuclideanExpr<Coefficients, BaseMatrix>> : std::true_type {};
    }

    /**
     * An expression converting each column vector in an Eigen matrix to Euclidean space.
     *
     * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept to_euclidean_expr = detail::is_to_euclidean_expr<T>::value;
#else
    inline constexpr bool to_euclidean_expr = detail::is_to_euclidean_expr<T>::value;
#endif


    /**
     * Either from_euclidean_expr or to_euclidean_expr.
     *
     * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept euclidean_expr = from_euclidean_expr<T> or to_euclidean_expr<T>;
#else
    inline constexpr bool euclidean_expr = from_euclidean_expr<T> or to_euclidean_expr<T>;
#endif


    /**
     * An object that is a native Eigen3 matrix (i.e., a class in the Eigen library descending from Eigen::MatrixBase).
     *
     * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_native = std::derived_from<std::decay_t<T>, Eigen::MatrixBase<std::decay_t<T>>> and
      not std::derived_from<std::decay_t<T>, internal::Eigen3Base<std::decay_t<T>>>;
#else
    inline constexpr bool eigen_native = std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> and
      not std::is_base_of_v<internal::Eigen3Base<std::decay_t<T>>, std::decay_t<T>>;
#endif


    /**
     * T is either a native Eigen matrix or a zero Eigen matrix.
     *
     * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_matrix = eigen_native<T> or eigen_zero_expr<T>;
#else
    inline constexpr bool eigen_matrix = eigen_native<T> or eigen_zero_expr<T>;
#endif

  } // namespace Eigen3


  // -------------------------------------------------------------------------------- //
  //  General OpenKalman concepts / type traits as defined for new Eigen matrix types //
  // -------------------------------------------------------------------------------- //

  // covariance_base
#ifdef __cpp_concepts
  template<typename T> requires
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    (Eigen3::eigen_native<T> and (is_triangular_v<T> or is_self_adjoint_v<T>))
  struct is_covariance_base<T>
#else
  template<typename T>
  struct is_covariance_base<T, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    (Eigen3::eigen_native<T> and (is_triangular_v<T> or is_self_adjoint_v<T>))>>
#endif
    : std::true_type {};


  // typed_matrix_base
#ifdef __cpp_concepts
  template<typename T> requires
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    Eigen3::to_euclidean_expr<T> or
    Eigen3::from_euclidean_expr<T> or
    Eigen3::eigen_native<T>
  struct is_typed_matrix_base<T>
#else
  template<typename T>
  struct is_typed_matrix_base<T, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    Eigen3::to_euclidean_expr<T> or
    Eigen3::from_euclidean_expr<T> or
    Eigen3::eigen_native<T>>>
#endif
    : std::true_type {};


  // ---------------------------------------------------------------

#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires std::same_as<T, std::decay_t<T>>
  struct is_element_gettable<T, 2>
#else
  template<typename T>
  struct is_element_gettable<T, 2, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and
    Eigen3::eigen_native<T>>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires std::same_as<T, std::decay_t<T>>
  struct is_element_gettable<T, 1>
#else
  template<typename T>
  struct is_element_gettable<T, 1, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and
    Eigen3::eigen_native<T>>>
#endif
    : std::bool_constant<MatrixTraits<T>::columns == 1> {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires std::same_as<T, std::decay_t<T>>
  struct is_element_settable<T, 2>
#else
  template<typename T>
  struct is_element_settable<T, 2, std::enable_if_t<
    std::is_same_v<T, std::decay_t<T>> and Eigen3::eigen_native<T>>>
#endif
    : std::bool_constant<not std::is_const_v<std::remove_reference_t<T>> and
      static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit)> {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires std::is_same_v<T, std::decay_t<T>>
  struct is_element_settable<T, 1>
#else
  template<typename T>
  struct is_element_settable<T, 1, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and
    Eigen3::eigen_native<T>>>
#endif
    : std::bool_constant<MatrixTraits<T>::columns == 1 and not std::is_const_v<std::remove_reference_t<T>> and
      static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit)> {};


  /////////////////////////////
  //    SelfAdjointMatrix    //
  /////////////////////////////

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_zero<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>>
    : std::bool_constant<is_zero_v<BaseMatrix>> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_identity<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>>
    : std::bool_constant<is_identity_v<BaseMatrix>> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_diagonal<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>,
    std::enable_if_t<not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix> and not is_1by1_v<BaseMatrix>>>
    : std::bool_constant<is_diagonal_v<BaseMatrix> or storage_triangle == TriangleType::diagonal> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_self_adjoint<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix> and storage_triangle != TriangleType::diagonal>>
    : std::true_type {};

  namespace Eigen3
  {
    template<typename T>
    struct is_upper_storage_triangle : OpenKalman::internal::class_trait<is_upper_storage_triangle, T> {};

    template<typename BaseMatrix>
    struct is_upper_storage_triangle<Eigen3::SelfAdjointMatrix<BaseMatrix, TriangleType::upper>>
      : std::true_type {};

    template<typename T>
    inline constexpr bool is_upper_storage_triangle_v = is_upper_storage_triangle<T>::value;

    template<typename T>
    struct is_lower_storage_triangle : OpenKalman::internal::class_trait<is_lower_storage_triangle, T> {};

    template<typename BaseMatrix>
    struct is_lower_storage_triangle<Eigen3::SelfAdjointMatrix<BaseMatrix, TriangleType::lower>>
      : std::true_type {};

    template<typename T>
    inline constexpr bool is_lower_storage_triangle_v = is_lower_storage_triangle<T>::value;
  } // namespace Eigen3


  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_strict<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>> : is_strict<BaseMatrix> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_element_gettable<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>, 2>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, 2> or is_element_gettable_v<BaseMatrix, 1>> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_element_gettable<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>, 1>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, 1> or
      (is_element_gettable_v<BaseMatrix, 2> and storage_triangle == TriangleType::diagonal)> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_element_settable<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>, 2>
    : std::bool_constant<is_element_settable_v<BaseMatrix, 2> or is_element_settable_v<BaseMatrix, 1>> {};

  template<typename BaseMatrix, TriangleType storage_triangle>
  struct is_element_settable<Eigen3::SelfAdjointMatrix<BaseMatrix, storage_triangle>, 1>
    : std::bool_constant<is_element_settable_v<BaseMatrix, 1> or
      (is_element_settable_v<BaseMatrix, 2> and storage_triangle == TriangleType::diagonal)> {};


  ////////////////////////////
  //    TriangularMatrix    //
  ////////////////////////////

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_zero<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>>
    : std::bool_constant<is_zero_v<BaseMatrix>> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_identity<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>>
    : std::bool_constant<is_identity_v<BaseMatrix>> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_diagonal<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>,
    std::enable_if_t<not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix> and not is_1by1_v<BaseMatrix>>>
    : std::bool_constant<is_diagonal_v<BaseMatrix> or triangle_type == TriangleType::diagonal> {};

  template<typename BaseMatrix>
  struct is_lower_triangular<Eigen3::TriangularMatrix<BaseMatrix, TriangleType::lower>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix>
  struct is_upper_triangular<Eigen3::TriangularMatrix<BaseMatrix, TriangleType::upper>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_strict<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>> : is_strict<BaseMatrix> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_element_gettable<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>, 2>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, 2> or is_element_gettable_v<BaseMatrix, 1>> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_element_gettable<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>, 1>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, 1> or
      (is_element_gettable_v<BaseMatrix, 2> and triangle_type == TriangleType::diagonal)> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_element_settable<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>, 2>
    : std::bool_constant<is_element_settable_v<BaseMatrix, 2> or is_element_settable_v<BaseMatrix, 1>> {};

  template<typename BaseMatrix, TriangleType triangle_type>
  struct is_element_settable<Eigen3::TriangularMatrix<BaseMatrix, triangle_type>, 1>
    : std::bool_constant<is_element_settable_v<BaseMatrix, 1> or
      (is_element_settable_v<BaseMatrix, 2> and triangle_type == TriangleType::diagonal)> {};


  // -------------------- //
  //    DiagonalMatrix    //
  // -------------------- //

  template<typename BaseMatrix>
  struct is_zero<Eigen3::DiagonalMatrix<BaseMatrix>>
    : std::bool_constant<is_zero_v<BaseMatrix>> {};

  template<typename BaseMatrix>
  struct is_diagonal<Eigen3::DiagonalMatrix<BaseMatrix>,
    std::enable_if_t<not is_zero_v<BaseMatrix> and not is_1by1_v<BaseMatrix>>>
    : std::true_type {};

  template<typename BaseMatrix>
  struct is_strict<Eigen3::DiagonalMatrix<BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::DiagonalMatrix<BaseMatrix>, N> : std::bool_constant<(N == 1 or  N == 2) and
      (is_element_gettable_v<BaseMatrix, 1> or is_element_gettable_v<BaseMatrix, 2>)> {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::DiagonalMatrix<BaseMatrix>, N> : std::bool_constant<(N == 1 or  N == 2) and
      (is_element_settable_v<BaseMatrix, 1> or is_element_settable_v<BaseMatrix, 2>)> {};


  // ------------------------------------------------------- //
  //    ZeroMatrix and other known Eigen zero expressions    //
  // ------------------------------------------------------- //

  template<typename BaseMatrix>
  struct is_zero<Eigen3::ZeroMatrix<BaseMatrix>> : std::true_type {};

  template<typename ArgType>
  struct is_strict<Eigen3::ZeroMatrix<ArgType>> : std::true_type {};

  template<typename Arg1, typename Arg2>
  struct is_zero<Eigen::Product<Arg1, Arg2>,
    std::enable_if_t<is_zero_v<Arg1> or is_zero_v<Arg2>>>
    : std::true_type {};

  template<typename Arg1, typename Arg2>
  struct is_zero<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<is_zero_v<Arg1> or is_zero_v<Arg2>>>
    : std::true_type {};

  template<typename Arg1, typename Arg2>
  struct is_zero<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<is_zero_v<Arg1> and is_zero_v<Arg2>>>
    : std::true_type {};

  template<typename Arg1, typename Arg2>
  struct is_zero<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<(is_zero_v<Arg1> and is_zero_v<Arg2>) or (is_identity_v<Arg1> and is_identity_v<Arg2>)>>
    : std::true_type {};

  template<typename Arg>
  struct is_zero<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>,
    std::enable_if_t<is_zero_v<Arg>>>
    : std::true_type {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::ZeroMatrix<BaseMatrix>, N>
    : std::bool_constant<N == 2 or (N == 1 and MatrixTraits<BaseMatrix>::columns == 1)> {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::ZeroMatrix<BaseMatrix>, N> : std::false_type {};


  ///////////////////////////
  //    ToEuclideanExpr    //
  ///////////////////////////

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<Coefficients::axes_only and is_element_settable_v<BaseMatrix, N>> {};


  /////////////////////////////
  //    FromEuclideanExpr    //
  /////////////////////////////

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<Eigen3::FromEuclideanExpr<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::FromEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::FromEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<(Coefficients::axes_only and is_element_settable_v<BaseMatrix, N>) or
    (Eigen3::to_euclidean_expr<BaseMatrix> and is_element_settable_v<typename MatrixTraits<BaseMatrix>::BaseMatrix, N>)>
    {};


  ///////////////////////////////////////////////////////////////
  //    Eigen Identity and other known diagonal expressions    //
  ///////////////////////////////////////////////////////////////

  template<typename Arg>
  struct is_identity<Eigen3::IdentityMatrix<Arg>>
    : std::bool_constant<Arg::RowsAtCompileTime == Arg::ColsAtCompileTime> {};

  /// Product of two identity matrices is also identity.
  template<typename Arg1, typename Arg2>
  struct is_identity<Eigen::Product<Arg1, Arg2>>
    : std::bool_constant<is_identity_v<Arg1> and is_identity_v<Arg2>> {};

  /// Product of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::Product<Arg1, Arg2>,
    std::enable_if_t<not((is_zero_v<Arg1> or is_zero_v<Arg2>) or
      (is_identity_v<Arg1> and is_identity_v<Arg2>) or
      (Arg1::RowsAtCompileTime == 1 and Arg2::ColsAtCompileTime == 1))>>
    : std::bool_constant<is_diagonal_v<Arg1> and is_diagonal_v<Arg2>> {};

  /// Diagonal matrix times a scalar is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<not is_zero_v<Arg1> and not is_zero_v<Arg2> and not is_1by1_v<Arg1> and not is_1by1_v<Arg2>>>
    : std::bool_constant<is_diagonal_v<Arg1> or is_diagonal_v<Arg2>> {};

  /// Diagonal matrix divided by a scalar is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_quotient_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<not is_zero_v<Arg1>>>
    : std::bool_constant<is_diagonal_v<Arg1>> {};

  /// Sum of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<
      not (is_zero_v<Arg1> and is_zero_v<Arg2>) and
      not (is_1by1_v<Arg1> and is_1by1_v<Arg2>)>>
    : std::bool_constant<is_diagonal_v<Arg1> and is_diagonal_v<Arg2>> {};

  /// Difference of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<
      not (is_zero_v<Arg1> and is_zero_v<Arg2>) and
      not (is_identity_v<Arg1> and is_identity_v<Arg2>) and
      not (is_1by1_v<Arg1> and is_1by1_v<Arg2>)>>
    : std::bool_constant<is_diagonal_v<Arg1> and is_diagonal_v<Arg2>> {};

  /// The negation of an identity matrix is diagonal.
  template<typename Arg>
  struct is_diagonal<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>,
    std::enable_if_t<not is_zero_v<Arg> and not is_1by1_v<Arg>>>
    : std::bool_constant<is_diagonal_v<Arg>> {};

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
