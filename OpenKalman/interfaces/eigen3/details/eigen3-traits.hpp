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


  namespace internal
  {
    // Defines the is_covariance_base type trait specifically for Eigen3.
#ifdef __cpp_concepts
    template<typename T>
    requires
      Eigen3::eigen_self_adjoint_expr<T> or
      Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or
      Eigen3::eigen_zero_expr<T> or
      (Eigen3::eigen_native<T> and (triangular_matrix<T> or self_adjoint_matrix<T>))
    struct is_covariance_base<T>
#else
    template<typename T>
    struct is_covariance_base<T, std::enable_if_t<
      Eigen3::eigen_self_adjoint_expr<T> or
      Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or
      Eigen3::eigen_zero_expr<T> or
      (Eigen3::eigen_native<T> and (triangular_matrix<T> or self_adjoint_matrix<T>))>>
#endif
    : std::true_type {};


    // Defines the is_typed_matrix_base type trait specifically for Eigen3.
#ifdef __cpp_concepts
    template<typename T>
    requires
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
  } // internal


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


  // -------------------------------- //
  //   Type traits for Eigen3 types   //
  // -------------------------------- //

  namespace internal
  {
    // ---------------- //
    //  is_zero_matrix  //
    // ---------------- //

#ifdef __cpp_concepts
    template<typename T> requires Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T>
    struct is_zero_matrix<T>
#else
    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T>>>
#endif
    : is_zero_matrix<typename MatrixTraits<T>::BaseMatrix> {};


#ifdef __cpp_concepts
    template<Eigen3::eigen_zero_expr T>
    struct is_zero_matrix<T>
#else
    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<Eigen3::eigen_zero_expr<T>>>
#endif
    : std::true_type {};


    // The product of two zero matrices is zero.
    template<typename Arg1, typename Arg2>
#ifdef __cpp_concepts
    requires zero_matrix<Arg1> or zero_matrix<Arg2>
    struct is_zero_matrix<Eigen::Product<Arg1, Arg2>>
#else
    struct is_zero_matrix<Eigen::Product<Arg1, Arg2>,
      std::enable_if_t<zero_matrix<Arg1> or zero_matrix<Arg2>>>
#endif
    : std::true_type {};


    // The product of a zero matrix and a scalar (or vice versa) is zero.
    template<typename Arg1, typename Arg2>
#ifdef __cpp_concepts
    requires zero_matrix<Arg1> or zero_matrix<Arg2>
    struct is_zero_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
    struct is_zero_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
      std::enable_if_t<zero_matrix<Arg1> or zero_matrix<Arg2>>>
#endif
    : std::true_type {};


    // The sum of two zero matrices is zero.
#ifdef __cpp_concepts
    template<zero_matrix Arg1, zero_matrix Arg2>
    struct is_zero_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct is_zero_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
      std::enable_if_t<zero_matrix<Arg1> and zero_matrix<Arg2>>>
#endif
    : std::true_type {};


    // The difference between two zero or identity matrices is zero.
    template<typename Arg1, typename Arg2>
#ifdef __cpp_concepts
    requires (zero_matrix<Arg1> and zero_matrix<Arg2>) or (identity_matrix<Arg1> and identity_matrix<Arg2>)
    struct is_zero_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
    struct is_zero_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
      std::enable_if_t<(zero_matrix<Arg1> and zero_matrix<Arg2>) or (identity_matrix<Arg1> and identity_matrix<Arg2>)>>
#endif
      : std::true_type {};


    // The negation of a zero matrix is zero.
#ifdef __cpp_concepts
    template<zero_matrix Arg>
    struct is_zero_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>>
#else
    template<typename Arg>
    struct is_zero_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>,
      std::enable_if_t<zero_matrix<Arg>>>
#endif
    : std::true_type {};


    // -------------------- //
    //  is_identity_matrix  //
    // -------------------- //

#ifdef __cpp_concepts
    template<typename T> requires Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T>
    struct is_identity_matrix<T>
#else
    template<typename T>
    struct is_identity_matrix<T,
      std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T>>>
#endif
      : is_identity_matrix<typename MatrixTraits<T>::BaseMatrix> {};


    template<typename Arg>
    struct is_identity_matrix<Eigen3::IdentityMatrix<Arg>>
      : std::bool_constant<Arg::RowsAtCompileTime == Arg::ColsAtCompileTime> {};


    // The product of two identity matrices is also identity.
    template<typename Arg1, typename Arg2>
    struct is_identity_matrix<Eigen::Product<Arg1, Arg2>>
      : std::bool_constant<identity_matrix<Arg1> and identity_matrix<Arg2>> {};


    // ------------- //
    //  is_diagonal  //
    // ------------- //

#ifdef __cpp_concepts
    template<Eigen3::eigen_diagonal_expr T>
    struct is_diagonal_matrix<T>
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<Eigen3::eigen_diagonal_expr<T>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<Eigen3::eigen_self_adjoint_expr T> requires diagonal_matrix<typename MatrixTraits<T>::BaseMatrix> or
      (MatrixTraits<T>::storage_type == TriangleType::diagonal)
    struct is_diagonal_matrix<T>
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T> and
      (diagonal_matrix<typename MatrixTraits<T>::BaseMatrix> or
        MatrixTraits<T>::storage_type == TriangleType::diagonal)>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<Eigen3::eigen_triangular_expr T> requires diagonal_matrix<typename MatrixTraits<T>::BaseMatrix> or
      (MatrixTraits<T>::triangle_type == TriangleType::diagonal)
    struct is_diagonal_matrix<T>
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<Eigen3::eigen_triangular_expr<T> and
      (diagonal_matrix<typename MatrixTraits<T>::BaseMatrix> or
        MatrixTraits<T>::triangle_type == TriangleType::diagonal)>>
#endif
      : std::true_type {};


    // The product of two diagonal matrices is also diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg1, diagonal_matrix Arg2>
    struct is_diagonal_matrix<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct is_diagonal_matrix<Eigen::Product<Arg1, Arg2>,
      std::enable_if_t<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>>>
#endif
      : std::true_type {};


    // A diagonal matrix times a scalar (or vice versa) is also diagonal.
#ifdef __cpp_concepts
    template<typename Arg1, typename Arg2> requires diagonal_matrix<Arg1> or diagonal_matrix<Arg2>
    struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
      std::enable_if_t<diagonal_matrix<Arg1> or diagonal_matrix<Arg2>>>
#endif
      : std::true_type {};


    // A diagonal matrix divided by a scalar is also diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg1, typename Arg2>
    struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_quotient_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_quotient_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
      std::enable_if_t<diagonal_matrix<Arg1>>>
#endif
      : std::true_type {};


    // The sum of two diagonal matrices is also diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg1, diagonal_matrix Arg2>
    struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
      std::enable_if_t<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>>>
#endif
      : std::true_type {};


    // The difference between two diagonal matrices is also diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg1, diagonal_matrix Arg2>
    struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
      Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
      std::enable_if_t<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>>>
#endif
      : std::true_type {};


    // The negation of an identity matrix is diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>>
#else
    template<typename Arg>
    struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>,
      std::enable_if_t<diagonal_matrix<Arg>>>
#endif
      : std::true_type {};


    // ------------------------ //
    //  is_self_adjoint_matrix  //
    // ------------------------ //

#ifdef __cpp_concepts
    template<Eigen3::eigen_self_adjoint_expr T>
    struct is_self_adjoint_matrix<T>
#else
    template<typename T>
    struct is_self_adjoint_matrix<T, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T>>>
#endif
      : std::true_type {};


    // ---------------------------- //
    //  is_lower_triangular_matrix  //
    // ---------------------------- //

#ifdef __cpp_concepts
    template<Eigen3::eigen_triangular_expr T> requires (MatrixTraits<T>::triangle_type == TriangleType::lower)
    struct is_lower_triangular_matrix<T>
#else
    template<typename T>
    struct is_lower_triangular_matrix<T,
      std::enable_if_t<Eigen3::eigen_triangular_expr<T> and (MatrixTraits<T>::triangle_type == TriangleType::lower)>>
#endif
    : std::true_type {};


    // ---------------------------- //
    //  is_upper_triangular_matrix  //
    // ---------------------------- //

#ifdef __cpp_concepts
    template<Eigen3::eigen_triangular_expr T> requires (MatrixTraits<T>::triangle_type == TriangleType::upper)
    struct is_upper_triangular_matrix<T>
#else
    template<typename T>
    struct is_upper_triangular_matrix<T,
      std::enable_if_t<Eigen3::eigen_triangular_expr<T> and (MatrixTraits<T>::triangle_type == TriangleType::upper)>>
#endif
      : std::true_type {};


    // ------------------- //
    //  is_self_contained  //
    // ------------------- //

#ifdef __cpp_concepts
    template<typename T> requires Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or Eigen3::to_euclidean_expr<T> or Eigen3::from_euclidean_expr<T>
    struct is_self_contained<T>
#else
    template<typename T>
    struct is_self_contained<T, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or Eigen3::to_euclidean_expr<T> or Eigen3::from_euclidean_expr<T>>>
#endif
      : std::bool_constant<self_contained<typename MatrixTraits<T>::BaseMatrix> and
          not std::is_reference_v<typename MatrixTraits<T>::BaseMatrix>> {};


#ifdef __cpp_concepts
    template<Eigen3::eigen_zero_expr T>
    struct is_self_contained<T>
#else
    template<typename T>
    struct is_self_contained<T, std::enable_if_t<Eigen3::eigen_zero_expr<T>>>
#endif
      : std::true_type {};



}


  /////////////////////////////
  //    SelfAdjointMatrix    //
  /////////////////////////////

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

  template<typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::DiagonalMatrix<BaseMatrix>, N> : std::bool_constant<(N == 1 or  N == 2) and
      (is_element_gettable_v<BaseMatrix, 1> or is_element_gettable_v<BaseMatrix, 2>)> {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::DiagonalMatrix<BaseMatrix>, N> : std::bool_constant<(N == 1 or  N == 2) and
      (is_element_settable_v<BaseMatrix, 1> or is_element_settable_v<BaseMatrix, 2>)> {};


  // ------------------------------------------------------- //
  //    ZeroMatrix and other known Eigen zero expressions    //
  // ------------------------------------------------------- //

  template<typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::ZeroMatrix<BaseMatrix>, N>
    : std::bool_constant<N == 2 or (N == 1 and MatrixTraits<BaseMatrix>::columns == 1)> {};

  template<typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::ZeroMatrix<BaseMatrix>, N> : std::false_type {};


  ///////////////////////////
  //    ToEuclideanExpr    //
  ///////////////////////////

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Eigen3::ToEuclideanExpr<Coefficients, BaseMatrix>, N>
  : std::bool_constant<Coefficients::axes_only and is_element_settable_v<BaseMatrix, N>> {};


  /////////////////////////////
  //    FromEuclideanExpr    //
  /////////////////////////////

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

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
