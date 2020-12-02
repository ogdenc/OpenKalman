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
      struct is_eigen_self_adjoint_expr : std::false_type {};

      template<typename NestedMatrix, TriangleType storage_triangle>
      struct is_eigen_self_adjoint_expr<SelfAdjointMatrix<NestedMatrix, storage_triangle>> : std::true_type {};
    }


    /**
     * Type T is a self-adjoint matrix based on the Eigen library.
     * /note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<std::decay_t<T>>::value;
#else
    inline constexpr bool eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<std::decay_t<T>>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_eigen_triangular_expr : std::false_type {};

      template<typename NestedMatrix, TriangleType triangle_type>
      struct is_eigen_triangular_expr<TriangularMatrix<NestedMatrix, triangle_type>> : std::true_type {};
    }


    /**
     * A triangular matrix based on the Eigen library.
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_triangular_expr = detail::is_eigen_triangular_expr<std::decay_t<T>>::value;
#else
    inline constexpr bool eigen_triangular_expr = detail::is_eigen_triangular_expr<std::decay_t<T>>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_eigen_diagonal_expr : std::false_type {};

      template<typename NestedMatrix>
      struct is_eigen_diagonal_expr<DiagonalMatrix<NestedMatrix>> : std::true_type {};
    }


    /**
     * A diagonal matrix based on the Eigen library.
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_diagonal_expr = detail::is_eigen_diagonal_expr<std::decay_t<T>>::value;
#else
    inline constexpr bool eigen_diagonal_expr = detail::is_eigen_diagonal_expr<std::decay_t<T>>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_eigen_zero_expr : std::false_type {};

      template<typename NestedMatrix>
      struct is_eigen_zero_expr<ZeroMatrix<NestedMatrix>> : std::true_type {};
    }


    /**
     * A zero matrix based on the Eigen library. (All coefficients are zero.)
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_zero_expr = detail::is_eigen_zero_expr<std::decay_t<T>>::value;
#else
    inline constexpr bool eigen_zero_expr = detail::is_eigen_zero_expr<std::decay_t<T>>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_from_euclidean_expr : std::false_type {};

      template<typename Coefficients, typename NestedMatrix>
      struct is_from_euclidean_expr<FromEuclideanExpr<Coefficients, NestedMatrix>> : std::true_type {};
    }


    /**
     * An expression converting each column vector in a Euclidean Eigen matrix from Euclidean space.
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept from_euclidean_expr = detail::is_from_euclidean_expr<std::decay_t<T>>::value;
#else
    inline constexpr bool from_euclidean_expr = detail::is_from_euclidean_expr<std::decay_t<T>>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_to_euclidean_expr : std::false_type {};

      template<typename Coefficients, typename NestedMatrix>
      struct is_to_euclidean_expr<ToEuclideanExpr<Coefficients, NestedMatrix>> : std::true_type {};
    }


    /**
     * An expression converting each column vector in an Eigen matrix to Euclidean space.
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept to_euclidean_expr = detail::is_to_euclidean_expr<std::decay_t<T>>::value;
#else
    inline constexpr bool to_euclidean_expr = detail::is_to_euclidean_expr<std::decay_t<T>>::value;
#endif


    /**
     * Either from_euclidean_expr or to_euclidean_expr.
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept euclidean_expr = from_euclidean_expr<T> or to_euclidean_expr<T>;
#else
    inline constexpr bool euclidean_expr = from_euclidean_expr<T> or to_euclidean_expr<T>;
#endif


    /**
     * T is either a native Eigen matrix or a zero Eigen matrix.
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept eigen_matrix = eigen_native<T> or eigen_zero_expr<T>;
#else
    inline constexpr bool eigen_matrix = eigen_native<T> or eigen_zero_expr<T>;
#endif


    namespace detail
    {
      template<typename T>
      struct is_upper_storage_triangle : std::false_type {};

      template<typename NestedMatrix>
      struct is_upper_storage_triangle<Eigen3::SelfAdjointMatrix<NestedMatrix, TriangleType::upper>> : std::true_type {};
    }


    /**
     * A self-adjoint matrix that stores data in the upper triangle.
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept upper_storage_triangle = detail::is_upper_storage_triangle<std::decay_t<T>>::value;
#else
    inline constexpr bool upper_storage_triangle = detail::is_upper_storage_triangle<std::decay_t<T>>::value;
#endif


    namespace detail
    {
      template<typename T>
      struct is_lower_storage_triangle : std::false_type {};

      template<typename NestedMatrix>
      struct is_lower_storage_triangle<Eigen3::SelfAdjointMatrix<NestedMatrix, TriangleType::lower>> : std::true_type {};
    }


    /**
     * A self-adjoint matrix that stores data in the lower triangle.
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept lower_storage_triangle = detail::is_lower_storage_triangle<std::decay_t<T>>::value;
#else
    inline constexpr bool lower_storage_triangle = detail::is_lower_storage_triangle<std::decay_t<T>>::value;
#endif

  } // namespace Eigen3


  namespace internal
  {
    // Defines the is_covariance_nestable type trait specifically for Eigen3.
#ifdef __cpp_concepts
    template<typename T>
    requires
      Eigen3::eigen_self_adjoint_expr<T> or
      Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or
      Eigen3::eigen_zero_expr<T> or
      (Eigen3::eigen_native<T> and (triangular_matrix<T> or self_adjoint_matrix<T>))
    struct is_covariance_nestable<T>
#else
    template<typename T>
    struct is_covariance_nestable<T, std::enable_if_t<
      Eigen3::eigen_self_adjoint_expr<T> or
      Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or
      Eigen3::eigen_zero_expr<T> or
      (Eigen3::eigen_native<T> and (triangular_matrix<T> or self_adjoint_matrix<T>))>>
#endif
    : std::true_type {};


    // Defines the is_typed_matrix_nestable type trait specifically for Eigen3.
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
    struct is_typed_matrix_nestable<T>
#else
    template<typename T>
    struct is_typed_matrix_nestable<T, std::enable_if_t<
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

  namespace internal
  {
#ifdef __cpp_concepts
    template<Eigen3::eigen_native T>
    struct is_element_gettable<T, 2>
#else
    template<typename T>
    struct is_element_gettable<T, 2, std::enable_if_t<Eigen3::eigen_native<T>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<Eigen3::eigen_native T> requires (MatrixTraits<T>::columns == 1)
    struct is_element_gettable<T, 1>
#else
    template<typename T>
    struct is_element_gettable<T, 1, std::enable_if_t<Eigen3::eigen_native<T> and (MatrixTraits<T>::columns == 1)>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<Eigen3::eigen_native T> requires (not std::is_const_v<std::remove_reference_t<T>>) and
      (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))
    struct is_element_settable<T, 2>
#else
    template<typename T>
    struct is_element_settable<T, 2, std::enable_if_t<
      Eigen3::eigen_native<T> and (not std::is_const_v<std::remove_reference_t<T>>) and
      (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<Eigen3::eigen_native T> requires (MatrixTraits<T>::columns == 1) and
      (not std::is_const_v<std::remove_reference_t<T>>) and
      (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))
    struct is_element_settable<T, 1>
#else
    template<typename T>
    struct is_element_settable<T, 1, std::enable_if_t<Eigen3::eigen_native<T> and (MatrixTraits<T>::columns == 1) and
      (not std::is_const_v<std::remove_reference_t<T>>) and
      (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))>>
#endif
      : std::true_type {};

  } // namespace internal


  // ================================ //
  //   Type traits for Eigen3 types   //
  // ================================ //

  namespace internal
  {
    // ---------------- //
    //  is_zero_matrix  //
    // ---------------- //

#ifdef __cpp_concepts
    template<typename T> requires (Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T>) and zero_matrix<nested_matrix_t<T>>
    struct is_zero_matrix<T>
      : std::true_type {};
#else
    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T>>>
      : is_zero_matrix<nested_matrix_t<T>> {};
#endif


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
    template<typename T> requires (Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T>) and
      identity_matrix<nested_matrix_t<T>>
    struct is_identity_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<(Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T>)>>
      : is_identity_matrix<nested_matrix_t<T>> {};
#endif


#ifdef __cpp_concepts
    template<typename Arg> requires (Arg::RowsAtCompileTime == Arg::ColsAtCompileTime)
    struct is_identity_matrix<Eigen3::IdentityMatrix<Arg>> : std::true_type {};
#else
    template<typename Arg>
    struct is_identity_matrix<Eigen3::IdentityMatrix<Arg>>
      : std::bool_constant<Arg::RowsAtCompileTime == Arg::ColsAtCompileTime> {};
#endif


    // The product of two identity matrices is also identity.
#ifdef __cpp_concepts
    template<identity_matrix Arg1, identity_matrix Arg2>
    struct is_identity_matrix<Eigen::Product<Arg1, Arg2>> : std::true_type {};
#else
    template<typename Arg1, typename Arg2>
    struct is_identity_matrix<Eigen::Product<Arg1, Arg2>>
      : std::bool_constant<identity_matrix<Arg1> and identity_matrix<Arg2>> {};
#endif


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
    template<Eigen3::eigen_self_adjoint_expr T> requires diagonal_matrix<nested_matrix_t<T>> or
      (MatrixTraits<T>::storage_type == TriangleType::diagonal)
    struct is_diagonal_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T>>>
      : std::bool_constant<diagonal_matrix<nested_matrix_t<T>> or
          MatrixTraits<T>::storage_type == TriangleType::diagonal> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::eigen_triangular_expr T> requires diagonal_matrix<nested_matrix_t<T>> or
      (MatrixTraits<T>::triangle_type == TriangleType::diagonal)
    struct is_diagonal_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<Eigen3::eigen_triangular_expr<T>>>
      : std::bool_constant<diagonal_matrix<nested_matrix_t<T>> or
          MatrixTraits<T>::triangle_type == TriangleType::diagonal> {};
#endif


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
    struct is_lower_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_lower_triangular_matrix<T, std::enable_if_t<Eigen3::eigen_triangular_expr<T>>>
      : std::bool_constant<MatrixTraits<T>::triangle_type == TriangleType::lower> {};
#endif


    // ---------------------------- //
    //  is_upper_triangular_matrix  //
    // ---------------------------- //

#ifdef __cpp_concepts
    template<Eigen3::eigen_triangular_expr T> requires (MatrixTraits<T>::triangle_type == TriangleType::upper)
    struct is_upper_triangular_matrix<T> : std::true_type {};
#else
    template<typename T>
    struct is_upper_triangular_matrix<T, std::enable_if_t<Eigen3::eigen_triangular_expr<T>>>
      : std::bool_constant<MatrixTraits<T>::triangle_type == TriangleType::upper> {};
#endif


    // ------------------- //
    //  is_self_contained  //
    // ------------------- //

#ifdef __cpp_concepts
    template<typename T> requires (Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or Eigen3::to_euclidean_expr<T> or Eigen3::from_euclidean_expr<T>) and
      self_contained<nested_matrix_t<T>> and
      (not std::is_reference_v<nested_matrix_t<T>>)
    struct is_self_contained<T> : std::true_type {};
#else
    template<typename T>
    struct is_self_contained<T, std::enable_if_t<
      Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or Eigen3::to_euclidean_expr<T> or Eigen3::from_euclidean_expr<T>>>
      : std::bool_constant<self_contained<nested_matrix_t<T>> and
          (not std::is_reference_v<nested_matrix_t<T>>)> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::eigen_zero_expr T>
    struct is_self_contained<T>
#else
    template<typename T>
    struct is_self_contained<T, std::enable_if_t<Eigen3::eigen_zero_expr<T>>>
#endif
      : std::true_type {};


    // --------------------- //
    //  is_element_gettable  //
    // --------------------- //

#ifdef __cpp_concepts
    template<Eigen3::eigen_self_adjoint_expr T, std::size_t N> requires
      (element_gettable<nested_matrix_t<T>, 2> and
        (N == 2 or MatrixTraits<T>::storage_type == TriangleType::diagonal)) or
      element_gettable<nested_matrix_t<T>, 1>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T>>>
      : std::bool_constant<(element_gettable<nested_matrix_t<T>, 2> and
          (N == 2 or MatrixTraits<T>::storage_type == TriangleType::diagonal)) or
          element_gettable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::eigen_triangular_expr T, std::size_t N> requires
      (element_gettable<nested_matrix_t<T>, 2> and
        (N == 2 or MatrixTraits<T>::triangle_type == TriangleType::diagonal)) or
      element_gettable<nested_matrix_t<T>, 1>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_triangular_expr<T>>>
      : std::bool_constant<(element_gettable<nested_matrix_t<T>, 2> and
          (N == 2 or MatrixTraits<T>::triangle_type == TriangleType::diagonal)) or
          element_gettable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::eigen_diagonal_expr T, std::size_t N> requires (N <= 2) and
      (element_gettable<nested_matrix_t<T>, 2> or
        element_gettable<nested_matrix_t<T>, 1>)
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_diagonal_expr<T> and (N <= 2)>>
      : std::bool_constant<element_gettable<nested_matrix_t<T>, 2> or
          element_gettable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::eigen_zero_expr T, std::size_t N> requires (N == 2) or (N == 1 and MatrixTraits<T>::columns == 1)
    struct is_element_gettable<T, N>
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_zero_expr<T> and
      ((N == 2) or (N == 1 and MatrixTraits<T>::columns == 1))>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<Eigen3::to_euclidean_expr T, std::size_t N> requires
      element_gettable<nested_matrix_t<T>, N>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<Eigen3::to_euclidean_expr<T>>>
      : is_element_gettable<nested_matrix_t<T>, N> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::from_euclidean_expr T, std::size_t N> requires
      element_gettable<nested_matrix_t<T>, N>
    struct is_element_gettable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_gettable<T, N, std::enable_if_t<Eigen3::from_euclidean_expr<T>>>
      : is_element_gettable<nested_matrix_t<T>, N> {};
#endif


    // --------------------- //
    //  is_element_settable  //
    // --------------------- //

#ifdef __cpp_concepts
    template<Eigen3::eigen_self_adjoint_expr T, std::size_t N> requires
      (element_settable<nested_matrix_t<T>, 2> and
        (N == 2 or MatrixTraits<T>::storage_type == TriangleType::diagonal)) or
      element_settable<nested_matrix_t<T>, 1>
    struct is_element_settable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T>>>
      : std::bool_constant<(element_settable<nested_matrix_t<T>, 2> and
          (N == 2 or MatrixTraits<T>::storage_type == TriangleType::diagonal)) or
          element_settable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::eigen_triangular_expr T, std::size_t N> requires
      (element_settable<nested_matrix_t<T>, 2> and
        (N == 2 or MatrixTraits<T>::triangle_type == TriangleType::diagonal)) or
      element_settable<nested_matrix_t<T>, 1>
    struct is_element_settable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_triangular_expr<T>>>
      : std::bool_constant<(element_settable<nested_matrix_t<T>, 2> and
          (N == 2 or MatrixTraits<T>::triangle_type == TriangleType::diagonal)) or
          element_settable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::eigen_diagonal_expr T, std::size_t N> requires (N <= 2) and
      (element_settable<nested_matrix_t<T>, 2> or
        element_settable<nested_matrix_t<T>, 1>)
    struct is_element_settable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_diagonal_expr<T> and (N <= 2)>>
      : std::bool_constant<element_settable<nested_matrix_t<T>, 2> or
          element_settable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::eigen_zero_expr T, std::size_t N>
    struct is_element_settable<T, N>
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_zero_expr<T>>>
#endif
      : std::false_type {};


#ifdef __cpp_concepts
    template<Eigen3::to_euclidean_expr T, std::size_t N> requires
      MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_t<T>, N>
    struct is_element_settable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<Eigen3::to_euclidean_expr<T> and
      MatrixTraits<T>::RowCoefficients::axes_only>>
      : is_element_settable<nested_matrix_t<T>, N> {};
#endif


#ifdef __cpp_concepts
    template<Eigen3::from_euclidean_expr T, std::size_t N> requires
      (MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_t<T>, N>) or
      (Eigen3::to_euclidean_expr<nested_matrix_t<T>> and
        element_settable<nested_matrix_t<nested_matrix_t<T>>, N>)
    struct is_element_settable<T, N> : std::true_type {};
#else
    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<Eigen3::from_euclidean_expr<T> and
      (not Eigen3::to_euclidean_expr<nested_matrix_t<T>>)>>
      : std::bool_constant<MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_t<T>, N>> {};

    template<typename T, std::size_t N>
    struct is_element_settable<T, N, std::enable_if_t<Eigen3::from_euclidean_expr<T> and
      Eigen3::to_euclidean_expr<nested_matrix_t<T>>>>
      : std::bool_constant<element_settable<nested_matrix_t<nested_matrix_t<T>>, N>> {};
#endif


  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
