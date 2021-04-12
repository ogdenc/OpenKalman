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
 * \brief Forward declarations for OpenKalman's Eigen3 interface.
 */

#ifndef OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP

#include <type_traits>


/**
 * \namespace OpenKalman::Eigen3
 * \brief Namespace for all Eigen3 interface definitions.
 *
 * \internal
 * \namespace OpenKalman::Eigen3::internal
 * \brief Namespace for definitions internal to the Eigen3 interface library.
 *
 * \namespace Eigen
 * \brief Eigen3's native namespace.
 *
 * \namespace Eigen::internal
 * \brief Eigen3's native namespace for internal definitions.
 */


namespace OpenKalman::Eigen3
{
  namespace internal
  {
    /**
     * \internal
     * \brief The ultimate base for matrix classes in OpenKalman.
     * \details This class is used solely to distinguish OpenKalman classes from native Eigen classes which are
     * also derived from Eigen::MatrixBase.
     */
    template<typename Derived>
    struct Eigen3Base : Eigen::MatrixBase<Derived> {};


    /*
     * \internal
     * \brief Base for matrix classes in OpenKalman.
     * \details This specializes the comma initializer for OpenKalman classes, and redefines the Zero and Identity
     * functions.
     */
    template<typename Derived, typename Nested>
    struct Eigen3MatrixBase;
  } // namespace internal


  /**
   * \brief Specifies a native Eigen3 matrix deriving from Eigen::MatrixBase.
   * \details This includes any original class in the Eigen library descending from Eigen::MatrixBase.
   * It does not include new classes added in OpenKalman, such as DiagonalMatrix or ZeroMatrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_native = std::derived_from<std::decay_t<T>, Eigen::MatrixBase<std::decay_t<T>>> and
    (not std::derived_from<std::decay_t<T>, internal::Eigen3Base<std::decay_t<T>>>);
#else
  inline constexpr bool eigen_native = std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> and
      (not std::is_base_of_v<internal::Eigen3Base<std::decay_t<T>>, std::decay_t<T>>);
#endif


  /**
   * \brief An alias for the Eigen identity matrix.
   * \details In Eigen, this does not need to be a \ref square_matrix.
   * \tparam NestedMatrix The nested matrix on which the identity is based.
   */
  template<typename NestedMatrix>
  using IdentityMatrix =
    Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<typename NestedMatrix::Scalar>, NestedMatrix>;


  // ------------------------------------------- //
  //  ZeroMatrix, eigen_zero_expr, eigen_matrix  //
  // ------------------------------------------- //

  /**
   * \brief A matrix in which all elements are automatically 0.
   * \note This is necessary because Eigen3 types do not distinguish between a zero matrix and a constant matrix.
   * \tparam Scalar The scalar type.
   * \tparam rows The number of rows.
   * \tparam columns The number of columns.
   */
  template<typename Scalar, std::size_t rows, std::size_t columns = 1>
  struct ZeroMatrix;


  namespace detail
  {
    template<typename T>
    struct is_eigen_zero_expr : std::false_type {};

    template<typename Scalar, std::size_t rows, std::size_t cols>
    struct is_eigen_zero_expr<ZeroMatrix<Scalar, rows, cols>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a zero matrix based on the Eigen library (i.e., ZeroMatrix).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_zero_expr = detail::is_eigen_zero_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool eigen_zero_expr = detail::is_eigen_zero_expr<std::decay_t<T>>::value;
#endif


  /**
   * \brief Specifies that T is a suitable nested matrix for OpenKalman's new Eigen matrix classes.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_matrix = eigen_native<T> or eigen_zero_expr<T>;
#else
  inline constexpr bool eigen_matrix = eigen_native<T> or eigen_zero_expr<T>;
#endif


  // ------------------------------------- //
  //  DiagonalMatrix, eigen_diagonal_expr  //
  // ------------------------------------- //

  /**
   * \brief A diagonal matrix.
   * \details The matrix is guaranteed to be diagonal. It is ::self_contained iff NestedMatrix is ::self_contained.
   * Implicit conversions are available from any \ref diagonal_matrix of compatible size.
   * \tparam NestedMatrix A \ref column_vector expression defining the diagonal elements.
   * Elements outside the diagonal are automatically 0.
   * \note This has the same name as Eigen::DiagonalMatrix, and is intended as a replacement.
   */
#ifdef __cpp_concepts
  template<column_vector NestedMatrix> requires eigen_matrix<NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalMatrix;


  namespace detail
  {
    template<typename T>
    struct is_eigen_diagonal_expr : std::false_type {};

    template<typename NestedMatrix>
    struct is_eigen_diagonal_expr<DiagonalMatrix<NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a diagonal matrix based on the Eigen library (i.e., DiaginalMatrix).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_diagonal_expr = detail::is_eigen_diagonal_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool eigen_diagonal_expr = detail::is_eigen_diagonal_expr<std::decay_t<T>>::value;
#endif


  // -------------------------------------------- //
  //  SelfAdjointMatrix, eigen_self_adjoint_expr  //
  // -------------------------------------------- //

  /**
   * \brief A self-adjoint matrix.
   * \details The matrix is guaranteed to be self-adjoint. It is ::self_contained iff NestedMatrix is ::self_contained.
   * It may \em also be a diagonal matrix if storage_triangle is TriangleType::diagonal.
   * Implicit conversions are available from any \ref self_adjoint_matrix of compatible size.
   * \tparam NestedMatrix A nested \ref square_matrix expression, on which the self-adjoint matrix is based.
   * \tparam storage_triangle The TriangleType (\ref TriangleType::lower "lower", \ref TriangleType::upper "upper", or
   * \ref TriangleType::diagonal "diagonal") in which the data is stored.
   * Matrix elements outside this triangle/diagonal are ignored. If the matrix is lower or upper triangular,
   * elements are mapped from this selected triangle to the elements in the other triangle to ensure that the matrix
   * is self-adjoint. If the matrix is diagonal, 0 is automatically mapped to each matrix element outside the diagonal.
   */
#ifdef __cpp_concepts
  template<square_matrix NestedMatrix, TriangleType storage_triangle = TriangleType::lower> requires
    eigen_matrix<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>
#else
  template<typename NestedMatrix, TriangleType storage_triangle = TriangleType::lower>
#endif
  struct SelfAdjointMatrix;


  namespace detail
  {
    template<typename T>
    struct is_eigen_self_adjoint_expr : std::false_type {};

    template<typename NestedMatrix, TriangleType storage_triangle>
    struct is_eigen_self_adjoint_expr<SelfAdjointMatrix<NestedMatrix, storage_triangle>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a self-adjoint matrix based on the Eigen library (i.e., SelfAdjointMatrix).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<std::decay_t<T>>::value;
#endif


  // ---------------------------------------------------------------------------------------------- //
  //  is_upper_triangular_storage, is_lower_triangular_storage, internal::same_storage_triangle_as  //
  // ---------------------------------------------------------------------------------------------- //

  namespace internal
  {
    template<typename T>
    struct is_upper_triangular_storage : std::false_type {};

    template<typename T>
    struct is_lower_triangular_storage : std::false_type {};
  }


  /**
   * \brief Specifies that T is an \ref eigen_self_adjoint_expr that stores data in the upper-right triangle.
   * \details This \em includes matrices that store data only along the diagonal.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept upper_triangular_storage = internal::is_upper_triangular_storage<std::decay_t<T>>::value;
#else
  inline constexpr bool upper_triangular_storage = internal::is_upper_triangular_storage<std::decay_t<T>>::value;
#endif


  /**
   * \brief Specifies that T is an \ref eigen_self_adjoint_expr that stores data in the lower-left triangle.
   * \details This \em includes matrices that store data only along the diagonal.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept lower_triangular_storage = internal::is_lower_triangular_storage<std::decay_t<T>>::value;
#else
  inline constexpr bool lower_triangular_storage = internal::is_lower_triangular_storage<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * \brief Specifies that two self-adjoint expressions have the same storage triangle type (upper or lower).
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T, typename U>
#ifdef __cpp_concepts
    concept same_storage_triangle_as =
    (upper_triangular_storage<T> and upper_triangular_storage<U>) or
      (lower_triangular_storage<T> and lower_triangular_storage<U>);
#else
    inline constexpr bool same_storage_triangle_as =
        (upper_triangular_storage<T> and upper_triangular_storage<U>) or
        (lower_triangular_storage<T> and lower_triangular_storage<U>);
#endif
  }


  // ----------------------------------------- //
  //  TriangularMatrix, eigen_triangular_expr  //
  // ----------------------------------------- //

  /**
   * \brief A triangular matrix.
   * \details The matrix is guaranteed to be triangular. It is ::self_contained iff NestedMatrix is ::self_contained.
   * It may \em also be a diagonal matrix if triangle_type is TriangleType::diagonal.
   * Implicit conversions are available from any \ref triangular_matrix of compatible size.
   * \tparam NestedMatrix A nested \ref square_matrix expression, on which the triangular matrix is based.
   * \tparam triangle_type The TriangleType (\ref TriangleType::lower "lower", \ref TriangleType::upper "upper", or
   * \ref TriangleType::diagonal "diagonal") in which the data is stored.
   * Matrix elements outside this triangle/diagonal are ignored. Instead, 0 is automatically mapped to each element
   * not within the selected triangle or diagonal, to ensure that the matrix is triangular.
   */
#ifdef __cpp_concepts
  template<square_matrix NestedMatrix, TriangleType triangle_type = TriangleType::lower> requires
    eigen_matrix<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>
#else
  template<typename NestedMatrix, TriangleType triangle_type = TriangleType::lower>
#endif
  struct TriangularMatrix;


  namespace detail
  {
    template<typename T>
    struct is_eigen_triangular_expr : std::false_type {};

    template<typename NestedMatrix, TriangleType triangle_type>
    struct is_eigen_triangular_expr<TriangularMatrix<NestedMatrix, triangle_type>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a triangular matrix based on the Eigen library (i.e., TriangularMatrix).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_triangular_expr = detail::is_eigen_triangular_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool eigen_triangular_expr = detail::is_eigen_triangular_expr<std::decay_t<T>>::value;
#endif


  // ------------------------------------ //
  //  ToEuclideanExpr, to_euclidean_expr  //
  // ------------------------------------ //

  /**
   * \brief An expression that transforms coefficients into Euclidean space for proper wrapping.
   * \details This is the counterpart expression to FromEuclideanExpr.
   * \tparam Coeffs The coefficient types.
   * \tparam NestedMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, eigen_matrix NestedMatrix = Eigen::Matrix<double, Coeffs::dimensions, 1>> requires
    (MatrixTraits<NestedMatrix>::rows == Coeffs::dimensions)
#else
  template<typename Coeffs, typename NestedMatrix = Eigen::Matrix<double, Coeffs::dimensions, 1>>
#endif
  struct ToEuclideanExpr;


  namespace detail
  {
    template<typename T>
    struct is_to_euclidean_expr : std::false_type {};

    template<typename Coefficients, typename NestedMatrix>
    struct is_to_euclidean_expr<ToEuclideanExpr<Coefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is an expression converting coefficients to Euclidean space (i.e., ToEuclideanExpr).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept to_euclidean_expr = detail::is_to_euclidean_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool to_euclidean_expr = detail::is_to_euclidean_expr<std::decay_t<T>>::value;
#endif


  // -------------------------------------------------------- //
  //  FromEuclideanExpr, from_euclidean_expr, euclidean_expr  //
  // -------------------------------------------------------- //

  /**
   * \brief An expression that transforms angular or other modular coefficients back from Euclidean space.
   * \details This is the counterpart expression to ToEuclideanExpr.
   * \tparam Coeffs The coefficient types.
   * \tparam NestedMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, typename NestedMatrix = Eigen::Matrix<double, Coeffs::euclidean_dimensions, 1>> requires
    (eigen_matrix<NestedMatrix> or to_euclidean_expr<NestedMatrix>) and
    (MatrixTraits<NestedMatrix>::rows == Coeffs::euclidean_dimensions)
#else
  template<typename Coeffs, typename NestedMatrix = Eigen::Matrix<double, Coeffs::euclidean_dimensions, 1>>
#endif
  struct FromEuclideanExpr;


  namespace detail
  {
    template<typename T>
    struct is_from_euclidean_expr : std::false_type {};

    template<typename Coefficients, typename NestedMatrix>
    struct is_from_euclidean_expr<FromEuclideanExpr<Coefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is an expression converting coefficients from Euclidean space (i.e., FromEuclideanExpr).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept from_euclidean_expr = detail::is_from_euclidean_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool from_euclidean_expr = detail::is_from_euclidean_expr<std::decay_t<T>>::value;
#endif


  /**
   * \brief Specifies that T is either \ref to_euclidean_expr or \ref from_euclidean_expr.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept euclidean_expr = to_euclidean_expr<T> or from_euclidean_expr<T>;
#else
  inline constexpr bool euclidean_expr = from_euclidean_expr<T> or to_euclidean_expr<T>;
#endif


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
