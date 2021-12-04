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
     * \details This class is used mainly to distinguish OpenKalman classes from native Eigen classes which are
     * also derived from Eigen::MatrixBase.
     */
    template<typename Derived>
    struct Eigen3Base;


    /**
     * \internal
     * \brief Base for matrix classes in OpenKalman.
     * \details This specializes the comma initializer for OpenKalman classes, and redefines the Zero and Identity
     * functions.
     */
    template<typename Derived, typename Nested>
    struct Eigen3MatrixBase;


    /**
     * \internal
     * \brief Base class for library-defined dynamic Eigen matrices.
     */
#ifdef __cpp_concepts
    template<typename Scalar, std::size_t rows, std::size_t columns>
#else
    template<typename Scalar, std::size_t rows, std::size_t columns, typename = void>
#endif
    struct EigenDynamicBase;

  } // namespace internal


  namespace internal
  {
    /**
     * \internal
     * \brief A type trait that must be instantiated for each native Eigen class deriving from Eigen::MatrixBase.
     * \tparam T
     */
    template<typename T>
    struct is_native_eigen_matrix : std::false_type {};
  }


  /**
   * \brief Specifies a native Eigen3 matrix or expression class deriving from Eigen::MatrixBase.
   * \details This includes any original class in the Eigen library descending from Eigen::DenseBase.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept native_eigen_matrix = internal::is_native_eigen_matrix<std::decay_t<T>>::value;
    // or (std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> and
    //(not std::is_base_of_v<internal::Eigen3Base<std::decay_t<T>>, std::decay_t<T>>));
#else
  inline constexpr bool native_eigen_matrix = internal::is_native_eigen_matrix<std::decay_t<T>>::value;
    // or (std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> and
    //(not std::is_base_of_v<internal::Eigen3Base<std::decay_t<T>>, std::decay_t<T>>));
#endif


  namespace internal
  {
    /**
     * \internal
     * \brief A type trait that must be instantiated for each native Eigen class that is convertible to Eigen::Matrix.
     * \tparam T
     */
    template<typename T>
    struct is_convertible_to_native_eigen_matrix : std::false_type {};
  }


  /**
   * \brief Specifies a native Eigen3 class that can be converted to Eigen::Matrix.
   * \details This should include any class in the Eigen library descending from Eigen::EigenBase.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept convertible_to_native_eigen_matrix =
    std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>> and
    requires(const std::decay_t<T>& t) {
      Eigen::Matrix<typename Eigen::internal::traits<T>::Scalar, Eigen::internal::traits<T>::RowsAtCompileTime,
            Eigen::internal::traits<T>::ColsAtCompileTime> {t};
    };
#else
  inline constexpr bool convertible_to_native_eigen_matrix =
    std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>> and
    std::is_constructible_v<Eigen::Matrix<typename Eigen::internal::traits<T>::Scalar,
      Eigen::internal::traits<T>::RowsAtCompileTime, Eigen::internal::traits<T>::ColsAtCompileTime>,
      const std::decay_t<T>&>;
#endif


  /**
   * \brief An alias for the Eigen identity matrix.
   * \note In Eigen, this does not need to be a \ref square_matrix.
   * \tparam NestedMatrix The nested matrix on which the identity is based.
   */
  template<typename NestedMatrix>
  using IdentityMatrix =
    Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<typename NestedMatrix::Scalar>, NestedMatrix>;


  // ------------------------------------- //
  //  ConstantMatrix, eigen_constant_expr  //
  // ------------------------------------- //

  /**
   * \brief A matrix in which all elements are a constant at compile time.
   * \note This is necessary because Eigen3 types do not provide for compile-time constant matrices.
   * \tparam Scalar The scalar type.
   * \tparam rows The number of rows (0 if \ref dynamic_rows).
   * \tparam columns The number of columns (0 if \ref dynamic_columns).
   * \todo Demote Scalar to an optional parameter defaulting to decltype(auto), and possibly remove it conditionally with "__cpp_nontype_template_args >= 201911L"
   */
#ifdef __cpp_concepts
  template<arithmetic_or_complex Scalar, auto constant, std::size_t rows, std::size_t columns = 1>
  requires std::convertible_to<decltype(constant), Scalar>
#else
  template<typename Scalar, auto constant, std::size_t rows, std::size_t columns = 1>
#endif
  struct ConstantMatrix;


  namespace detail
  {
    template<typename T>
    struct is_eigen_constant_expr : std::false_type {};

    template<typename Scalar, auto constant, std::size_t rows, std::size_t cols>
    struct is_eigen_constant_expr<ConstantMatrix<Scalar, constant, rows, cols>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a constant matrix based on the Eigen library (i.e., ConstantMatrix).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_constant_expr = detail::is_eigen_constant_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool eigen_constant_expr = detail::is_eigen_constant_expr<std::decay_t<T>>::value;
#endif


  // ----------------------------- //
  //  ZeroMatrix, eigen_zero_expr  //
  // ----------------------------- //

  /**
   * \brief A matrix in which all elements are automatically 0.
   * \note This is necessary because Eigen3 types do not distinguish between a zero matrix and a constant matrix.
   * \tparam Scalar The scalar type.
   * \tparam rows The number of rows (0 if \ref dynamic_rows).
   * \tparam columns The number of columns (0 if \ref dynamic_columns).
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
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_zero_expr = detail::is_eigen_zero_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool eigen_zero_expr = detail::is_eigen_zero_expr<std::decay_t<T>>::value;
#endif


  // -------------- //
  //  eigen_matrix  //
  // -------------- //

  /**
   * \brief Specifies that T is a suitable nested matrix for OpenKalman's new Eigen matrix classes.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_matrix = native_eigen_matrix<T> or eigen_zero_expr<T> or eigen_constant_expr<T>;
#else
  inline constexpr bool eigen_matrix = native_eigen_matrix<T> or eigen_zero_expr<T> or eigen_constant_expr<T>;
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
  template<eigen_matrix NestedMatrix> requires dynamic_columns<NestedMatrix> or column_vector<NestedMatrix>
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
  template<typename NestedMatrix, TriangleType storage_triangle =
      (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal : TriangleType::lower)>
  requires (eigen_diagonal_expr<NestedMatrix> and not complex_number<typename MatrixTraits<NestedMatrix>::Scalar>) or
    (eigen_matrix<NestedMatrix>  and (dynamic_shape<NestedMatrix> or square_matrix<NestedMatrix>))
#else
  template<typename NestedMatrix, TriangleType storage_triangle =
    (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal : TriangleType::lower)>
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
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<std::decay_t<T>>::value;
#endif


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
  template<typename NestedMatrix, TriangleType triangle_type = (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal :
      (upper_triangular_matrix<NestedMatrix> ? TriangleType::upper : TriangleType::lower))>
  requires eigen_diagonal_expr<NestedMatrix> or
    (eigen_matrix<NestedMatrix> and (dynamic_shape<NestedMatrix> or square_matrix<NestedMatrix>))
#else
  template<typename NestedMatrix, TriangleType triangle_type = (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal :
    (upper_triangular_matrix<NestedMatrix> ? TriangleType::upper : TriangleType::lower))>
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
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_triangular_expr = detail::is_eigen_triangular_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool eigen_triangular_expr = detail::is_eigen_triangular_expr<std::decay_t<T>>::value;
#endif


  // ---------------- //
  //  eigen_matrix_t  //
  // ---------------- //

  /**
   * \brief An alias for a self-contained, writable, native Eigen matrix.
   * \tparam Scalar Scalar type of the matrix (defaults to the Scalar type of T).
   * \tparam rows Number of rows in the native matrix (0 if not fixed at compile time).
   * \tparam cols Number of columns in the native matrix (0 if not fixed at compile time).
   */
  template<typename Scalar, std::size_t rows, std::size_t columns = 1>
  using eigen_matrix_t = Eigen::Matrix<Scalar, rows == 0 ? Eigen::Dynamic : (Eigen::Index) rows,
    columns == 0 ? Eigen::Dynamic : (Eigen::Index) columns>;


  // ------------------------------------ //
  //  ToEuclideanExpr, to_euclidean_expr  //
  // ------------------------------------ //

  /**
   * \brief An expression that transforms coefficients into Euclidean space for proper wrapping.
   * \details This is the counterpart expression to FromEuclideanExpr.
   * \tparam Coefficients The coefficient types.
   * \tparam NestedMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename NestedMatrix = eigen_matrix_t<double, Coefficients::dimensions, 1>>
  requires (eigen_matrix<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>) and
    (dynamic_coefficients<Coefficients> == dynamic_rows<NestedMatrix>) and
    (not fixed_coefficients<Coefficients> or Coefficients::dimensions == MatrixTraits<NestedMatrix>::rows) and
    (not dynamic_coefficients<Coefficients> or
      std::same_as<typename Coefficients::Scalar, typename MatrixTraits<NestedMatrix>::Scalar>)
#else
  template<typename Coefficients, typename NestedMatrix = Eigen3::eigen_matrix_t<double, Coefficients::dimensions, 1>>
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
   * \tparam Coefficients The coefficient types.
   * \tparam NestedMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients,
    typename NestedMatrix = eigen_matrix_t<double, Coefficients::euclidean_dimensions, 1>>
  requires (eigen_matrix<NestedMatrix> or to_euclidean_expr<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>) and
    (dynamic_coefficients<Coefficients> == dynamic_rows<NestedMatrix>) and
    (not fixed_coefficients<Coefficients> or Coefficients::euclidean_dimensions == MatrixTraits<NestedMatrix>::rows) and
    (not dynamic_coefficients<Coefficients> or
      std::same_as<typename Coefficients::Scalar, typename MatrixTraits<NestedMatrix>::Scalar>)
#else
  template<typename Coefficients,
    typename NestedMatrix = Eigen3::eigen_matrix_t<double, Coefficients::euclidean_dimensions, 1>>
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
   */
  template<typename T>
#ifdef __cpp_concepts
  concept from_euclidean_expr = detail::is_from_euclidean_expr<std::decay_t<T>>::value;
#else
  inline constexpr bool from_euclidean_expr = detail::is_from_euclidean_expr<std::decay_t<T>>::value;
#endif


  /**
   * \brief Specifies that T is either \ref to_euclidean_expr or \ref from_euclidean_expr.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept euclidean_expr = to_euclidean_expr<T> or from_euclidean_expr<T>;
#else
  inline constexpr bool euclidean_expr = from_euclidean_expr<T> or to_euclidean_expr<T>;
#endif


} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
