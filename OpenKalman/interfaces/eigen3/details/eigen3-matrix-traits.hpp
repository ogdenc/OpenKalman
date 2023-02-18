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
 * \brief Concepts as applied to native Eigen3 matrix classes.
 */

#ifndef OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP
#define OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP

#include <type_traits>


namespace OpenKalman::Eigen3
{
  // ------------------------------------------------------------------------------------------- //
  //   eigen_SelfAdjointView, eigen_TriangularView, eigen_DiagonalMatrix, eigen_DiagonalWrapper  //
  // ------------------------------------------------------------------------------------------- //

  namespace detail
  {
    template<typename T>
    struct is_eigen_SelfAdjointView : std::false_type {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_eigen_SelfAdjointView<Eigen::SelfAdjointView<MatrixType, UpLo>> : std::true_type {};


    template<typename T>
    struct is_eigen_TriangularView : std::false_type {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_eigen_TriangularView<Eigen::TriangularView<MatrixType, UpLo>> : std::true_type {};


    template<typename T>
    struct is_eigen_DiagonalMatrix : std::false_type {};

    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct is_eigen_DiagonalMatrix<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
      : std::true_type {};


    template<typename T>
    struct is_eigen_DiagonalWrapper : std::false_type {};

    template<typename DiagonalVectorType>
    struct is_eigen_DiagonalWrapper<Eigen::DiagonalWrapper<DiagonalVectorType>> : std::true_type {};


    template<typename T>
    struct is_eigen_Identity : std::false_type {};

    template<typename Scalar, typename Arg>
    struct is_eigen_Identity<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>>
      : std::true_type {};
  }


  /**
   * \brief T is of type Eigen::SelfAdjointView.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_SelfAdjointView = detail::is_eigen_SelfAdjointView<std::decay_t<T>>::value;
#else
  constexpr bool eigen_SelfAdjointView = detail::is_eigen_SelfAdjointView<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::TriangularView.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_TriangularView = detail::is_eigen_TriangularView<std::decay_t<T>>::value;
#else
  constexpr bool eigen_TriangularView = detail::is_eigen_TriangularView<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_DiagonalMatrix = detail::is_eigen_DiagonalMatrix<std::decay_t<T>>::value;
#else
  constexpr bool eigen_DiagonalMatrix = detail::is_eigen_DiagonalMatrix<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_DiagonalWrapper = detail::is_eigen_DiagonalWrapper<std::decay_t<T>>::value;
#else
  constexpr bool eigen_DiagonalWrapper = detail::is_eigen_DiagonalWrapper<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is a general eigen native type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept native_eigen_general =
#else
  constexpr bool native_eigen_general =
#endif
    native_eigen_matrix<T> or native_eigen_array<T> or eigen_SelfAdjointView<T> or eigen_TriangularView<T> or
    eigen_DiagonalMatrix<T> or eigen_DiagonalWrapper<T> or
    (std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>> and
      not std::is_base_of_v<Eigen3Base, std::decay_t<T>>);


  /**
   * \brief T is an Eigen identity matrix (not necessarily an \ref identity_matrix).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_Identity = detail::is_eigen_Identity<std::decay_t<T>>::value;
#else
  constexpr bool eigen_Identity = detail::is_eigen_Identity<std::decay_t<T>>::value;
#endif



} // namespace OpenKalman::Eigen3


namespace OpenKalman::internal
{
  using namespace OpenKalman::Eigen3;

  // ---------------------- //
  //  is_modifiable_native  //
  // ---------------------- //

  // no_assignment_operator is a private base class of Cwise___Operator, Select, DiagonalWrapper, and a few others.
  // This also includes ZeroMatrix, which derives from CwiseNullaryOperator.
#ifdef __cpp_concepts
  template<typename T, typename U> requires std::is_base_of_v<Eigen::internal::no_assignment_operator, T>
  struct is_modifiable_native<T, U>
#else
    template<typename T, typename U>
      struct is_modifiable_native<T, U, std::enable_if_t<std::is_base_of_v<Eigen::internal::no_assignment_operator, T>>>
#endif
    : std::false_type {};


  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel, typename U>
  struct is_modifiable_native<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>, U>
    : std::bool_constant<bool(Eigen::internal::traits<
      Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>::Flags & Eigen::LvalueBit) and
      (not has_const<XprType>::value) and
      (BlockRows == Eigen::Dynamic or row_dimension_of_v<U> == BlockRows) and
      (BlockRows != Eigen::Dynamic or row_dimension_of_v<U> == dynamic_size) and
      (BlockCols == Eigen::Dynamic or column_dimension_of_v<U> == BlockCols) and
      (BlockCols != Eigen::Dynamic or column_dimension_of_v<U> == dynamic_size) and
      (std::is_same_v<scalar_type_of_t<XprType>, scalar_type_of_t<U>>)> {};


  template<typename XprType, typename U>
  struct is_modifiable_native<Eigen::Inverse<XprType>, U>
    : std::false_type {};


  template<typename LhsType, typename RhsType, int Option, typename U>
  struct is_modifiable_native<Eigen::Product<LhsType, RhsType, Option>, U>
    : std::false_type {};


  template<typename MatrixType, int RowFactor, int ColFactor, typename U>
  struct is_modifiable_native<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, U>
    : std::false_type {};


  template<typename MatrixType, int Direction, typename U>
  struct is_modifiable_native<Eigen::Reverse<MatrixType, Direction>, U>
    : std::false_type {};

}


namespace OpenKalman
{

  // -------------- //
  //  MatrixTraits  //
  // -------------- //

  /**
   * \internal
   * \brief Default matrix traits for any \ref native_eigen_matrix.
   * \tparam M The matrix.
   */
#ifdef __cpp_concepts
  template<typename M> requires Eigen3::native_eigen_matrix<M> or Eigen3::native_eigen_array<M>
  struct MatrixTraits<M>
#else
  template<typename M>
  struct MatrixTraits<M, std::enable_if_t<Eigen3::native_eigen_matrix<M> or Eigen3::native_eigen_array<M>>>
#endif
  {
  private:

    // Identify the correct Eigen::Matrix based on template parameters and the traits of M.
    template<typename S, std::size_t r, std::size_t c>
    using Nat = Eigen::Matrix<S, r == dynamic_size ? Eigen::Dynamic : (Eigen::Index) r,
      c == dynamic_size ? Eigen::Dynamic : (Eigen::Index) c,
      (std::decay_t<M>::Flags & Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign>;

    using Scalar = scalar_type_of_t<M>;
    static constexpr std::size_t rows = row_dimension_of_v<M>;
    static constexpr std::size_t columns = column_dimension_of_v<M>;

  public:

    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = std::conditional_t<hermitian_matrix<Nat<Scalar, dim, dim>>,
      Nat<Scalar, dim, dim>, SelfAdjointMatrix<Nat<Scalar, dim, dim>, storage_triangle>>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows>
    using TriangularMatrixFrom = std::conditional_t<triangular_matrix<Nat<Scalar, dim, dim>>,
      Nat<Scalar, dim, dim>, TriangularMatrix<Nat<Scalar, dim, dim>, triangle_type>>;


    template<std::size_t dim = rows>
    using DiagonalMatrixFrom = std::conditional_t<diagonal_matrix<Nat<Scalar, dim, 1>>,
      Nat<Scalar, dim, 1>, DiagonalMatrix<Nat<Scalar, dim, 1>>>;


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::Eigen3AdapterBase<Derived, M>;


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::native_eigen_matrix<Arg>, int> = 0>
#endif
    static Arg&& make(Arg&& arg) noexcept
    {
      return std::forward<Arg>(arg);
    }


    /**
     * \brief Make a fixed matrix of size M from a list of coefficients in row-major order.
     * \details Makes a matrix using size information from matrix M, based on the following rules:
     *  - If M has fixed shape, there must be one input for each element.
     *  - If M has fixed rows and dynamic columns, the number of inputs must be a multiple of the number of rows.
     *  - If M has fixed columns and dynamic rows, the number of inputs must be a multiple of the number of columns.
     *  - Otherwise, if both rows and columns are dynamic, the result will be a fixed column vector.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> Arg, std::convertible_to<Scalar> ... Args> requires
      (rows == dynamic_size or columns == dynamic_size or (1 + sizeof...(Args) == rows * columns)) and
      (rows == dynamic_size or columns != dynamic_size or ((1 + sizeof...(Args)) % rows == 0)) and
      (rows != dynamic_size or columns == dynamic_size or ((1 + sizeof...(Args)) % columns == 0))
#else
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wdiv-by-zero"
    template<typename Arg, typename ... Args, std::enable_if_t<
      std::conjunction_v<std::is_convertible<Arg, Scalar>, std::is_convertible<Args, Scalar>...> and
      (rows == dynamic_size or columns == dynamic_size or (1 + sizeof...(Args) == rows * columns)) and
      (rows == dynamic_size or columns != dynamic_size or ((1 + sizeof...(Args)) % rows == 0)) and
      (rows != dynamic_size or columns == dynamic_size or ((1 + sizeof...(Args)) % columns == 0)), int> = 0>
#endif
    static auto make(const Arg arg, const Args ... args)
    {
      using namespace Eigen3;

      if constexpr (rows != dynamic_size and columns != dynamic_size)
        return ((eigen_matrix_t<Scalar, rows, columns> {} << arg), ... , args).finished();
      else if constexpr (rows != dynamic_size and columns == dynamic_size)
        return ((eigen_matrix_t<Scalar, rows, (1 + sizeof...(Args)) / rows> {} << arg), ... , args).finished();
      else if constexpr (rows == dynamic_size and columns != dynamic_size)
        return ((eigen_matrix_t<Scalar, (1 + sizeof...(Args)) / columns, columns> {} << arg), ... , args).finished();
      else
      {
        static_assert(rows == dynamic_size and columns == dynamic_size);
        return ((eigen_matrix_t<Scalar, 1 + sizeof...(Args), 1> {} << arg), ... , args).finished();
      }
    }
#ifndef __cpp_concepts
# pragma GCC diagnostic pop
#endif

  };


} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP
