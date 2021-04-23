/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP
#define OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{

  // ConstantMatrix is declared in eigen3-forward-declarations.hpp.

  template<typename Scalar, auto constant, std::size_t rows_, std::size_t columns>
#ifdef __cpp_concepts
    requires (rows_ > 0) and (columns > 0) and std::is_arithmetic_v<Scalar>
#endif
  struct ConstantMatrix : Eigen3::internal::Eigen3Base<ConstantMatrix<Scalar, constant, rows_, columns>>
  {

  private:

#ifndef __cpp_concepts
    static_assert((rows_ > 0) and (columns > 0) and std::is_arithmetic_v<Scalar>);
#endif

  public:

    /**
     * \brief Default constructor.
     */
    ConstantMatrix() {}


    /**
     * \brief Element accessor.
     * \note Does not do any bounds checking.
     * \param r The row.
     * \param c The column.
     * \return The element at row r and column c (always the constant).
     */
    constexpr Scalar operator()(std::size_t r, std::size_t c) const
    {
      assert(r < rows_);
      assert(c < columns);
      return constant;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always the constant).
     */
#ifdef __cpp_concepts
    constexpr Scalar operator[](std::size_t i) const requires (rows_ == 1) or (columns == 1)
#else
    template<std::size_t r = rows_, std::enable_if_t<(r == 1) or (columns == 1), int> = 0>
    constexpr Scalar operator[](std::size_t i) const
#endif
    {
      assert(rows_ == 1 or i < rows_);
      assert(columns == 1 or i < columns);
      return constant;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always the constant).
     */
#ifdef __cpp_concepts
    constexpr Scalar operator()(std::size_t i) const requires (rows_ == 1) or (columns == 1)
#else
    template<std::size_t r = rows_, std::enable_if_t<(r == 1) or (columns == 1), int> = 0>
    constexpr Scalar operator()(std::size_t i) const
#endif
    {
      assert(rows_ == 1 or i < rows_);
      assert(columns == 1 or i < columns);
      return constant;
    }


    /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
    static constexpr Eigen::Index rows() { return rows_; }


    /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
    static constexpr Eigen::Index cols() { return columns; }


    /// \internal \note Required by Eigen 3 for this to be used in an Eigen::CwiseBinaryOp.
    using Nested = ConstantMatrix;

  };


} // OpenKalman::Eigen3


namespace OpenKalman
{
  // -------- //
  //  Traits  //
  // -------- //

  template<typename Scalar_, auto constant_, std::size_t rows_, std::size_t columns_>
  struct MatrixTraits<Eigen3::ConstantMatrix<Scalar_, constant_, rows_, columns_>>
  {
    using Scalar = Scalar_;

    static constexpr auto constant = constant_;

    static constexpr std::size_t rows = rows_;

    static constexpr std::size_t columns = columns_;


  private:

    using Matrix = Eigen3::ConstantMatrix<Scalar, constant, rows, columns>;

  public:

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = Eigen::Matrix<S, r, c>;


    using SelfContainedFrom = Matrix;


    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<Matrix, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<Matrix, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<Eigen3::ConstantMatrix<S, constant, dim, 1>>;


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Matrix>;


    static constexpr auto zero()
    {
      return Eigen3::ZeroMatrix<Scalar, rows, columns> {};
    }


    static constexpr auto identity()
    {
      return Eigen::Matrix<Scalar, rows, rows>::Identity();
    }


  };

} // namespace OpenKalman


namespace OpenKalman::Eigen3
{

  // ----------- //
  //  Overloads  //
  // ----------- //

  /// Convert to self-contained version of the matrix.
#ifdef __cpp_concepts
  template<typename...Ts, eigen_constant_expr Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t
  row_count(Arg&& arg)
  {
    return MatrixTraits<Arg>::rows;
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t
  column_count(Arg&& arg)
  {
    return MatrixTraits<Arg>::columns;
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    return DiagonalMatrix {std::forward<Arg>(arg)};
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto constant = MatrixTraits<Arg>::constant;
    return ConstantMatrix<Scalar, constant, MatrixTraits<Arg>::rows, 1> {};
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    constexpr auto rows = MatrixTraits<Arg>::rows;
    constexpr auto cols = MatrixTraits<Arg>::columns;
    if constexpr (rows == cols)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      constexpr auto constant = MatrixTraits<Arg>::constant;
      return ConstantMatrix<Scalar, constant, cols, rows> {};
    }
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    return transpose(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  constexpr auto
  determinant(Arg&& arg) noexcept
  {
    return static_cast<decltype(MatrixTraits<Arg>::constant)>(0);
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    return static_cast<decltype(MatrixTraits<Arg>::constant)>(MatrixTraits<Arg>::constant * MatrixTraits<Arg>::rows);
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      constexpr auto constant = MatrixTraits<Arg>::constant;
      return ConstantMatrix<Scalar, constant, MatrixTraits<Arg>::rows, 1> {};
    }
  }


  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr A>
#else
  template<typename A, std::enable_if_t<eigen_constant_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    constexpr auto constant = MatrixTraits<A>::constant;
    constexpr auto dim = MatrixTraits<A>::rows;
    Scalar elem = constant * OpenKalman::internal::constexpr_sqrt(Scalar {MatrixTraits<A>::columns});
    auto col1 = Eigen::Matrix<Scalar, dim, 1>::Constant(elem);
    ConstantMatrix<Scalar, 0, dim, dim - 1> othercols;
    return concatenate_horizontal(col1, othercols);
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr A>
#else
  template<typename A, std::enable_if_t<eigen_constant_expr<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    constexpr auto constant = MatrixTraits<A>::constant;
    constexpr auto dim = MatrixTraits<A>::columns;
    Scalar elem = constant * OpenKalman::internal::constexpr_sqrt(Scalar {MatrixTraits<A>::rows});
    auto row1 = Eigen::Matrix<Scalar, 1, dim>::Constant(elem);
    ConstantMatrix<Scalar, 0, dim - 1, dim> otherrows;
    return concatenate_vertical(row1, otherrows);
  }


  /// Get an element of a ZeroMatrix matrix. Always 0.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg&, const std::size_t row, const std::size_t col)
  {
    assert(row < MatrixTraits<Arg>::rows);
    assert(col < MatrixTraits<Arg>::columns);
    return MatrixTraits<Arg>::constant;
  }


  /// Get an element of a one-column ZeroMatrix matrix. Always 0.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg&, const std::size_t row)
  {
    assert(row < MatrixTraits<Arg>::rows);
    return MatrixTraits<Arg>::constant;
  }


  /// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    assert(index < MatrixTraits<Arg>::columns);
    return reduce_columns(std::forward<Arg>(arg));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_constant_expr Arg> requires (index < MatrixTraits<Arg>::columns)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    eigen_constant_expr<Arg> and (index < MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    return reduce_columns(std::forward<Arg>(arg));
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

#ifdef __cpp_concepts
  template<eigen_constant_expr Arg1, eigen_constant_expr Arg2> requires
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns) and
    std::same_as<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<eigen_constant_expr<Arg1> and eigen_constant_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns) and
    std::is_same_v<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>, int> = 0>
#endif
  constexpr auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto newconst = MatrixTraits<Arg1>::constant + MatrixTraits<Arg2>::constant;
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    return ConstantMatrix<Scalar, newconst, MatrixTraits<Arg1>::rows, MatrixTraits<Arg1>::columns> {};
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg1, eigen_constant_expr Arg2> requires
  (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns) and
    std::same_as<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<eigen_constant_expr<Arg1> and eigen_constant_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns) and
    std::is_same_v<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>, int> = 0>
#endif
  constexpr auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto newconst = MatrixTraits<Arg1>::constant - MatrixTraits<Arg2>::constant;
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    return ConstantMatrix<Scalar, newconst, MatrixTraits<Arg1>::rows, MatrixTraits<Arg1>::columns> {};
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg1, eigen_constant_expr Arg2> requires
  (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows) and
    std::same_as<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<eigen_constant_expr<Arg1> and eigen_constant_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows) and
    std::is_same_v<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto newconst = MatrixTraits<Arg1>::constant * MatrixTraits<Arg2>::constant * MatrixTraits<Arg2>::rows;
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    return ConstantMatrix<Scalar, newconst, MatrixTraits<Arg1>::rows, MatrixTraits<Arg2>::columns> {};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (eigen_constant_expr<Arg1> and std::is_arithmetic_v<Arg2>) or
    (std::is_arithmetic_v<Arg1> and eigen_constant_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(eigen_constant_expr<Arg1> and std::is_arithmetic_v<Arg2>) or
    (std::is_arithmetic_v<Arg1> and eigen_constant_expr<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(eigen_constant_expr<Arg1>)
    {
      constexpr auto constant = MatrixTraits<Arg1>::constant;
      using Scalar = typename MatrixTraits<Arg1>::Scalar;
      constexpr std::size_t r = MatrixTraits<Arg1>::rows;
      constexpr std::size_t c = MatrixTraits<Arg1>::columns;

      if constexpr (constant == 0) return std::forward<Arg1>(arg1);
      else return Eigen::Matrix<Scalar, r, c>::Constant(MatrixTraits<Arg1>::constant * arg2);
    }
    else
    {
      constexpr auto constant = MatrixTraits<Arg2>::constant;
      using Scalar = typename MatrixTraits<Arg2>::Scalar;
      constexpr std::size_t r = MatrixTraits<Arg2>::rows;
      constexpr std::size_t c = MatrixTraits<Arg2>::columns;

      if constexpr (constant == 0) return std::forward<Arg2>(arg2);
      else return Eigen::Matrix<Scalar, r, c>::Constant(arg1 * MatrixTraits<Arg2>::constant);
    }
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg1, typename Arg2> requires std::is_arithmetic_v<Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_constant_expr<Arg1> and std::is_arithmetic_v<Arg2>, int> = 0>
#endif
  constexpr decltype(auto) operator/(Arg1&& arg1, Arg2 arg2)
  {
    constexpr auto constant = MatrixTraits<Arg1>::constant;
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    constexpr std::size_t r = MatrixTraits<Arg1>::rows;
    constexpr std::size_t c = MatrixTraits<Arg1>::columns;

    if constexpr (constant == 0) return std::forward<Arg1>(arg1);
    else return Eigen::Matrix<Scalar, r, c>::Constant(MatrixTraits<Arg1>::constant / arg2);
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg&& arg)
  {
    constexpr auto constant = MatrixTraits<Arg>::constant;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr std::size_t r = MatrixTraits<Arg>::rows;
    constexpr std::size_t c = MatrixTraits<Arg>::columns;

    if constexpr (constant == 0) return std::forward<Arg>(arg);
    else return ConstantMatrix<Scalar, -constant, r, c> {};
  }


}

#endif //OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP
