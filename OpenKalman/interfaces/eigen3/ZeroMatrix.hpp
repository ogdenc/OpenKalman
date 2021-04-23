/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
#define OPENKALMAN_EIGEN3_ZEROMATRIX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{

  namespace detail
  {

#ifdef __cpp_concepts
    template<typename Derived, typename Scalar, std::size_t rows, std::size_t columns>
#else
    template<typename Derived, typename Scalar, std::size_t rows, std::size_t columns, typename = void>
#endif
    struct ZeroMatrixDynamicBase {};


    /*
     * Dynamic rows and Dynamic columns
     */
    template<typename Derived, typename Scalar>
    struct ZeroMatrixDynamicBase<Derived, Scalar, 0, 0> : Eigen3::internal::Eigen3Base<Derived>
    {

    private:

      const std::size_t rows_;
      const std::size_t cols_;

    public:

      /**
       * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
       * \param r Number of rows.
       * \param c Number of columns.
       */
      ZeroMatrixDynamicBase(std::size_t r, std::size_t c) : rows_ {r}, cols_ {c} {}


      /**
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<eigen_native M>
#else
      template<typename M, std::enable_if_t<eigen_native<M>, int> = 0>
#endif
      ZeroMatrixDynamicBase(M&& m) : rows_ {m.rows()}, cols_ {m.cols()} {}


      /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
      Eigen::Index rows() const { return rows_; }


      /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
      Eigen::Index cols() const { return cols_; }

    };


    /*
     * Dynamic rows and fixed columns
     */
    template<typename Derived, typename Scalar, std::size_t columns>
#ifdef __cpp_concepts
      requires (columns > 0)
    struct ZeroMatrixDynamicBase<Derived, Scalar, 0, columns>
#else
    struct ZeroMatrixDynamicBase<Derived, Scalar, 0, columns, std::enable_if_t<(columns > 0)>>
#endif
      : Eigen3::internal::Eigen3Base<Derived>
    {

    private:

      const std::size_t rows_;

    public:

      /**
       * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
       * \param r Number of rows.
       */
      ZeroMatrixDynamicBase(std::size_t r) : rows_ {r} {}


      /**
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<eigen_native M> requires dynamic_rows<M> and (MatrixTraits<M>::columns == columns)
#else
      template<typename M, std::enable_if_t<
        eigen_native<M> and dynamic_rows<M> and (MatrixTraits<M>::columns == columns), int> = 0>
#endif
      ZeroMatrixDynamicBase(M&& m) : rows_ {m.rows()} {}


      /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
      Eigen::Index rows() const { return rows_; }


      /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
      static constexpr Eigen::Index cols() { return columns; }

    };


    /*
     * Fixed rows and dynamic columns
     */
    template<typename Derived, typename Scalar, std::size_t rows_>
#ifdef __cpp_concepts
      requires (rows_ > 0)
    struct ZeroMatrixDynamicBase<Derived, Scalar, rows_, 0>
#else
      struct ZeroMatrixDynamicBase<Derived, Scalar, rows_, 0, std::enable_if_t<(rows_ > 0)>>
#endif
      : Eigen3::internal::Eigen3Base<Derived>
    {

    private:

      const std::size_t cols_;

    public:

      /**
       * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
       * \param c Number of columns.
       */
      ZeroMatrixDynamicBase(std::size_t c) : cols_ {c} {}


      /**
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<eigen_native M> requires (MatrixTraits<M>::rows == rows_) and dynamic_columns<M>
#else
      template<typename M, std::enable_if_t<
        eigen_native<M> and (MatrixTraits<M>::rows == rows_) and dynamic_columns<M>, int> = 0>
#endif
      ZeroMatrixDynamicBase(M&& m) : cols_ {m.cols()} {}


      /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
      static constexpr Eigen::Index rows() { return rows_; }


      /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
      Eigen::Index cols() const { return cols_; }

    };


    /*
     * Fixed rows and fixed columns
     */
    template<typename Derived, typename Scalar, std::size_t rows_, std::size_t columns>
#ifdef __cpp_concepts
    requires (rows_ > 0) and (columns > 0)
    struct ZeroMatrixDynamicBase<Derived, Scalar, rows_, columns>
#else
    struct ZeroMatrixDynamicBase<Derived, Scalar, rows_, columns, std::enable_if_t<(rows_ > 0) and (columns > 0)>>
#endif
      : Eigen3::internal::Eigen3Base<Derived>
    {

      /**
       * \brief Default constructor.
       */
      ZeroMatrixDynamicBase() {};


      /**
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<eigen_native M> requires (MatrixTraits<M>::rows == rows_) and (MatrixTraits<M>::columns == columns)
#else
      template<typename M, std::enable_if_t<
        eigen_native<M> and (MatrixTraits<M>::rows == rows_) and (MatrixTraits<M>::columns == columns), int> = 0>
#endif
      ZeroMatrixDynamicBase(M&& m) {}


      /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
      static constexpr Eigen::Index rows() { return rows_; }


      /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
      static constexpr Eigen::Index cols() { return columns; }

    };

  } // namespace detail


  // ------------ //
  //  ZeroMatrix  //
  // ------------ //

  // ZeroMatrix is declared in eigen3-forward-declarations.hpp.
  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct ZeroMatrix : detail::ZeroMatrixDynamicBase<ZeroMatrix<Scalar, rows, columns>, Scalar, rows, columns>
  {

  private:

    using Base = detail::ZeroMatrixDynamicBase<ZeroMatrix, Scalar, rows, columns>;

  public:

    using Base::Base;

    /**
     * \brief Element accessor.
     * \param r The row.
     * \param c The column.
     * \return The element at row r and column c (always zero of type Scalar).
     */
    constexpr Scalar operator()(std::size_t r, std::size_t c) const
    {
      assert(Eigen::Index(r) < this->rows());
      assert(Eigen::Index(c) < this->cols());
      return 0;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always zero of type Scalar).
     */
#ifdef __cpp_concepts
    constexpr Scalar operator[](std::size_t i) const requires (rows == 1) or (columns == 1)
#else
    template<std::size_t rows_ = rows, std::enable_if_t<(rows_ == 1) or (columns == 1), int> = 0>
      constexpr Scalar operator[](std::size_t i) const
#endif
    {
      assert(rows == 1 or Eigen::Index(i) < this->rows());
      assert(columns == 1 or Eigen::Index(i) < this->cols());
      return 0;
    }


    /**
     * \brief Element accessor for a row or column vector.
     * \param i The row or column.
     * \return The element at row or column i (always zero of type Scalar).
     */
#ifdef __cpp_concepts
    constexpr Scalar operator()(std::size_t i) const requires (rows == 1) or (columns == 1)
#else
    template<std::size_t rows_ = rows, std::enable_if_t<(rows_ == 1) or (columns == 1), int> = 0>
      constexpr Scalar operator()(std::size_t i) const
#endif
    {
      assert(rows == 1 or Eigen::Index(i) < this->rows());
      assert(columns == 1 or Eigen::Index(i) < this->cols());
      return 0;
    }


    /// \internal \note Required by Eigen 3 for this to be used in an Eigen::CwiseBinaryOp.
    using Nested = ZeroMatrix;

  };


  /**
   * \brief Deduction guide for constructing a ZeroMatrix based on the shape of M.
   */
#ifdef __cpp_concepts
  template<eigen_native M>
#else
  template<typename M, std::enable_if_t<eigen_native<M>, int> = 0>
#endif
  ZeroMatrix(M&&) -> ZeroMatrix<typename MatrixTraits<M>::Scalar, MatrixTraits<M>::rows, MatrixTraits<M>::columns>;


} // OpenKalman::Eigen3


namespace OpenKalman
{
  // -------- //
  //  Traits  //
  // -------- //

  template<typename Scalar_, std::size_t rows_, std::size_t columns_>
  struct MatrixTraits<Eigen3::ZeroMatrix<Scalar_, rows_, columns_>>
  {
    using Scalar = Scalar_;

    static constexpr std::size_t rows = rows_;
    static constexpr std::size_t columns = columns_;

  private:

    using Matrix = Eigen3::ZeroMatrix<Scalar, rows, columns>;

    template<typename S, std::size_t r, std::size_t c>
    using Nat =
    Eigen::Matrix<S, r == 0 ? Eigen::Dynamic : (Eigen::Index) r, c == 0 ? Eigen::Dynamic : (Eigen::Index) c>;

  public:

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = Nat<S, r, c>;


    using SelfContainedFrom = Matrix;


    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<Matrix, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<Matrix, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<Nat<S, dim, 1>>;


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Matrix>;


#ifdef __cpp_concepts
    static auto zero() requires (rows > 0) and (columns > 0)
#else
    template<std::size_t rw = rows_, std::enable_if_t<(rw > 0) and (columns > 0), int> = 0>
    static auto zero()
#endif
    {
      return Eigen3::ZeroMatrix<Scalar, rows, columns> {};
    }


#ifdef __cpp_concepts
    static auto zero(std::size_t r, std::size_t c) requires (rows == 0) or (columns == 0)
#else
    template<std::size_t rw = rows_, std::enable_if_t<(rw == 0) or (columns == 0), int> = 0>
    static auto zero(std::size_t r, std::size_t c)
#endif
    {
      return Eigen3::ZeroMatrix<Scalar, rows, columns> {r, c};
    }


#ifdef __cpp_concepts
    static auto identity() requires (rows > 0)
#else
    template<std::size_t rw = rows_, std::enable_if_t<(rw > 0), int> = 0>
    static auto identity()
#endif
    {
      return Nat<Scalar, rows, rows>::Identity();
    }


#ifdef __cpp_concepts
    static auto identity(std::size_t i) requires (rows == 0)
#else
    template<std::size_t rw = rows_, std::enable_if_t<(rw == 0), int> = 0>
    static auto identity(std::size_t i)
#endif
    {
      return Nat<Scalar, 0, 0>::Identity(i, i);
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
  template<typename...Ts, eigen_zero_expr Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t row_count(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>)
      return arg.rows();
    else
      return MatrixTraits<Arg>::rows;
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t column_count(Arg&& arg)
  {
    if constexpr (dynamic_columns<Arg>)
      return arg.cols();
    else
      return MatrixTraits<Arg>::columns;
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    if constexpr (dynamic_rows<Arg>)
    {
      const std::size_t dim = row_count(arg);
      return ZeroMatrix<Scalar, 0, 0>(dim, dim);
    }
    else
    {
      constexpr auto dim = MatrixTraits<Arg>::rows;
      return ZeroMatrix<Scalar, dim, dim>();
    }
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and
    (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    if constexpr (dynamic_rows<Arg>)
    {
      return ZeroMatrix<Scalar, 0, 1> {row_count(arg)};
    }
    else
    {
      return ZeroMatrix<Scalar, MatrixTraits<Arg>::rows, 1> {};
    }
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    if constexpr (dynamic_rows<Arg> and dynamic_columns<Arg>)
    {
      return ZeroMatrix<Scalar, 0, 0> {column_count(arg), row_count(arg)};
    }
    else if constexpr (dynamic_rows<Arg> and not dynamic_columns<Arg>)
    {
      constexpr auto cols = MatrixTraits<Arg>::columns;
      return ZeroMatrix<Scalar, cols, 0> {cols, row_count(arg)};
    }
    else if constexpr (not dynamic_rows<Arg> and dynamic_columns<Arg>)
    {
      constexpr auto rows = MatrixTraits<Arg>::rows;
      return ZeroMatrix<Scalar, 0, rows> {column_count(arg), rows};
    }
    else // if constexpr (not dynamic_rows<Arg> and not dynamic_columns<Arg>)
    {
      constexpr auto rows = MatrixTraits<Arg>::rows;
      constexpr auto cols = MatrixTraits<Arg>::columns;
      if constexpr (rows == cols)
        return std::forward<Arg>(arg);
      else
        return ZeroMatrix<Scalar, cols, rows> {};
    }
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    return transpose(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and
    (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto
  determinant(Arg&& arg) noexcept
  {
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and
    (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (dynamic_rows<Arg>)
    {
      return ZeroMatrix<Scalar, 0, 1> {row_count(arg)};
    }
    else
    {
      constexpr auto rows = MatrixTraits<Arg>::rows;
      return ZeroMatrix<Scalar, rows, 1> {};
    }
  }


  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<eigen_zero_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    if constexpr (dynamic_rows<A>)
    {
      auto dim = row_count(a);
      return ZeroMatrix<Scalar, 0, 0> {dim, dim};
    }
    else
    {
      constexpr auto dim = MatrixTraits<A>::rows;
      return ZeroMatrix<Scalar, dim, dim> {};
    }
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<eigen_zero_expr<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    if constexpr (dynamic_columns<A>)
    {
      auto dim = column_count(a);
      return ZeroMatrix<Scalar, 0, 0> {dim, dim};
    }
    else
    {
      constexpr auto dim = MatrixTraits<A>::columns;
      return ZeroMatrix<Scalar, dim, dim> {};
    }
  }


  /// Get an element of a ZeroMatrix matrix. Always 0.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg& arg, const std::size_t row, const std::size_t col)
  {
    assert(row < row_count(arg));
    assert(col < column_count(arg));
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Get an element of a one-column ZeroMatrix matrix. Always 0.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg& arg, const std::size_t row)
  {
    assert(row < row_count(arg));
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    assert(index < column_count(arg));
    return reduce_columns(std::forward<Arg>(arg));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_zero_expr Arg> requires (not dynamic_columns<Arg>) and
    (index < MatrixTraits<Arg>::columns)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    eigen_zero_expr<Arg> and (not dynamic_columns<Arg>) and (index < MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    if constexpr(column_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return reduce_columns(std::forward<Arg>(arg));
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

#ifdef __cpp_concepts
  template<eigen_matrix Arg1, eigen_matrix Arg2> requires (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_matrix<Arg1> and eigen_matrix<Arg2> and (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns),
    int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr (zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      static_assert((zero_matrix<Arg1>));
      return std::forward<Arg2>(arg2);
    }
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg1, eigen_matrix Arg2> requires (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_matrix<Arg1> and eigen_matrix<Arg2> and (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns),
    int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      static_assert((zero_matrix<Arg1>));
      return -std::forward<Arg2>(arg2);
    }
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg1, eigen_matrix Arg2> requires (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_matrix<Arg1> and eigen_matrix<Arg2> and (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    constexpr auto rows = MatrixTraits<Arg1>::rows;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    if constexpr (dynamic_rows<Arg1> and dynamic_columns<Arg2>)
    {
      return ZeroMatrix<Scalar, 0, 0> {row_count(arg1), column_count(arg2)};
    }
    else if constexpr (dynamic_rows<Arg1> and not dynamic_columns<Arg2>)
    {
      return ZeroMatrix<Scalar, rows, cols> {row_count(arg1)};
    }
    else if constexpr (not dynamic_rows<Arg1> and dynamic_columns<Arg2>)
    {
      return ZeroMatrix<Scalar, rows, cols> {column_count(arg2)};
    }
    else
    {
      static_assert((not dynamic_rows<Arg1> and not dynamic_columns<Arg2>));
      return ZeroMatrix<Scalar, rows, cols> {};
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (eigen_zero_expr<Arg1> and std::is_arithmetic_v<Arg2>) or
    (std::is_arithmetic_v<Arg1> and eigen_zero_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(eigen_zero_expr<Arg1> and std::is_arithmetic_v<Arg2>) or
    (std::is_arithmetic_v<Arg1> and eigen_zero_expr<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(eigen_zero_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }



  /**
   * \brief Divide an \ref eigen_zero_expr by a scalar.
   * \tparam Arg1 An \ref eigen_zero_expr.
   * \tparam Arg2 An arithmetic scalar type.
   * \return If it does not throw a divide-by-zero exception, the result will be \ref eigen_zero_expr.
   */
  #ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires eigen_zero_expr<Arg1> and std::is_arithmetic_v<Arg2>
  #else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_zero_expr<Arg1> and std::is_arithmetic_v<Arg2>, int> = 0>
  #endif
  constexpr decltype(auto) operator/(Arg1&& arg1, Arg2&& arg2)
  {
    if (arg2 == 0) throw std::runtime_error("ZeroMatrix / 0: divide by zero error");
    return std::forward<Arg1>(arg1);
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


}

#endif //OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
