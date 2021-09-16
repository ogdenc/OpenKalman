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
 * \brief Definitions for Eigen3::ZeroMatrix
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

      /*
       * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
       * \param r Number of rows.
       * \param c Number of columns.
       */
      ZeroMatrixDynamicBase(std::size_t r, std::size_t c) : rows_ {r}, cols_ {c} {}


      /*
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<typename M> requires (MatrixTraits<M>::rows == 0) and (MatrixTraits<M>::columns == 0)
#else
      template<typename M, std::enable_if_t<(MatrixTraits<M>::rows == 0) and (MatrixTraits<M>::columns == 0), int> = 0>
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

      /*
       * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
       * \param r Number of rows.
       */
      ZeroMatrixDynamicBase(std::size_t r) : rows_ {r} {}


      /*
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<typename M> requires dynamic_rows<M> and (MatrixTraits<M>::columns == columns)
#else
      template<typename M, std::enable_if_t<dynamic_rows<M> and (MatrixTraits<M>::columns == columns), int> = 0>
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

      /*
       * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
       * \param c Number of columns.
       */
      ZeroMatrixDynamicBase(std::size_t c) : cols_ {c} {}


      /*
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<typename M> requires (MatrixTraits<M>::rows == rows_) and dynamic_columns<M>
#else
      template<typename M, std::enable_if_t<(MatrixTraits<M>::rows == rows_) and dynamic_columns<M>, int> = 0>
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

      /*
       * \brief Default constructor.
       */
      ZeroMatrixDynamicBase() {};


      /*
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<typename M> requires (MatrixTraits<M>::rows == rows_) and (MatrixTraits<M>::columns == columns)
#else
      template<typename M, std::enable_if_t<
        (MatrixTraits<M>::rows == rows_) and (MatrixTraits<M>::columns == columns), int> = 0>
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

    /**
     * \brief Construct a ZeroMatrix.
     * \details The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, ZeroMatrix {2, 3} constructs a 2-by-3 dynamic matrix, ZeroMatrix {3} constructs a
     * 2-by-3 matrix in which there are two fixed row dimensions and three dynamic column dimensions, and
     * ZeroMatrix {} constructs a fixed matrix.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    ZeroMatrix(const Args...args) : Base {static_cast<std::size_t>(args)...} {}


    /**
     * \brief Construct a ZeroMatrix based on the shape of another matrix.
     * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
     */
#ifdef __cpp_concepts
    template<typename M> requires (MatrixTraits<M>::rows == rows) and (MatrixTraits<M>::columns == columns)
#else
    template<typename M, std::enable_if_t<
      (MatrixTraits<M>::rows == rows) and (MatrixTraits<M>::columns == columns), int> = 0>
#endif
    ZeroMatrix(M&& m) : Base {std::forward<M>(m)} {}


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


  /**
   * \brief Make a ZeroMatrix of a given fixed or dynamic shape.
   * \tparam Scalar The scalar type.
   * \tparam rows The number of rows (or 0 if the rows are dynamic).
   * \tparam columns The number of columns (or 0 if the columns are dynamic).
   * \tparam Args Row or column (or both) arguments (in that order if both are given) necessary to define any
   * dynamic dimensions. Unnecessary parameters are discarded.
   * \return A ZeroMatrix<Scalar, rows, columns>.
   */
#ifdef __cpp_concepts
  template<typename Scalar, std::size_t rows, std::size_t columns, std::convertible_to<std::size_t>...Args> requires
    (sizeof...(Args) <= 2)
#else
  template<typename Scalar, std::size_t rows, std::size_t columns, typename...Args, std::enable_if_t<
    (std::is_convertible_v<Args, std::size_t> and ... and (sizeof...(Args) <= 2)), int> = 0>
#endif
  inline auto make_ZeroMatrix(const Args...args)
  {
    constexpr std::size_t arg_count = (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0);
    return std::apply([] (auto...as) {
      return ZeroMatrix<Scalar, rows, columns> {as...};
      }, OpenKalman::internal::tuple_slice<0, arg_count>(std::forward_as_tuple(args...)));
  }

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

  public:

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = Eigen3::eigen_matrix_t<S, r, c>;


    using SelfContainedFrom = Matrix;


    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<Matrix, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<Matrix, triangle_type>;


    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<Eigen3::eigen_matrix_t<S, dim, 1>>;


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Matrix>;


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return Eigen3::ZeroMatrix<Scalar, rows, columns> {static_cast<std::size_t>(args)...};
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return Eigen3::eigen_matrix_t<Scalar, rows, rows>::Identity(
        static_cast<Eigen::Index>(args)..., static_cast<Eigen::Index>(args)...);
    }


  };

} // namespace OpenKalman


#endif //OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
