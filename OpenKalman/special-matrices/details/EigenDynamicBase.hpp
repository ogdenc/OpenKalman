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
 * \brief Definitions for Eigen3::internal::EigenDynamicBase
 * \details This is the base class for library-defined dynamic Eigen matrices.
 */

#ifndef OPENKALMAN_EIGEN3_EIGENDYNAMICBASE_HPP
#define OPENKALMAN_EIGEN3_EIGENDYNAMICBASE_HPP

#include <type_traits>

namespace OpenKalman::Eigen3::internal
{
  /**
   * \internal
   * \brief Specialization for dynamic rows and Dynamic columns
   */
  template<typename Scalar>
  struct EigenDynamicBase<Scalar, dynamic_extent, dynamic_extent>
  {
    /**
     * \internal
     * \brief Construct an EigenDynamicBase with dynamic rows and dynamic columns.
     * \param r Number of rows.
     * \param c Number of columns.
     */
    EigenDynamicBase(std::size_t r, std::size_t c) : m_rows {r}, m_cols {c} {}


    /**
     * \internal
     * \brief Construct an EigenDynamicBase based on the shape of another matrix M.
     * \tparam M The matrix to be used as a shape template. M must have a compatible shape.
     */
#ifdef __cpp_concepts
    template<typename M>
    requires (not std::same_as<M, EigenDynamicBase>)
#else
    template<typename M, std::enable_if_t<not std::is_same_v<M, EigenDynamicBase>, int> = 0>
#endif
    EigenDynamicBase(M&& m) : m_rows {row_count(m)}, m_cols {column_count(m)} {}


    EigenDynamicBase(const EigenDynamicBase&) = default;
    EigenDynamicBase(EigenDynamicBase&&) = default;


    /**
     * \internal
     * \brief Copy constructor
     */
    EigenDynamicBase& operator=(const EigenDynamicBase& m)
    {
      assert(m.m_rows == m_rows);
      assert(m.m_cols == m_cols);
      return *this;
    }


    /**
     * \internal
     * \brief Move constructor
     */
    EigenDynamicBase& operator=(EigenDynamicBase&& m)
    {
      assert(m.m_rows == m_rows);
      assert(m.m_cols == m_cols);
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another compatible EigenDynamicBase.
     */
#ifdef __cpp_concepts
    template<typename M>
    requires (not std::same_as<M, EigenDynamicBase>)
#else
    template<typename M, std::enable_if_t<not std::is_same_v<M, EigenDynamicBase>, int> = 0>
#endif
    auto& operator=(M&& m)
    {
      assert(row_count(m) == m_rows);
      assert(column_count(m) == m_cols);
      return *this;
    }


    /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
    Eigen::Index rows() const { return m_rows; }


    /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
    Eigen::Index cols() const { return m_cols; }


    /**
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    static decltype(auto) zero(const std::size_t r, const std::size_t c)
    {
      return ZeroMatrix<Scalar, dynamic_extent, dynamic_extent> {r, c};
    }


    /**
     * \brief Synonym for zero().
     * \deprecated Use zero() instead. Provided for compatibility with Eigen Zero() member.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    [[deprecated("Use zero() instead.")]]
    static decltype(auto) Zero(const Eigen::Index r, Eigen::Index c)
    {
      return zero(r, c);
    }

  private:

    const std::size_t m_rows;
    const std::size_t m_cols;

  };


  // ----------------------------- //


  /**
   * \overload \internal
   * \brief Specialization for dynamic rows and fixed columns
   */
#ifdef __cpp_concepts
  template<typename Scalar, std::size_t columns>
  requires (columns != dynamic_extent)
  struct EigenDynamicBase<Scalar, dynamic_extent, columns>
#else
  template<typename Scalar, std::size_t columns>
  struct EigenDynamicBase<Scalar, dynamic_extent, columns, std::enable_if_t<(columns != dynamic_extent)>>
#endif
  {
    /**
     * \internal
     * \brief Construct an EigenDynamicBase with dynamic rows and dynamic columns.
     * \param r Number of rows.
     */
    EigenDynamicBase(std::size_t r) : m_rows {r} {}


    /**
     * \internal
     * \brief Construct an EigenDynamicBase based on the shape of another matrix M.
     * \tparam M The matrix to be used as a shape template. M must have a compatible shape.
     */
#ifdef __cpp_concepts
    template<typename M>
    requires (not std::same_as<M, EigenDynamicBase>) and
      (dynamic_columns<M> or column_extent_of_v<M> == columns)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_columns<M> or column_extent_of<M>::value == columns), int> = 0>
#endif
    EigenDynamicBase(M&& m) : m_rows {row_count(m)}
    {
      if constexpr (dynamic_columns<M>) assert(column_count(m) == columns);
    }


    EigenDynamicBase(const EigenDynamicBase&) = default;
    EigenDynamicBase(EigenDynamicBase&&) = default;


    /**
     * \internal
     * \brief Copy constructor
     */
    EigenDynamicBase& operator=(const EigenDynamicBase& m)
    {
      assert(m.m_rows == m_rows);
      return *this;
    }


    /**
     * \internal
     * \brief Move constructor
     */
    EigenDynamicBase& operator=(EigenDynamicBase&& m)
    {
      assert(m.m_rows == m_rows);
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another compatible EigenDynamicBase.
     */
#ifdef __cpp_concepts
    template<typename M>
    requires (not std::same_as<M, EigenDynamicBase>) and
      (dynamic_columns<M> or column_extent_of_v<M> == columns)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_columns<M> or column_extent_of<M>::value == columns), int> = 0>
#endif
    auto& operator=(M&& m)
    {
      if constexpr (dynamic_columns<M>) assert(column_count(m) == columns);
      assert(row_count(m) == m_rows);
      return *this;
    }


    /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
    Eigen::Index rows() const { return m_rows; }


    /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
    static constexpr Eigen::Index cols() { return columns; }


    /**
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    static decltype(auto) zero(const std::size_t r)
    {
      return ZeroMatrix<Scalar, dynamic_extent, columns> {r};
    }


    /**
     * \brief Synonym for zero().
     * \deprecated Use zero() instead. Provided for compatibility with Eigen Zero() member.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    [[deprecated("Use zero() instead.")]]
    static decltype(auto) Zero(const Eigen::Index r, Eigen::Index c)
    {
      assert(c == columns);
      return zero(r);
    }

  private:

    const std::size_t m_rows;

  };


  // ----------------------------- //


  /**
   * \overload \internal
   * \brief Specialization for fixed rows and dynamic columns
   */
#ifdef __cpp_concepts
  template<typename Scalar, std::size_t rows_>
  requires (rows_ != dynamic_extent)
  struct EigenDynamicBase<Scalar, rows_, dynamic_extent>
#else
  template<typename Scalar, std::size_t rows_>
    struct EigenDynamicBase<Scalar, rows_, dynamic_extent, std::enable_if_t<(rows_ != dynamic_extent)>>
#endif
  {
    /**
     * \internal
     * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
     * \param c Number of columns.
     */
    EigenDynamicBase(std::size_t c) : m_cols {c} {}


    /**
     * \internal
     * \brief Construct an EigenDynamicBase based on the shape of another matrix M.
     * \tparam M The matrix to be used as a shape template. M must have a compatible shape.
     */
#ifdef __cpp_concepts
    template<typename M>
    requires (not std::same_as<M, EigenDynamicBase>) and (dynamic_rows<M> or row_extent_of_v<M> == rows_)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_rows<M> or row_extent_of<M>::value == rows_), int> = 0>
#endif
    EigenDynamicBase(M&& m) : m_cols {column_count(m)}
    {
      if constexpr (dynamic_rows<M>) assert(row_count(m) == rows_);
    }


    EigenDynamicBase(const EigenDynamicBase&) = default;
    EigenDynamicBase(EigenDynamicBase&&) = default;


    /**
     * \internal
     * \brief Copy constructor
     */
    EigenDynamicBase& operator=(const EigenDynamicBase& m)
    {
      assert(m.m_cols == m_cols);
      return *this;
    }


    /**
     * \internal
     * \brief Move constructor
     */
    EigenDynamicBase& operator=(EigenDynamicBase&& m)
    {
      assert(m.m_cols == m_cols);
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another compatible EigenDynamicBase.
     */
#ifdef __cpp_concepts
    template<typename M>
    requires (not std::same_as<M, EigenDynamicBase>) and (dynamic_rows<M> or row_extent_of_v<M> == rows_)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_rows<M> or row_extent_of<M>::value == rows_), int> = 0>
#endif
    auto& operator=(M&& m)
    {
      if constexpr (dynamic_rows<M>) assert(row_count(m) == rows_);
      assert(column_count(m) == m_cols);
      return *this;
    }


    /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
    static constexpr Eigen::Index rows() { return rows_; }


    /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
    Eigen::Index cols() const { return m_cols; }


    /**
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    static decltype(auto) zero(const std::size_t c)
    {
      return ZeroMatrix<Scalar, rows_, dynamic_extent> {c};
    }


    /**
     * \brief Synonym for zero().
     * \deprecated Use zero() instead. Provided for compatibility with Eigen Zero() member.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    [[deprecated("Use zero() instead.")]]
    static decltype(auto) Zero(const Eigen::Index r, Eigen::Index c)
    {
      assert(r == rows_);
      return zero(c);
    }

  private:

    const std::size_t m_cols;

  };


  // ----------------------------- //


  /**
   * \overload \internal
   * \brief Specialization for fixed rows and fixed columns
   */
#ifdef __cpp_concepts
  template<typename Scalar, std::size_t rows_, std::size_t columns>
  requires (rows_ != dynamic_extent) and (columns != dynamic_extent)
  struct EigenDynamicBase<Scalar, rows_, columns>
#else
  template<typename Scalar, std::size_t rows_, std::size_t columns>
  struct EigenDynamicBase<Scalar, rows_, columns, std::enable_if_t<
    (rows_ != dynamic_extent) and (columns != dynamic_extent)>>
#endif
  {
    /**
     * \internal
     * \internal
     * \brief Default constructor.
     */
    EigenDynamicBase() {};


    /**
     * \internal
     * \brief Construct an EigenDynamicBase based on the shape of another matrix M.
     * \tparam M The matrix to be used as a shape template. M must have a compatible shape.
     */
#ifdef __cpp_concepts
    template<typename M>
    requires (not std::same_as<M, EigenDynamicBase>) and (dynamic_rows<M> or row_extent_of_v<M> == rows_) and
      (dynamic_columns<M> or column_extent_of_v<M> == columns)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_rows<M> or row_extent_of<M>::value == rows_) and
      (dynamic_columns<M> or column_extent_of<M>::value == columns), int> = 0>
#endif
    EigenDynamicBase(M&& m)
    {
      if constexpr (dynamic_rows<M>) assert(row_count(m) == rows_);
      if constexpr (dynamic_columns<M>) assert(column_count(m) == columns);
    }


    /**
     * \internal
     * \brief Assign from another compatible EigenDynamicBase.
     */
#ifdef __cpp_concepts
    template<typename M>
    requires (not std::same_as<M, EigenDynamicBase>) and (dynamic_rows<M> or row_extent_of_v<M> == rows_) and
      (dynamic_columns<M> or column_extent_of_v<M> == columns)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_rows<M> or row_extent_of<M>::value == rows_) and
      (dynamic_columns<M> or column_extent_of<M>::value == columns), int> = 0>
#endif
    auto& operator=(M&& m)
    {
      if constexpr (dynamic_rows<M>) assert(row_count(m) == rows_);
      if constexpr (dynamic_columns<M>) assert(column_count(m) == columns);
      return *this;
    }


    /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
    static constexpr Eigen::Index rows() { return rows_; }


    /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
    static constexpr Eigen::Index cols() { return columns; }


    /**
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    static decltype(auto) zero()
    {
      return ZeroMatrix<Scalar, rows_, columns> {};
    }


    /**
     * \brief Synonym for zero().
     * \deprecated Use zero() instead. Provided for compatibility with Eigen Zero() member.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    [[deprecated("Use zero() instead.")]]
    static decltype(auto) Zero()
    {
      return zero();
    }

  };


} // OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN3_EIGENDYNAMICBASE_HPP
