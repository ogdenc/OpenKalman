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
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    dynamic_rows<NestedMatrix> and dynamic_columns<NestedMatrix>
  struct EigenDynamicBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct EigenDynamicBase<Derived, NestedMatrix, std::enable_if_t<
    dynamic_rows<NestedMatrix> and dynamic_columns<NestedMatrix>>>
#endif
    : MatrixTraits<NestedMatrix>::template MatrixBaseFrom<Derived>
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
    EigenDynamicBase(M&& m) : m_rows {runtime_dimension_of<0>(m)}, m_cols {runtime_dimension_of<1>(m)} {}


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
      assert(runtime_dimension_of<0>(m) == m_rows);
      assert(runtime_dimension_of<1>(m) == m_cols);
      return *this;
    }


  private:

    friend struct IndexTraits<Derived, 0>;
    friend struct IndexTraits<Derived, 1>;


    /// \internal \return The number of fixed rows. \note Used by IndexTraits::dimension_at_runtime.
    Eigen::Index get_rows_at_runtime() const { return m_rows; }


    /// \internal \return The number of fixed columns. \note Used by IndexTraits::dimension_at_runtime.
    Eigen::Index get_columns_at_runtime() const { return m_cols; }


    const std::size_t m_rows;
    const std::size_t m_cols;

  };


  // ----------------------------- //


  /**
   * \overload \internal
   * \brief Specialization for dynamic rows and fixed columns
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    dynamic_rows<NestedMatrix> and (not dynamic_columns<NestedMatrix>)
  struct EigenDynamicBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct EigenDynamicBase<Derived, NestedMatrix, std::enable_if_t<
    dynamic_rows<NestedMatrix> and (not dynamic_columns<NestedMatrix>)>>
#endif
    : MatrixTraits<NestedMatrix>::template MatrixBaseFrom<Derived>
  {
    private:

      static constexpr auto nested_cols = column_dimension_of_v<NestedMatrix>;

    public:

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
      (dynamic_columns<M> or column_dimension_of_v<M> == nested_cols)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_columns<M> or column_dimension_of<M>::value == nested_cols), int> = 0>
#endif
    EigenDynamicBase(M&& m) : m_rows {runtime_dimension_of<0>(m)}
    {
      if constexpr (dynamic_columns<M>) assert(runtime_dimension_of<1>(m) == nested_cols);
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
      (dynamic_columns<M> or column_dimension_of_v<M> == nested_cols)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_columns<M> or column_dimension_of<M>::value == nested_cols), int> = 0>
#endif
    auto& operator=(M&& m)
    {
      if constexpr (dynamic_columns<M>) assert(runtime_dimension_of<1>(m) == nested_cols);
      assert(runtime_dimension_of<0>(m) == m_rows);
      return *this;
    }

  private:

    friend struct IndexTraits<Derived, 0>;


    /// \internal \return The number of fixed rows. \note Used by IndexTraits::dimension_at_runtime.
    Eigen::Index get_rows_at_runtime() const { return m_rows; }


    const std::size_t m_rows;

  };


  // ----------------------------- //


  /**
   * \overload \internal
   * \brief Specialization for fixed rows and dynamic columns
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    (not dynamic_rows<NestedMatrix>) and dynamic_columns<NestedMatrix>
  struct EigenDynamicBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct EigenDynamicBase<Derived, NestedMatrix, std::enable_if_t<
    (not dynamic_rows<NestedMatrix>) and dynamic_columns<NestedMatrix>>>
#endif
    : MatrixTraits<NestedMatrix>::template MatrixBaseFrom<Derived>
  {
  private:

    static constexpr auto nested_rows = row_dimension_of_v<NestedMatrix>;

  public:

    /**
     * \internal
     * \brief Construct an EigenDynamicBase with dynamic rows and dynamic columns.
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
    requires (not std::same_as<M, EigenDynamicBase>) and (dynamic_rows<M> or row_dimension_of_v<M> == nested_rows)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_rows<M> or row_dimension_of<M>::value == nested_rows), int> = 0>
#endif
    EigenDynamicBase(M&& m) : m_cols {runtime_dimension_of<1>(m)}
    {
      if constexpr (dynamic_rows<M>) assert(runtime_dimension_of<0>(m) == nested_rows);
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
    requires (not std::same_as<M, EigenDynamicBase>) and (dynamic_rows<M> or row_dimension_of_v<M> == nested_rows)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_rows<M> or row_dimension_of<M>::value == nested_rows), int> = 0>
#endif
    auto& operator=(M&& m)
    {
      if constexpr (dynamic_rows<M>) assert(runtime_dimension_of<0>(m) == nested_rows);
      assert(runtime_dimension_of<1>(m) == m_cols);
      return *this;
    }

  private:

    friend struct IndexTraits<Derived, 1>;


    /// \internal \return The number of fixed columns. \note Used by IndexTraits::dimension_at_runtime.
    Eigen::Index get_columns_at_runtime() const { return m_cols; }


    const std::size_t m_cols;

  };


  // ----------------------------- //


  /**
   * \overload \internal
   * \brief Specialization for fixed nested_rows and fixed columns
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    (not dynamic_rows<NestedMatrix>) and (not dynamic_columns<NestedMatrix>)
  struct EigenDynamicBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct EigenDynamicBase<Derived, NestedMatrix, std::enable_if_t<
    (not dynamic_rows<NestedMatrix>) and (not dynamic_columns<NestedMatrix>)>>
#endif
    : MatrixTraits<NestedMatrix>::template MatrixBaseFrom<Derived>
  {
  private:

    static constexpr auto nested_rows = row_dimension_of_v<NestedMatrix>;
    static constexpr auto nested_cols = column_dimension_of_v<NestedMatrix>;

  public:

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
    requires (not std::same_as<M, EigenDynamicBase>) and (dynamic_rows<M> or row_dimension_of_v<M> == nested_rows) and
      (dynamic_columns<M> or column_dimension_of_v<M> == nested_cols)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_rows<M> or row_dimension_of<M>::value == nested_rows) and
      (dynamic_columns<M> or column_dimension_of<M>::value == nested_cols), int> = 0>
#endif
    EigenDynamicBase(M&& m)
    {
      if constexpr (dynamic_rows<M>) assert(runtime_dimension_of<0>(m) == nested_rows);
      if constexpr (dynamic_columns<M>) assert(runtime_dimension_of<1>(m) == nested_cols);
    }


    /**
     * \internal
     * \brief Assign from another compatible EigenDynamicBase.
     */
#ifdef __cpp_concepts
    template<typename M>
    requires (not std::same_as<M, EigenDynamicBase>) and (dynamic_rows<M> or row_dimension_of_v<M> == nested_rows) and
      (dynamic_columns<M> or column_dimension_of_v<M> == nested_cols)
#else
    template<typename M, std::enable_if_t<(not std::is_same_v<M, EigenDynamicBase>) and
      (dynamic_rows<M> or row_dimension_of<M>::value == nested_rows) and
      (dynamic_columns<M> or column_dimension_of<M>::value == nested_cols), int> = 0>
#endif
    auto& operator=(M&& m)
    {
      if constexpr (dynamic_rows<M>) assert(runtime_dimension_of<0>(m) == nested_rows);
      if constexpr (dynamic_columns<M>) assert(runtime_dimension_of<1>(m) == nested_cols);
      return *this;
    }

  };


} // OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN3_EIGENDYNAMICBASE_HPP
