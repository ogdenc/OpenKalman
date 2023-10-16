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
 * \internal
 * \file
 * \brief Definition of MatrixBase.
 */

#ifndef OPENKALMAN_MATRIXBASE_HPP
#define OPENKALMAN_MATRIXBASE_HPP


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Ultimate base of typed matrices and covariance matrices.
   * \tparam Derived The fully derived matrix type.
   * \tparam NestedMatrix The nested native matrix, which can be const or an lvalue reference, or both, or neither.
   */
#ifdef __cpp_concepts
  template<indexible Derived, indexible NestedMatrix> requires (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Derived, typename NestedMatrix>
#endif
  struct MatrixBase : internal::library_base_t<Derived, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif


    /**
     * \internal
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr MatrixBase() noexcept requires self_contained<NestedMatrix> and std::default_initializable<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<self_contained<T> and
      std::is_default_constructible<NestedMatrix>::value, int> = 0>
    constexpr MatrixBase() noexcept
#endif
      : m_arg {} {}


    /**
     * \internal
     * \brief Construct from a nestable type.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<typename Arg> requires (not std::is_base_of_v<MatrixBase, std::decay_t<Arg>>) and
    std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<not std::is_base_of_v<MatrixBase, std::decay_t<Arg>> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    constexpr explicit MatrixBase(Arg&& arg) noexcept : m_arg {std::forward<Arg>(arg)} {}


    /** \internal
     * \brief Assign from a nestable type.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::derived_from<std::decay_t<Arg>, MatrixBase>)
#else
    template<typename Arg, std::enable_if_t<not std::is_base_of_v<MatrixBase, std::decay_t<Arg>>, int> = 0>
#endif
    constexpr auto& operator=(Arg&& arg) noexcept
    {
      static_assert(not std::is_const_v<std::remove_reference_t<NestedMatrix>>, "Nested matrix cannot be modified.");
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        m_arg = std::forward<Arg>(arg);
      }
      return *this;
    }


    /**
     * \brief Get the nested matrix.
     */
    constexpr decltype(auto) nested_matrix() & noexcept { return (m_arg); }


    /// \overload
    constexpr decltype(auto) nested_matrix() const & noexcept { return (m_arg); }


    /// \overload
    constexpr decltype(auto) nested_matrix() && noexcept { return (std::move(*this).m_arg); }


    /// \overload
    constexpr decltype(auto) nested_matrix() const && noexcept { return (std::move(*this).m_arg); }


    /**
     * Access the coefficient at row i and column j
     * \param i The row.
     * \param j The column.
     * \return If <code>element_settable<Derived&, 2></code>, the element is settable. Therefore,
     * this function returns an object that can be assigned the coefficient to be set.
     * Otherwise, it will return the (non-settable) coefficient as a value.
     */
    constexpr auto operator()(std::size_t i, std::size_t j) &
    {
      static_assert(element_gettable<Derived&, 2>);

      if constexpr (element_settable<Derived&, 2>)
        return ElementAccessor(static_cast<Derived&>(*this), i, j);
      else
        return get_element(static_cast<Derived&>(*this), i, j);
    }


    /// \overload
    constexpr auto operator()(std::size_t i, std::size_t j) &&
    {
      static_assert(element_gettable<Derived&&, 2>);

      if constexpr (element_settable<Derived&&, 2>)
        return ElementAccessor(static_cast<Derived&&>(*this), i, j);
      else
        return get_element(static_cast<Derived&&>(*this), i, j);
    }


    /// \overload
    constexpr auto operator()(std::size_t i, std::size_t j) const &
    {
      static_assert(element_gettable<const Derived&, 2>);

      return get_element(static_cast<const Derived&>(*this), i, j);
    }


    /// \overload
    constexpr auto operator()(std::size_t i, std::size_t j) const &&
    {
      static_assert(element_gettable<const Derived&&, 2>);

      return get_element(static_cast<const Derived&&>(*this), i, j);
    }


    /**
     * Access the coefficient at row i
     * \param i The row.
     * \return If <code>element_settable<Derived&, 1></code>, the element is settable. Therefore,
     * this function returns an object that can be assigned the coefficient to be set.
     * Otherwise, it will return the (non-settable) coefficient as a value.
     */
    constexpr auto operator[](std::size_t i) &
    {
      if constexpr (element_settable<Derived&, 1>)
        return ElementAccessor(static_cast<Derived&>(*this), i);
      else if constexpr (diagonal_matrix<Derived> and element_settable<Derived&, 2>)
        return ElementAccessor(static_cast<Derived&>(*this), i, i);
      else
      {
        if constexpr (element_gettable<Derived&, 1>)
          return get_element(static_cast<Derived&>(*this), i);
        else
        {
          static_assert(diagonal_matrix<Derived> and element_gettable<Derived&, 2>);
          return get_element(static_cast<Derived&>(*this), i, i);
        }
      }
    }


    /// \overload
    constexpr auto operator[](std::size_t i) &&
    {
      if constexpr (element_settable<Derived&&, 1>)
        return ElementAccessor(static_cast<Derived&&>(*this), i);
      else if constexpr (diagonal_matrix<Derived> and element_settable<Derived&&, 2>)
        return ElementAccessor(static_cast<Derived&&>(*this), i, i);
      else
      {
        if constexpr (element_gettable<Derived&&, 1>)
          return get_element(static_cast<Derived&&>(*this), i);
        else
        {
          static_assert(diagonal_matrix<Derived> and element_gettable<Derived&&, 2>);
          return get_element(static_cast<Derived&&>(*this), i, i);
        }
      }
    }


    /// \overload
    constexpr auto operator[](std::size_t i) const &
    {
      if constexpr (element_gettable<const Derived&, 1>)
        return get_element(static_cast<const Derived&>(*this), i);
      else
      {
        static_assert(diagonal_matrix<Derived> and element_gettable<const Derived&, 2>);
        return get_element(static_cast<const Derived&>(*this), i, i);
      }
    }


    /// \overload
    constexpr auto operator[](std::size_t i) const &&
    {
      if constexpr (element_gettable<const Derived&&, 1>)
        return get_element(static_cast<const Derived&&>(*this), i);
      else
      {
        static_assert(diagonal_matrix<Derived> and element_gettable<const Derived&&, 2>);
        return get_element(static_cast<const Derived&&>(*this), i, i);
      }
    }


    /// \overload
    constexpr auto operator()(std::size_t i) { return operator[](i); }


    /// \overload
    constexpr auto operator()(std::size_t i) const { return operator[](i); }

  private:

    NestedMatrix m_arg; //< The nested matrix.

  };

}

#endif //OPENKALMAN_BASEMATRIX_HPP
