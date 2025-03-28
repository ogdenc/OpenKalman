/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref collection_view_interface
 */

#ifndef OPENKALMAN_COLLECTION_VIEW_HPP
#define OPENKALMAN_COLLECTION_VIEW_HPP

#include <type_traits>
#include <tuple>
#include "values/concepts/fixed.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "collections/concepts/tuple_like.hpp"
#include "collections/concepts/sized_random_access_range.hpp"

namespace OpenKalman::collections
{
#if not defined(__cpp_explicit_this_parameter) or not defined(_cpp_concepts)
  namespace detail
  {
    template<typename D, typename = void>
    struct derived_size_of : std::integral_constant<std::size_t, dynamic_size> {};

    template<typename D>
    struct derived_size_of<D, std::enable_if_t<value::fixed<decltype(size(std::declval<D>()))>>>
      : value::fixed_number_of<decltype(size(std::declval<D>()))> {};
  } // namespace detail
#endif


  /**
   * \internal
   * \brief A CRTP helper class template for defining a view of a \ref collection.
   * \details The derived class must define at least the following:
   * \code
   * constexpr decltype(auto) operator[](value::index auto i) const;
   * constexpr value::index auto size() const;
   * \endcode
   * If the <code>size</code> function returns a \rev value::fixed "fixed" index and the subscript <code>operator[]</code> function
   * returns a value when \ref value::index "index" i is \ref value::fixed "fixed", the resulting view will be \ref tuple_like.
   * Independently of this, if the subscript <code>operator[]</code> function returns a value when
   * \ref value::index "index" i is \ref value::dynamic "dynamic", the resulting view will be a \ref sized_random_access_range.
   * \tparam Derived The derived view class.
   */
  template<typename Derived>
#ifdef __cpp_lib_ranges
  struct collection_view_interface : std::ranges::view_interface<Derived>
#else
  struct collection_view_interface : ranges::view_interface<Derived>
#endif
  {
    static_assert(std::is_class_v<Derived> && std::is_same_v<Derived, std::decay_t<Derived>>);


    /**
     * \brief Get element i the derived object, assuming it is a fixed-size range.
     * \details This effectively makes a range \ref tuple_like.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i, typename Self> requires (size_of_v<Derived> != dynamic_size) and (i < size_of_v<Derived>)
    constexpr decltype(auto)
    get(this Self&& self)
    {
      return collections::get(std::forward<Self>(self), std::integral_constant<std::size_t, i>{});
    }
#else
    template<std::size_t i, std::enable_if_t<(detail::derived_size_of<Derived>::value != dynamic_size) and
      (i < detail::derived_size_of<Derived>::value), int> = 0>
    constexpr decltype(auto)
    get() &
    {
      return collections::get(static_cast<Derived&>(*this), std::move(i));
    }

    template<std::size_t i, std::enable_if_t<(detail::derived_size_of<Derived>::value != dynamic_size) and
      (i < detail::derived_size_of<Derived>::value), int> = 0>
    constexpr decltype(auto)
    get() const &
    {
      return collections::get(static_cast<const Derived&>(*this), std::move(i));
    }

    template<std::size_t i, std::enable_if_t<(detail::derived_size_of<Derived>::value != dynamic_size) and
      (i < detail::derived_size_of<Derived>::value), int> = 0>
    constexpr decltype(auto)
    get() &&
    {
      return collections::get(static_cast<Derived&&>(*this), std::move(i));
    }

    template<std::size_t i, std::enable_if_t<(detail::derived_size_of<Derived>::value != dynamic_size) and
      (i < detail::derived_size_of<Derived>::value), int> = 0>
    constexpr decltype(auto)
    get() const &&
    {
      return collections::get(static_cast<const Derived&&>(*this), std::move(i));
    }
#endif


    /**
     * \brief Subscript operator
     */
#ifdef __cpp_explicit_this_parameter
    template<typename Self, value::index I> requires value::dynamic<I> or (size_of_v<Derived> == dynamic_size) or
      (value::fixed_number_of_v<I> < size_of_v<Derived>)
    constexpr decltype(auto)
    operator[](this Self&& self, I i)
    {
      return collections::get(std::forward<Self>(self), std::move(i));
    }
#else
    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) &
    {
      if constexpr (value::fixed<I> and detail::derived_size_of<Derived>::value != dynamic_size)
        static_assert(value::fixed_number_of<I>::value < detail::derived_size_of<Derived>::value);
      return collections::get(static_cast<Derived&>(*this), std::move(i));
    }

    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr (value::fixed<I> and detail::derived_size_of<Derived>::value != dynamic_size)
        static_assert(value::fixed_number_of<I>::value < detail::derived_size_of<Derived>::value);
      return collections::get(static_cast<const Derived&>(*this), std::move(i));
    }

    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) &&
    {
      if constexpr (value::fixed<I> and detail::derived_size_of<Derived>::value != dynamic_size)
        static_assert(value::fixed_number_of<I>::value < detail::derived_size_of<Derived>::value);
      return collections::get(static_cast<Derived&&>(*this), std::move(i));
    }

    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &&
    {
      if constexpr (value::fixed<I> and detail::derived_size_of<Derived>::value != dynamic_size)
        static_assert(value::fixed_number_of<I>::value < detail::derived_size_of<Derived>::value);
      return collections::get(static_cast<const Derived&&>(*this), std::move(i));
    }
#endif

  };


} // namespace std


#endif //OPENKALMAN_COLLECTION_VIEW_HPP
