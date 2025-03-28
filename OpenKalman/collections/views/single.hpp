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
 * \brief Definition of \ref collections::single_view and \ref collections::views::single.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_SINGLE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_SINGLE_HPP

#include <type_traits>
#include <tuple>
#include "collection_view_interface.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view that wraps an object and presents it as a one-element \ref collection.
   * \details If the base type is not itself a \ref collection, it will be presented as a singleton collection (only one component).
   * The following should compile:
   * \code
   * static_assert(single_view{4}[0] == 4);
   * static_assert(std::size_of_v<single_view<int>> == 1);
   * static_assert(equal_to{}(single_view{5}, std::vector{5}));
   * \endcode
   * \sa views::single
   */
  template<typename T>
  struct single_view : collection_view_interface<single_view<T>>
  {
#ifdef __cpp_concepts
    constexpr single_view() requires std::default_initializable<T> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr single_view() {}
#endif


#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<T, Arg&&>, int> = 0>
#endif
    explicit constexpr single_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


#ifdef __cpp_explicit_this_parameter
    template<value::index I> requires (value::dynamic<I> or value::fixed_number_of_v<I> == 0)
    constexpr decltype(auto)
    operator[](this auto&& self, I i)
    {
      if constexpr(value::dynamic<I>) if (i != 0) throw std::out_of_range("index to a single_view must be 0");
      return std::get<0>(std::forward<decltype(self)>(self).t);
    }
#else
    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr(value::fixed<I>) static_assert(value::fixed_number_of_v<I> == 0);
      else if (i != 0) throw std::out_of_range("index to a single_view must be 0");
      return std::get<0>(t);
    }


    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) &&
    {
      if constexpr(value::fixed<I>) static_assert(value::fixed_number_of_v<I> == 0);
      else if (i != 0) throw std::out_of_range("index to a single_view must be 0");
      return std::get<0>(std::move(*this).t);
    }
#endif


#ifdef __cpp_concepts
    constexpr value::index auto size() const
#else
    constexpr auto size() const
#endif
    {
      return std::integral_constant<std::size_t, 1> {};
    }

  private:

    std::tuple<T> t;
  };


  /**
   * \brief Deduction guide
   */
  template<typename Arg>
  single_view(Arg&&) -> single_view<Arg>;

} // namespace OpenKalman


namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::collections::single_view<T>> : integral_constant<std::size_t, 1> {};

  template<size_t i, typename T>
  struct tuple_element<i, OpenKalman::collections::single_view<T>> { static_assert(i == 0); using type = T; };
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct single_impl
    {
      template<typename T>
      constexpr auto
      operator() [[nodiscard]] (T&& t) const { return single_view {std::forward<T>(t)}; }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref single_view.
   * \details The expression <code>views::single(arg)</code> is expression-equivalent
   * to <code>single_view(arg)</code> for any suitable \ref collection arg.
   * \sa single_view
   */
  inline constexpr detail::single_impl single;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_SINGLE_HPP
