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
 * \brief Definition of \ref collections::reverse_view and \ref collections::views::reverse.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_REVERSE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_REVERSE_HPP

#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/classes/operation.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/functions/get.hpp"
#include "collections/views/internal/tuple_size_base.hpp"
#include "collections/views/internal/tuple_element_base.hpp"
#include "collection_view_interface.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view that wraps a \ref collection and presents it as a \ref collection in reverse order.
   * \details
   * The following should compile:
   * \code
   * static_assert(std::tuple_size_v<reverse_view<std::tuple<int, double>>> == 2);
   * static_assert(std::tuple_size_v<reverse_view<std::tuple<>>> == 0);
   * static_assert(std::is_same_v<std::tuple_element_t<0, reverse_view<std::tuple<float, int, double>>>, double>);
   * static_assert(std::is_same_v<std::tuple_element_t<1, reverse_view<std::tuple<float, int, double>>>, int>);
   * static_assert(std::is_same_v<std::tuple_element_t<2, reverse_view<std::tuple<float, int, double>>>, float>);
   * static_assert(collections::get<0>(reverse_view {std::tuple{4, 5.}}) == 5.);
   * static_assert(collections::get<1>(reverse_view {std::tuple{4, 5.}}) == 4);
   * static_assert(collections::get<0>(reverse_view {std::tuple{4, std::monostate{}}}) == std::monostate{});
   * static_assert((reverse_view {std::vector{3, 4, 5}}[0u]) == 5);
   * static_assert((reverse_view {std::vector{3, 4, 5}}[1u]) == 4);
   * static_assert((reverse_view {std::vector{3, 4, 5}}[2u]) == 3);
   * \endcode
   * \sa views::reverse
   */
#ifdef __cpp_lib_ranges
  template<collection T>
#else
  template<typename T>
#endif
  struct reverse_view : collection_view_interface<reverse_view<T>>
  {
#ifdef __cpp_concepts
    constexpr reverse_view() requires std::default_initializable<T> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr reverse_view() {}
#endif


#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<T, Arg&&>, int> = 0>
#endif
    explicit constexpr reverse_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


#ifdef __cpp_explicit_this_parameter
    template<value::index I> requires (value::fixed<I> or sized_random_access_range<T>) and
      (value::dynamic<I> or size_of_v<T> == dynamic_size or value::fixed_number_of_v<I> < size_of_v<T>)
    constexpr decltype(auto)
    operator[](this auto&& self, I i)
    {
      return collections::get(std::forward<decltype(self)>(self).t,
        value::operation {std::minus{}, value::operation {std::minus{}, get_collection_size(self.t), std::move(i)}, std::integral_constant<std::size_t, 1>{}});
    }
#else
    template<typename I, std::enable_if_t<value::index<I> and (value::fixed<I> or sized_random_access_range<T>), int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr(value::fixed<I> and size_of_v<T> != dynamic_size) static_assert(value::fixed_number_of_v<I> < size_of_v<T>);
      return get(t, value::operation {std::minus{}, value::operation {std::minus{}, get_collection_size(t), std::move(i)}, std::integral_constant<std::size_t, 1>{}});
    }


    template<typename I, std::enable_if_t<value::index<I> and (value::fixed<I> or sized_random_access_range<T>), int> = 0>
    constexpr decltype(auto)
    operator[](I i) &&
    {
      if constexpr(value::fixed<I> and size_of_v<T> != dynamic_size) static_assert(value::fixed_number_of_v<I> < size_of_v<T>);
      return get(std::move(*this).t, value::operation {std::minus{}, value::operation {std::minus{}, get_collection_size(t), std::move(i)}, std::integral_constant<std::size_t, 1>{}});
    }
#endif


#ifdef __cpp_concepts
    constexpr value::index auto size() const
#else
    constexpr auto size() const
#endif
    {
      return get_collection_size(t);
    }

  private:

    T t;
  };


  /**
   * \brief Deduction guide
   */
  template<typename Arg>
  reverse_view(Arg&&) -> reverse_view<Arg>;

} // namespace OpenKalman::collections


namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::collections::reverse_view<T>> : OpenKalman::collections::internal::tuple_size_base<T> {};

  template<size_t i, typename T>
  struct tuple_element<i, OpenKalman::collections::reverse_view<T>>
    : OpenKalman::collections::internal::tuple_element_base<
        OpenKalman::value::operation<std::minus<>, tuple_size<std::decay_t<T>>, integral_constant<std::size_t, i + 1_uz>>, T> {};
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct reverse_impl
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<reverse_impl>
#endif
    {
#ifdef __cpp_concepts
      template<collection R>
#else
      template<typename R, std::enable_if_t<collection<R>, int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (R&& r) const { return reverse_view {std::forward<R>(r)}; }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref reverse_view.
   * \details The expression <code>views::reverse(arg)</code> is expression-equivalent
   * to <code>reverse_view(arg)</code> for any suitable \ref collection arg.
   * \sa reverse_view
   */
  inline constexpr detail::reverse_impl reverse;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_REVERSE_HPP
