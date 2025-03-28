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
 * \brief Definition of \ref collections::slice_view and \ref collections::views::slice.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP

#include <tuple>
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/classes/operation.hpp"
#include "values/functions/cast_to.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/functions/get.hpp"
#include "collection_view_interface.hpp"
#include "collections/concepts/internal/tuple_element_base.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view representing a slice of a \ref collection.
   * \details
   * The following should compile:
   * \code
   * \endcode
   * \tparam Offset The offset to the beginning of the slice
   * \tparam Extent The size of the slice
   * \sa views::slice
   */
#ifdef __cpp_lib_ranges
  template<collection T, value::index Offset, value::index Extent> requires (not tuple_like<T>) or
    ((value::dynamic<Offset> or value::fixed_number_of_v<Offset> <= std::tuple_size_v<std::decay_t<T>>) and
    (value::dynamic<Extent> or value::fixed_number_of_v<Extent> <= std::tuple_size_v<std::decay_t<T>>) and
    (value::dynamic<Offset> or value::dynamic<Extent> or value::fixed_number_of_v<Offset> + value::fixed_number_of_v<Extent> <= std::tuple_size_v<std::decay_t<T>>))
#else
  template<typename T, typename Offset, typename Extent>
#endif
  struct slice_view : collection_view_interface<slice_view<T, Offset, Extent>>
  {
#ifdef __cpp_concepts
    constexpr slice_view() requires std::default_initializable<T> and std::default_initializable<Offset> and std::default_initializable<Extent> = default;
#else
    template<typename aT = void, std::enable_if_t<std::is_void_v<aT> and std::is_default_constructible_v<T> and
      std::is_default_constructible_v<Offset> and std::is_default_constructible_v<Extent>, int> = 0>
    constexpr slice_view() {}
#endif


#ifdef __cpp_concepts
    template<typename Arg, value::index O, value::index E> requires
      std::constructible_from<T, Arg&&> and std::constructible_from<Offset, O&&> and std::constructible_from<Extent, E&&>
#else
    template<typename Arg, typename O, typename E, std::enable_if_t<std::is_constructible_v<T, Arg&&> and
      std::is_constructible_v<Offset, O&&> and std::is_constructible_v<Extent, E&&>, int> = 0>
#endif
    explicit constexpr slice_view(Arg&& arg, O&& o, E&& e) : my_t {std::forward<Arg>(arg)},
      my_offset{std::forward<O>(o)}, my_extent {std::forward<E>(e)} {}


#ifdef __cpp_explicit_this_parameter
    template<value::index I> requires (value::fixed<I> or sized_random_access_range<T>) and
      (value::dynamic<I> or value::dynamic<Extent> or value::fixed_number_of_v<I> < value::fixed_number_of_v<Extent>)
    constexpr decltype(auto)
    operator[](this auto&& self, I i)
    {
      auto global_i = value::operation{std::plus{}, std::move(i), std::forward<decltype(self)>(self).my_offset};
      return get(std::forward<decltype(self)>(self).my_t, std::move(global_i));
    }
#else
    template<typename I, std::enable_if_t<value::index<I> and (value::fixed<I> or sized_random_access_range<T>), int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr(value::fixed<I> and value::fixed<Extent>) static_assert(value::fixed_number_of_v<I> < value::fixed_number_of_v<Extent>);
      auto global_i = value::operation{std::plus{}, std::move(i), my_offset};
      return get(my_t, std::move(global_i));
    }


    template<typename I, std::enable_if_t<value::index<I> and (value::fixed<I> or sized_random_access_range<T>), int> = 0>
    constexpr decltype(auto)
    operator[](I i) &&
    {
      if constexpr(value::fixed<I> and value::fixed<Extent>) static_assert(value::fixed_number_of_v<I> < value::fixed_number_of_v<Extent>);
      auto global_i = value::operation{std::plus{}, std::move(i), std::move(*this).my_offset};
      return get(std::move(*this).my_t, std::move(global_i));
    }
#endif


#ifdef __cpp_explicit_this_parameter
    constexpr value::index auto
    size(this auto&& self)
    {
      return std::forward<decltype(self)>(self).my_extent;
    }
#else
    constexpr auto
    size() const &
    {
      return my_extent;
    }

    constexpr auto
    size() &&
    {
      return std::move(*this).my_extent;
    }
#endif

  private:

    T my_t {};

    Offset my_offset {};

    Extent my_extent {};

  };


  /**
   * \brief Deduction guide
   */
#ifdef __cpp_concepts
  template<typename Arg, value::index O, value::index E>
#else
  template<typename Arg, typename O, typename E, std::enable_if_t<value::index<O> and value::index<E>, int> = 0>
#endif
  slice_view(Arg&&, O&&, E&&) -> slice_view<Arg, O, E>;

} // namespace OpenKalman


namespace std
{
  template<typename T, typename O, typename E>
  struct tuple_size<OpenKalman::collections::slice_view<T, O, E>> : OpenKalman::value::fixed_number_of<E> {};

  template<size_t i, typename T, typename O, typename E>
  struct tuple_element<i, OpenKalman::collections::slice_view<T, O, E>>
      : OpenKalman::collections::internal::tuple_element_base<OpenKalman::value::operation<std::plus<>, O, std::integral_constant<std::size_t, i>>, T> {};
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct slice_impl
    {
#ifdef __cpp_concepts
      template<collection T, value::index O, value::index E>
#else
      template<typename T, typename O, typename E, std::enable_if_t<collection<T> and value::index<O> and value::index<E>, int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (T&& t, O&& o, E&& e) const { return slice_view {std::forward<T>(t), std::forward<O>(o), std::forward<E>(e)}; }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref slice_view.
   * \details The expression <code>views::slice(arg)</code> is expression-equivalent
   * to <code>slice_view(arg)</code> for any suitable \ref collection arg.
   * \sa slice_view
   */
  inline constexpr detail::slice_impl slice;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP
