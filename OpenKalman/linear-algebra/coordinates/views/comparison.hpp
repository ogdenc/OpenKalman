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
 * \brief Definition of \ref coordinate::comparison_view and \ref coordinate::views::comparison.
 */

#ifndef OPENKALMAN_COORDINATES_VIEWS_COMPARISON_HPP
#define OPENKALMAN_COORDINATES_VIEWS_COMPARISON_HPP

#include "basics/concepts/sized_random_access_range.hpp"
#include "basics/functions/get_collection_size.hpp"
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_collection.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_collection_element.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \internal
   * \brief A view that wraps a \ref descriptor_collection and presents it as another \ref descriptor_collection with unaltered components.
   * \details
   * The following should compile:
   * \code
   * static_assert(equal_to{}(comparison_view{std::tuple{4, 5.}}, std::tuple{4, 5.}));
   * static_assert(equal_to{}(std::tuple{4, 5.}, comparison_view{std::tuple{4, 5.}}));
   * static_assert(equal_to{}(comparison_view{std::vector{4, 5, 6}}, std::vector{4, 5, 6}));
   * static_assert(equal_to{}(std::array{4, 5, 6}, comparison_view{std::array{4, 5, 6}}));
   * \endcode
   * \sa views::comparison
   */
#ifdef __cpp_lib_ranges
  template<descriptor_collection T>
#else
  template<typename T>
#endif
  struct comparison_view : collections::collection_view_interface<comparison_view<T>>
  {
#ifdef __cpp_concepts
    constexpr comparison_view() requires std::default_initializable<T> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr comparison_view() {}
#endif


#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<T, Arg&&>, int> = 0>
#endif
    explicit constexpr comparison_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


#ifdef __cpp_explicit_this_parameter
    template<value::index I> requires (value::fixed<I> or sized_random_access_range<T>) and
      (value::dynamic<I> or collections::size_of_v<T> == dynamic_size or (value::fixed_number_of_v<I> < collections::size_of_v<T>))
    constexpr decltype(auto)
    operator[](this auto&& self, I i)
    {
      return internal::get_descriptor_collection_element(std::forward<decltype(self)>(self).t, std::move(i));
    }
#else
    template<typename I, std::enable_if_t<value::index<I> and (value::fixed<I> or sized_random_access_range<T>), int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr(value::fixed<I> and collections::size_of_v<T> != dynamic_size) static_assert(value::fixed_number_of_v<I> < collections::size_of_v<T>);
      return internal::get_descriptor_collection_element(t, std::move(i));
    }


    template<typename I, std::enable_if_t<value::index<I> and (value::fixed<I> or sized_random_access_range<T>), int> = 0>
    constexpr decltype(auto)
    operator[](I i) &&
    {
      if constexpr(value::fixed<I> and collections::size_of_v<T> != dynamic_size) static_assert(value::fixed_number_of_v<I> < collections::size_of_v<T>);
      return internal::get_descriptor_collection_element(std::move(*this).t, std::move(i));
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
  comparison_view(Arg&&) -> comparison_view<Arg>;

} // namespace OpenKalman::coordinate


#ifndef __cpp_concepts
namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::coordinate::comparison_view<T>> : tuple_size<OpenKalman::collection_view_interface<OpenKalman::coordinate::comparison_view<T>>> {};

  template<size_t i, typename T>
  struct tuple_element<i, OpenKalman::coordinate::comparison_view<T>> : tuple_element<i, OpenKalman::collection_view_interface<OpenKalman::coordinate::comparison_view<T>>> {};
} // namespace std
#endif


namespace OpenKalman::coordinate::views
{
  namespace detail
  {
    struct comparison_impl
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<comparison_impl>
#endif
    {
#ifdef __cpp_concepts
      template<descriptor_collection R>
#else
      template<typename R, std::enable_if_t<descriptor_collection<R>, int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (R&& r) const { return comparison_view<R> {std::forward<R>(r)}; }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref comparison_view.
   * \details The expression <code>views::comparison(arg)</code> is expression-equivalent
   * to <code>comparison_view(arg)</code> for any suitable \ref collection arg.
   * \sa comparison_view
   */
  inline constexpr detail::comparison_impl comparison;


} // namespace OpenKalman::coordinate


#endif //OPENKALMAN_COORDINATES_VIEWS_COMPARISON_HPP
