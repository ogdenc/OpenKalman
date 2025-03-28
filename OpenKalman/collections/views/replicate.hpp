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
 * \brief Definition of \ref collections::replicate_view and \ref views::replicate.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_REPLICATE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_REPLICATE_HPP

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
   * \brief A view that replicates a \ref collection some number of times.
   * \details
   * The following should compile:
   * \code
   * static_assert(std::tuple_size_v<replicate_view<std::tuple<int, double>, std::integral_constant<std::size_t, 3>>> == 6);
   * static_assert(std::is_same_v<std::tuple_element_t<5, replicate_view<std::tuple<double, int, float>, std::integral_constant<std::size_t, 2>>>, float>);
   * static_assert(get<3>(replicate_view {std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}}) == 5);
   * static_assert((replicate_view {std::vector{3, 4, 5}, 2u}[4]), 4);
   * static_assert((replicate_view {std::vector{3, 4, 5}, 2u}[std::integral_constant<std::size_t, 5>{}]), 5);
   * \endcode
   * \sa views::replicate
   */
#ifdef __cpp_lib_ranges
  template<collection T, value::index Factor>
#else
  template<typename T, typename Factor>
#endif
  struct replicate_view : collection_view_interface<replicate_view<T, Factor>>
  {
#ifdef __cpp_concepts
    constexpr replicate_view() requires std::default_initializable<T> and std::default_initializable<Factor> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT> and std::is_default_constructible_v<Factor>, int> = 0>
    constexpr replicate_view() {}
#endif


#ifdef __cpp_concepts
    template<typename Arg, value::index F> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, typename F, std::enable_if_t<value::index<F> and
      std::is_constructible_v<T, Arg&&> and std::is_constructible_v<Factor, F&&>, int> = 0>
#endif
    explicit constexpr replicate_view(Arg&& arg, F&& f) : t {std::forward<Arg>(arg)}, factor {std::forward<F>(f)} {}


#ifdef __cpp_explicit_this_parameter
    template<value::index I> requires (value::fixed<I> or sized_random_access_range<T>) and
      (value::dynamic<I> or value::dynamic<Factor> or size_of_v<T> == dynamic_size or
        value::fixed_number_of_v<I> < size_of_v<T> * value::fixed_number_of_v<Factor>)
    constexpr decltype(auto)
    operator[](this auto&& self, I i)
    {
      return collections::get(std::forward<decltype(self)>(self).t,
        value::operation {std::modulus{}, std::move(i), get_collection_size(self.t)});
    }
#else
    template<typename I, std::enable_if_t<value::index<I> and
      ((value::fixed<I> and value::fixed<Factor>) or sized_random_access_range<T>), int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr(value::fixed<I> and value::fixed<Factor> and size_of_v<T> != dynamic_size)
        static_assert(value::fixed_number_of_v<I> < size_of_v<T> * value::fixed_number_of_v<Factor>);
      return get(t, value::operation {std::modulus{}, std::move(i), get_collection_size(t)});
    }


    template<typename I, std::enable_if_t<value::index<I> and
      ((value::fixed<I> and value::fixed<Factor>) or sized_random_access_range<T>), int> = 0>
    constexpr decltype(auto)
    operator[](I i) &&
    {
      if constexpr(value::fixed<I> and value::fixed<Factor> and size_of_v<T> != dynamic_size)
        static_assert(value::fixed_number_of_v<I> < size_of_v<T> * value::fixed_number_of_v<Factor>);
      return get(std::move(*this).t, value::operation {std::modulus{}, std::move(i), get_collection_size(t)});
    }
#endif


#ifdef __cpp_concepts
    constexpr value::index auto size() const
#else
    constexpr auto size() const
#endif
    {
      return value::operation {std::multiplies{}, get_collection_size(t), factor};
    }

  private:

    T t;
    Factor factor;
  };


  /**
   * \brief Deduction guide
   */
  template<typename Arg, typename F>
  replicate_view(Arg&&, F&&) -> replicate_view<Arg, F>;

} // namespace OpenKalman::collections


namespace std
{
  template<typename T, typename F>
  struct tuple_size<OpenKalman::collections::replicate_view<T, F>>
    : OpenKalman::value::operation<multiplies<size_t>, OpenKalman::collections::internal::tuple_size_base<T>, F> {};

  template<size_t i, typename T, typename F>
  struct tuple_element<i, OpenKalman::collections::replicate_view<T, F>>
    : OpenKalman::collections::internal::tuple_element_base<
        value::operation<std::modulus<std::size_t>, integral_constant<std::size_t, i>, tuple_size<std::decay_t<T>>>, T> {};
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct replicate_impl
    {
#ifdef __cpp_concepts
      template<collection R, value::index F>
#else
      template<typename R, typename F, std::enable_if_t<collection<R> and value::index<F>, int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (R&& r, F&& f) const { return replicate_view {std::forward<R>(r), std::forward<F>(f)}; }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref replicate_view.
   * \details The expression <code>views::replicate(arg)</code> is expression-equivalent
   * to <code>replicate_view(arg)</code> for any suitable \ref collection arg.
   * \sa replicate_view
   */
  inline constexpr detail::replicate_impl replicate;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_REPLICATE_HPP
