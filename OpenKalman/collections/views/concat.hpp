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
 * \brief Definition of \ref collections::concat_view and \ref collections::views::concat.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_CONCAT_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_CONCAT_HPP

#include <tuple>
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/classes/operation.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collection_view_interface.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A view that concatenates some number of \ref collection "collections".
   * \details
   * The following should compile:
   * \code
   * static_assert(std::tuple_size_v<concat_view<std::tuple<>, std::tuple<>>> == 0);
   * static_assert(std::tuple_size_v<concat_view<std::tuple<double, int>, std::tuple<unsigned, std::tuple<>, float>>> == 5);
   * static_assert(std::is_same_v<std::tuple_element_t<1, concat_view<std::tuple<double, int>, std::tuple<unsigned, std::tuple<>, float>>>, int>);
   * static_assert(std::is_same_v<std::tuple_element_t<3, concat_view<std::tuple<double, int>, std::tuple<unsigned, std::tuple<>, float>>>, std::tuple<>>);
   * static_assert(collections::get<2>(concat_view {std::tuple{4, 5.}, std::tuple{6.f, std::tuple{}}}) == 6.f);
   * static_assert(collections::get<3>(concat_view {std::tuple{4, 5.}, std::tuple{6.f, std::tuple{}}}) == std::tuple{});
   * static_assert((concat_view {std::vector{3, 4, 5}, std::vector{6, 7, 8}}[2u]), 5);
   * static_assert((concat_view {std::vector{3, 4, 5}, std::vector{6, 7, 8}}[std::integral_constant<std::size_t, 3>{}]), 6);
   * \endcode
   * \sa views::concat
   */
#ifdef __cpp_lib_ranges
  template<collection...Ts>
#else
  template<typename...Ts>
#endif
  struct concat_view : collection_view_interface<concat_view<Ts...>>
  {
  private:

    template<std::size_t c = 0, std::size_t local = 0, typename Tup, std::size_t...cs>
    static constexpr auto
    make_index_table_tup(const Tup& tup, std::index_sequence<cs...> seq = std::index_sequence<>{})
    {
      if constexpr (c < std::tuple_size_v<Tup>)
      {
        if constexpr (local < std::tuple_size_v<std::decay_t<std::tuple_element_t<c, Tup>>>)
          return make_index_table_tup<c, local + 1>(tup, std::index_sequence<cs..., c>{});
        else
          return make_index_table_tup<c + 1, 0>(tup, seq);
      }
      else return std::tuple {std::integral_constant<std::size_t, cs>{}...};
    }


    template<typename Tup, std::size_t...cs>
#ifdef __cpp_lib_constexpr_vector
    static constexpr auto
#else
    static auto
#endif
    make_index_table_range(const Tup& tup, std::index_sequence<cs...>)
    {
      std::vector<std::size_t> table;
      auto it = std::back_inserter(table);
      (std::fill_n(it, get_collection_size(std::get<cs>(tup)), cs), ...);
      return table;
    }


    template<typename Tup>
    static constexpr auto
    make_index_table(const Tup& tup)
    {
      if constexpr ((... and tuple_like<Ts>)) return make_index_table_tup(tup);
      else return make_index_table_range(tup, std::make_index_sequence<std::tuple_size_v<Tup>>{});
    }


    template<std::size_t c = 0, typename Tup, typename CurrLoc = std::integral_constant<std::size_t, 0>, typename...Locs>
    static constexpr auto
    make_start_table(const Tup& tup, CurrLoc currloc = {}, Locs...locs)
    {
      if constexpr (c < std::tuple_size_v<Tup>)
      {
        auto next_loc = value::operation {std::plus{}, currloc, get_collection_size(std::get<c>(tup))};
        return make_start_table<c + 1>(tup, std::move(next_loc), std::move(locs)..., std::move(currloc));
      }
      else return std::tuple {std::move(locs)...};
    }

  public:

#ifdef __cpp_concepts
    constexpr concat_view() requires std::default_initializable<std::tuple<Ts...>> = default;
#else
    template<typename aT = void, std::enable_if_t<std::is_void_v<aT> and std::is_default_constructible_v<std::tuple<Ts...>>, int> = 0>
    constexpr concat_view() {}
#endif


#ifdef __cpp_concepts
    template<typename...Args> requires std::constructible_from<std::tuple<Ts...>, Args&&...>
#else
    template<typename...Args, std::enable_if_t<std::is_constructible_v<std::tuple<Ts...>, Args&&...>, int> = 0>
#endif
    explicit constexpr concat_view(Args&&...args) : my_ts {std::forward<Args>(args)...} {}


#ifdef __cpp_explicit_this_parameter
    template<value::index I> requires (value::fixed<I> or (... and sized_random_access_range<Ts>)) and
      (value::dynamic<I> or ((size_of_v<Ts> == dynamic_size) or ... or
        (value::fixed_number_of_v<I> < (0_uz + ... + size_of_v<Ts>))))
    constexpr decltype(auto)
    operator[](this auto&& self, I i)
    {
      auto c = get(std::forward<decltype(self)>(self).index_table, i);
      auto local_i = value::operation{std::minus{}, std::move(i), get(std::forward<decltype(self)>(self).start_table, c)};
      return get(get(std::forward<decltype(self)>(self).my_ts, std::move(c)), std::move(local_i));
    }
#else
    template<typename I, std::enable_if_t<value::index<I> and (value::fixed<I> or (... and sized_random_access_range<Ts>)), int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr(value::fixed<I> and (... and (size_of_v<Ts> != dynamic_size)))
        static_assert(value::fixed_number_of_v<I> < (0_uz + ... + size_of_v<Ts>));
      auto c = get(index_table, i);
      auto local_i = value::operation{std::minus{}, std::move(i), get(start_table, c)};
      return get(get(my_ts, std::move(c)), std::move(local_i));
    }


    template<typename I, std::enable_if_t<value::index<I> and (value::fixed<I> or (... and sized_random_access_range<Ts>)), int> = 0>
    constexpr decltype(auto)
    operator[](I i) &&
    {
      if constexpr(value::fixed<I> and (... and (size_of_v<Ts> != dynamic_size)))
        static_assert(value::fixed_number_of_v<I> < (0_uz + ... + size_of_v<Ts>));
      auto c = get(std::move(*this).index_table, i);
      auto local_i = value::operation{std::minus{}, std::move(i), get(std::move(*this).start_table, c)};
      return get(get(std::move(*this).my_ts, std::move(c)), std::move(local_i));
    }
#endif

  private:

    struct size_impl
    {
      constexpr auto operator()() const { return std::integral_constant<std::size_t, 0> {}; }

      template<typename U, typename...Us>
      constexpr auto operator()(const U& u, const Us&... us) const
      {
        return value::operation {std::plus{}, get_collection_size(u), operator()(us...)};
      }
    };

  public:

#ifdef __cpp_concepts
    constexpr value::index auto size() const
#else
    constexpr auto size() const
#endif
    {
      return std::apply(size_impl{}, my_ts);
    }

  private:

    std::tuple<Ts...> my_ts;

    decltype(make_index_table(my_ts)) index_table = make_index_table(my_ts);

    decltype(make_start_table(my_ts)) start_table = make_start_table(my_ts);

  };


  /**
   * \brief Deduction guide
   */
  template<typename...Args>
  concat_view(Args&&...) -> concat_view<Args...>;

} // namespace OpenKalman::collections


namespace std
{
  template<typename...Ts>
  struct tuple_size<OpenKalman::collections::concat_view<Ts...>>
    : OpenKalman::value::fixed_number_of<decltype(std::declval<OpenKalman::collections::concat_view<Ts...>>().size())> {};

  template<size_t i, typename...Ts>
  struct tuple_element<i, OpenKalman::collections::concat_view<Ts...>>
  {
    using type = decltype(std::declval<OpenKalman::collections::concat_view<Ts...>>()[std::integral_constant<std::size_t, i>{}]);
  };

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct concat_impl
    {
#ifdef __cpp_concepts
      template<collection...R>
#else
      template<typename...R, std::enable_if_t<(... and collection<R>), int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (R&&...r) const { return concat_view<R...> {std::forward<R>(r)...}; }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref concat_view.
   * \details The expression <code>views::concat(arg)</code> is expression-equivalent
   * to <code>concat_view(arg)</code> for any suitable \ref collection arg.
   * \sa concat_view
   */
  inline constexpr detail::concat_impl concat;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_CONCAT_HPP
