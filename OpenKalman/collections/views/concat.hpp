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
 * \brief Definition of \ref collections::concat_tuple_view and \ref collections::views::concat.
 */

#ifndef OPENKALMAN_VIEWS_CONCAT_HPP
#define OPENKALMAN_VIEWS_CONCAT_HPP

#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<std::size_t i>
    constexpr auto concat_tuple_view_indices()
    {
      return std::pair {std::integral_constant<std::size_t, 0_uz>{}, std::integral_constant<std::size_t, 0_uz>{}};
    }

    template<std::size_t i, typename T, typename...Ts>
    constexpr auto concat_tuple_view_indices()
    {
      constexpr std::size_t size = std::tuple_size_v<std::decay_t<T>>;
      if constexpr (i < size) return std::pair {std::integral_constant<std::size_t, 0_uz>{}, std::integral_constant<std::size_t, i>{}};
      else
      {
        auto [a, b] = concat_tuple_view_indices<i - size, Ts...>();
        return std::pair {std::integral_constant<std::size_t, 1_uz + decltype(a)::value>{}, b};
      }
    }
  };


  /**
   * \brief A view to a concatenation of some number of other \ref tuple_like object
   * \details This is similar to std::tuple_cat, but it allows for concatenation of any \ref tuple_like object.
   * \tparam Ts A set of base \ref tuple_like objects
   */
#ifdef __cpp_concepts
  template<tuple_like...Ts>
#else
  template<typename...Ts>
#endif
  struct concat_tuple_view
  {
#ifdef __cpp_concepts
    constexpr concat_tuple_view() requires (... and std::default_initializable<Ts>) = default;
#else
    template<typename Tup = std::tuple<Ts...>, std::enable_if_t<std::is_default_constructible_v<Tup>, int> = 0>
    constexpr concat_tuple_view() {};
#endif


#ifdef __cpp_concepts
    template<typename...Args> requires (... and std::constructible_from<Ts, Args&&>)
#else
    template<typename...Args, std::enable_if_t<(... and std::is_constructible_v<Ts, Args&&>), int> = 0>
#endif
    explicit constexpr concat_tuple_view(Args&&...args) : tup {std::forward<Args>(args)...} {}


    /**
     * \brief Get element i of a \ref concat_tuple_view
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>))
#else
    template<std::size_t i, std::enable_if_t<(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>)), int> = 0>
#endif
    friend constexpr decltype(auto)
    get(const concat_tuple_view& v)
    {
      auto [element, index] = std::decay_t<decltype(detail::concat_tuple_view_indices<i, Ts...>())>();
      return get(get(v.tup, element), index);
    }


    /**
     * \overload
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>))
#else
    template<std::size_t i, std::enable_if_t<(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>)), int> = 0>
#endif
    friend constexpr decltype(auto)
    get(concat_tuple_view&& v)
    {
      auto [element, index] = std::decay_t<decltype(detail::concat_tuple_view_indices<i, Ts...>())>();
      return get(get(std::move(v).tup, element), index);
    }

  private:

    std::tuple<Ts...> tup;
  };


  /**
   * \brief Deduction guide
   */
  template<typename...Args>
  concat_tuple_view(Args&&...) -> concat_tuple_view<Args...>;

}


namespace std
{
  template<typename...Ts>
  struct tuple_size<OpenKalman::collections::concat_tuple_view<Ts...>>
    : std::integral_constant<std::size_t, (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>)> {};


  template<std::size_t i, typename...Ts>
  struct tuple_element<i, OpenKalman::collections::concat_tuple_view<Ts...>>
  {
    static_assert(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>));
    using indices = decltype(OpenKalman::collections::detail::concat_tuple_view_indices<i, Ts...>());
    using element = std::tuple_element_t<0, indices>;
    using index = std::tuple_element_t<1, indices>;
    using type = std::tuple_element_t<index::value, std::decay_t<std::tuple_element_t<element::value, std::tuple<Ts...>>>>;
  };
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct concat_adaptor
    {
  #ifdef __cpp_concepts
      template<viewable_collection...R> requires (sizeof...(R) > 0)
  #else
      template<typename...R, std::enable_if_t<(sizeof...(R) > 0) and (... and viewable_collection<R>), int> = 0>
  #endif
      constexpr auto
      operator() (R&&...r) const
      {
#if __cpp_lib_ranges_concat >= 202403L
        namespace cv = std::ranges::views;
#else
        namespace cv = ranges::views;
#endif
        if constexpr (sizeof...(R) == 1)
          return all(std::forward<R>(r)...);
        else if constexpr ((... and tuple_like<R>))
          return concat_tuple_view {all(std::forward<R>(r))...} | all;
        else
          return cv::concat(all(std::forward<R>(r))...) | all;
      }

    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure for a set of concatenated \ref collection objects.
   */
  inline constexpr detail::concat_adaptor concat;

}


#endif //OPENKALMAN_VIEWS_CONCAT_HPP
