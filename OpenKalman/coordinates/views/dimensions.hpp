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
 * \brief Definition of \ref coordinates::views::dimensions.
 */

#ifndef OPENKALMAN_COORDINATES_VIEWS_DIMENSIONS_HPP
#define OPENKALMAN_COORDINATES_VIEWS_DIMENSIONS_HPP

#include "collections/collections.hpp"
#include "coordinates/concepts/pattern_collection.hpp"
#include "coordinates/functions/get_dimension.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief A view to the dimensions of a \ref uniformly_gettable \ref pattern_collection
   */
  template<typename T>
  struct dimensions_tuple_view
  {
    static_assert(collections::uniformly_gettable<T>);


#ifdef __cpp_lib_concepts
    constexpr dimensions_tuple_view() requires std::default_initializable<T> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<T>, int> = 0>
    constexpr dimensions_tuple_view() {};
#endif


#ifdef __cpp_lib_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<stdex::constructible_from<T, Arg&&>, int> = 0>
#endif
    explicit constexpr dimensions_tuple_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


    /**
     * \brief Get element i of a \ref dimensions_tuple_view
     */
    template<std::size_t i>
    friend constexpr decltype(auto)
    get(const dimensions_tuple_view& v)
    {
      if constexpr (i < collections::size_of_v<T>)
        return get_dimension(collections::get<i>(v.t));
      else
        return std::integral_constant<std::size_t, 1>{};
    }


    /**
    * \brief Get element i of a \ref dimensions_tuple_view
    */
    template<size_t i>
    friend constexpr decltype(auto)
    get(dimensions_tuple_view&& v)
    {
      if constexpr (i < collections::size_of_v<T>)
        return get_dimension(collections::get<i>(std::move(v).t));
      else
        return std::integral_constant<std::size_t, 1>{};
    }

  private:

    T t;
  };

}


namespace std
{
  template<typename T>
  struct tuple_size<OpenKalman::coordinates::dimensions_tuple_view<T>>
    : OpenKalman::collections::size_of<T> {};


  template<std::size_t i, typename T>
  struct tuple_element<i, OpenKalman::coordinates::dimensions_tuple_view<T>>
    : OpenKalman::coordinates::dimension_of<OpenKalman::collections::collection_element_t<i, T>> {};
}


namespace OpenKalman::coordinates::views
{
  namespace detail
  {
    struct dimensions_adaptor
    {
  #ifdef __cpp_concepts
      template<pattern_collection P> requires collections::viewable_collection<P> or collections::uniformly_gettable<P>
  #else
      template<typename P, std::enable_if_t<pattern_collection<P> and
        (collections::viewable_collection<P> or collections::uniformly_gettable<P>), int> = 0>
  #endif
      constexpr auto
      operator() (P&& p) const
      {
        if constexpr (collections::viewable_collection<P>)
        {
          auto f = [pt = std::make_tuple(collections::views::all(std::forward<P>(p)))](auto&& i)
          {
            if constexpr (values::size_compares_with<decltype(i), collections::size_of<P>, &stdex::is_gteq>)
            {
              return std::integral_constant<std::size_t, 1>{};
            }
            else if constexpr (values::size_compares_with<decltype(i), collections::size_of<P>, &stdex::is_lt>)
            {
              return get_dimension(collections::get_element(std::get<0>(pt), std::forward<decltype(i)>(i)));
            }
            else if (i < collections::get_size(std::get<0>(pt)))
            {
              return static_cast<std::size_t>(get_dimension(collections::get_element(std::get<0>(pt), std::forward<decltype(i)>(i))));
            }
            else
            {
              return 1_uz;
            }
          };
          return collections::views::generate(std::move(f));
        }
        else
        {
          return dimensions_tuple_view<P>{std::forward<P>(p)} | collections::views::all;
        }
      }
    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure for the dimensions of a \ref pattern_collection.
   */
  inline constexpr detail::dimensions_adaptor dimensions;

}


#endif
