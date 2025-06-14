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
 * \brief Definition of \ref collections::repeat_tuple_view and \ref collections::views::repeat.
 */

#ifndef OPENKALMAN_VIEWS_REPEAT_HPP
#define OPENKALMAN_VIEWS_REPEAT_HPP

#include "values/concepts/size.hpp"
#include "values/concepts/fixed.hpp"
#include "all.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view of a tuple that replicates a particular value N number of times
   * \tparam N The number of copies
   * \tparam T The type of the object to be replicated
   */
  template<std::size_t N, typename T>
  struct repeat_tuple_view
  {
#ifdef __cpp_lib_concepts
    constexpr repeat_tuple_view() requires std::default_initializable<T> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr repeat_tuple_view() {};
#endif


#ifdef __cpp_lib_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<T, Arg&&>, int> = 0>
#endif
    explicit constexpr repeat_tuple_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


    /**
     * \return The underlying replicated value
     */
    constexpr T value() const { return t; }


    /**
     * \brief Get element i of a \ref repeat_tuple_view
     */
  #ifdef __cpp_concepts
    template<std::size_t i> requires (i < N)
  #else
    template<std::size_t i, std::enable_if_t<i < N, int> = 0>
  #endif
    friend constexpr T
    get(const repeat_tuple_view& v)
    {
      return v.t;
    }


    /**
    * \brief Get element i of a \ref repeat_tuple_view
    */
  #ifdef __cpp_concepts
    template<size_t i> requires (i < N)
  #else
    template<size_t i, std::enable_if_t<i < N, int> = 0>
  #endif
    friend constexpr T
    get(repeat_tuple_view&& v)
    {
      return std::move(v).t;
    }

  private:

    T t;
  };

}


namespace std
{
  template<std::size_t N, typename T>
  struct tuple_size<OpenKalman::collections::repeat_tuple_view<N, T>> : std::integral_constant<size_t, N> {};


  template<std::size_t i, std::size_t N, typename T>
  struct tuple_element<i, OpenKalman::collections::repeat_tuple_view<N, T>>
  {
    static_assert(i < N);
    using type = T;
  };
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct repeat_adaptor
    {
#ifdef __cpp_lib_ranges
      template<std::move_constructible W, values::size Bound = std::unreachable_sentinel_t> requires
        std::is_object_v<W> and std::same_as<W, std::remove_cv_t<W>>
#else
      template<typename W, typename Bound = unreachable_sentinel_t, typename = void>
#endif
      constexpr auto
      operator() [[nodiscard]] (W&& value, Bound&& bound = {}) const
      {
        if constexpr (values::fixed<Bound>)
        {
  #ifdef __cpp_lib_ranges
          namespace cv = std::ranges::views;
  #else
          namespace cv = ranges::views;
  #endif
          if constexpr (values::fixed_number_of_v<Bound> == 1) return cv::single(std::forward<W>(value)) | all;
          else return repeat_tuple_view<values::fixed_number_of_v<Bound>, W> {std::forward<W>(value)} | all;
        }
        else
        {
#ifdef __cpp_lib_ranges_repeat
          namespace cv = std::ranges::views;
#else
          namespace cv = ranges::views;
#endif
          return cv::repeat(std::forward<W>(value), std::forward<Bound>(bound)) | all;
        }
      }
    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure for a set of repeatenated \ref collection objects.
   */
  inline constexpr detail::repeat_adaptor repeat;

}


#endif 
