/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for strides function.
 */

#ifndef OPENKALMAN_STRIDES_HPP
#define OPENKALMAN_STRIDES_HPP

#include<vector>

namespace OpenKalman::internal
{
  namespace detail
  {
    template<Layout l, std::size_t count, typename T, typename CurrStride, std::size_t I, std::size_t...Is, typename...Strides>
    constexpr auto strides_impl(const T& t, CurrStride curr_stride, std::index_sequence<I, Is...>, Strides...strides)
    {
      if constexpr (sizeof...(Is) == 0)
      {
        if constexpr (l == Layout::right)
          return std::tuple {curr_stride, strides...};
        else
          return std::tuple {strides..., curr_stride};
      }
      else
      {
        auto curr_dim = get_index_dimension_of<l == Layout::right ? count - 1 - I : I>(t);
        auto next_stride = [](CurrStride curr_stride, auto curr_dim)
        {
          if constexpr (value::static_index<CurrStride, std::ptrdiff_t> and value::static_index<decltype(curr_dim)>)
            return std::integral_constant<std::ptrdiff_t, std::decay_t<CurrStride>::value * decltype(curr_dim)::value>{};
          else
            return static_cast<std::ptrdiff_t>(curr_stride) * static_cast<std::ptrdiff_t>(curr_dim);
        }(curr_stride, curr_dim);

        if constexpr (l == Layout::right)
          return strides_impl<l, count>(t, next_stride, std::index_sequence<Is...>{}, curr_stride, strides...);
        else
          return strides_impl<l, count>(t, next_stride, std::index_sequence<Is...>{}, strides..., curr_stride);
      }
    }


    template<typename T, std::size_t...Is>
    constexpr bool strides_tuple_impl(std::index_sequence<Is...>)
    {
      return (... and (std::is_convertible_v<std::tuple_element_t<Is, T>, std::ptrdiff_t> or
        value::static_index<std::tuple_element_t<Is, T>, std::ptrdiff_t>));
    }

    template<typename T>
#ifdef __cpp_concepts
    concept strides_tuple =
#else
    constexpr bool strides_tuple =
#endif
      strides_tuple_impl<T>(std::make_index_sequence<std::tuple_size_v<T>>{});

  } // namespace detail


  /**
   * \internal
   * \brief Returns a tuple <code>std::tuple&lt;S...&gt;</code> comprising the strides of a strided tensor or matrix.
   * \details Each of the strides <code>S</code> satisfies one of the following:
   * - <code>std::convertible_to&lt;S, std::ptrdiff_t&gt;</code>; or
   * - <code>value::static_index&lt;S, std::ptrdiff_t&gt;</code>.
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T> requires interface::layout_defined_for<T> and
    (interface::indexible_object_traits<std::decay_t<T>>::layout != Layout::stride or interface::strides_defined_for<T>) and
    (interface::indexible_object_traits<std::decay_t<T>>::layout != Layout::none)
  detail::strides_tuple auto
#else
  template<typename T, std::enable_if_t<interface::count_indices_defined_for<T> and interface::layout_defined_for<T> and
    (interface::indexible_object_traits<std::decay_t<T>>::layout != Layout::stride or interface::strides_defined_for<T>) and
    (interface::indexible_object_traits<std::decay_t<T>>::layout != Layout::none), int> = 0>
  auto
#endif
  strides(const T& t)
  {
    constexpr Layout l = interface::indexible_object_traits<std::decay_t<T>>::layout;

    if constexpr (l == Layout::stride)
    {
#ifndef __cpp_concepts
      static_assert(detail::strides_tuple<decltype(interface::indexible_object_traits<std::decay_t<T>>::strides(t))>);
#endif
      return interface::indexible_object_traits<std::decay_t<T>>::strides(t);
    }
    else if constexpr (value::static_index<decltype(count_indices(t))>)
    {
      constexpr std::size_t count = std::decay_t<decltype(count_indices(t))>::value;
      constexpr std::integral_constant<std::ptrdiff_t, 1> N1;
      return detail::strides_impl<l, count>(t, N1, std::make_index_sequence<count>{});
    }
    else
    {
      std::size_t count = count_indices(t);
      std::vector<std::ptrdiff_t> vec(count);
      if constexpr (l == Layout::left)
      {
        auto v = vec.begin();
        std::ptrdiff_t curr_stride = 1;
        for (int i = 0; i < count; ++i)
        {
          *v = curr_stride;
          curr_stride *= get_index_dimension_of(t, i);
          ++v;
        }
      }
      else
      {
        auto v = vec.end();
        std::ptrdiff_t curr_stride = 1;
        for (int i = 1; i <= count; ++i)
        {
          --v;
          *v = curr_stride;
          curr_stride *= get_index_dimension_of(t, count - i);
        }
      }
      return vec;
    }
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_STRIDES_HPP
