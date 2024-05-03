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
 * \brief The sum function.
 */

#ifndef OPENKALMAN_SUM_HPP
#define OPENKALMAN_SUM_HPP

#include<complex>


namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t...Ix, typename T0, typename T1>
    constexpr auto sum_constants(std::index_sequence<Ix...> seq, T0&& t0, T1&& t1)
    {
      auto c = constant_coefficient{t0} + constant_coefficient{t1};
      return make_constant<T0>(std::move(c), internal::best_vector_space_descriptor(
        get_vector_space_descriptor<Ix>(t0), get_vector_space_descriptor<Ix>(t1))...);
    }


    template<typename T>
    constexpr T&& sum_impl(T&& t) { return std::forward<T>(t); }


    template<typename T0, typename T1, typename...Ts>
    constexpr decltype(auto) sum_impl(T0&& t0, T1&& t1, Ts&&...ts)
    {
      if constexpr ((zero<T0> or zero<T1> or (constant_matrix<T0> and constant_matrix<T1>)) and not same_shape_as<T0, T1>)
      {
        if (not same_shape(t0, t1)) throw std::invalid_argument {"In sum function, vector space descriptors of arguments do not match"};
      }

      if constexpr (zero<T0>)
      {
        return sum_impl(std::forward<T1>(t1), std::forward<Ts>(ts)...);
      }
      else if constexpr (zero<T1>)
      {
        return sum_impl(std::forward<T0>(t0), std::forward<Ts>(ts)...);
      }
      else if constexpr ((constant_matrix<T0> and constant_matrix<T1>))
      {
        constexpr std::make_index_sequence<std::max({index_count_v<Ts>...})> seq;
        auto ret {sum_impl(seq, sum_constants(seq, std::forward<T0>(t0), std::forward<T1>(t1)), std::forward<Ts>(ts)...)};
        return ret;
      }
      else if constexpr (constant_matrix<T0> and sizeof...(Ts) > 0) // Shift T0 right and hope that it will combine with another constant
      {
        return sum_impl(std::forward<T1>(t1), sum_impl(std::forward<T0>(t0), std::forward<Ts>(ts)...));
      }
      else if constexpr (constant_matrix<T1> and sizeof...(Ts) > 0) // Shift T1 right and hope that it will combine with another constant
      {
        return sum_impl(std::forward<T0>(t0), sum_impl(std::forward<T1>(t1), std::forward<Ts>(ts)...));
      }
      else if constexpr (diagonal_matrix<T0> and diagonal_matrix<T1>)
      {
        auto ret {sum_impl(to_diagonal(sum_impl(diagonal_of(std::forward<T0>(t0)), diagonal_of(std::forward<T1>(t1)))), std::forward<Ts>(ts)...)};
        return ret;
      }
      else if constexpr (interface::sum_defined_for<T0, T0&&, T1&&, Ts&&...>)
      {
        return interface::library_interface<std::decay_t<T0>>::sum(std::forward<T0>(t0), std::forward<T1>(t1), std::forward<Ts>(ts)...);
      }
      else if constexpr (interface::sum_defined_for<T0, T0&&, T1&&>)
      {
        return sum_impl(interface::library_interface<std::decay_t<T0>>::sum(std::forward<T0>(t0), std::forward<T1>(t1)), std::forward<Ts>(ts)...);
      }
      else if constexpr (interface::sum_defined_for<T1, T1&&, T0&&>)
      {
        return sum_impl(interface::library_interface<std::decay_t<T1>>::sum(std::forward<T1>(t1), std::forward<T0>(t0)), std::forward<Ts>(ts)...);
      }
      else if constexpr (interface::sum_defined_for<T0, T0&&, decltype(to_native_matrix<T0>(std::declval<T1&&>()))>)
      {
        return sum_impl(interface::library_interface<std::decay_t<T0>>::sum(std::forward<T0>(t0), to_native_matrix<T0>(std::forward<T1>(t1))), std::forward<Ts>(ts)...);
      }
      else if constexpr (interface::sum_defined_for<T1, T1&&, decltype(to_native_matrix<T1>(std::declval<T0&&>()))>)
      {
        return sum_impl(interface::library_interface<std::decay_t<T1>>::sum(std::forward<T1>(t1), to_native_matrix<T1>(std::forward<T0>(t0))), std::forward<Ts>(ts)...);
      }
      else
      {
        // \todo Add a default loop in case the library interface does not include sums?
      }
    }
  } // namespace detail


  /**
   * \brief Element-by-element sum of one or more objects.
   */
#ifdef __cpp_concepts
  template<indexible...Ts> requires (sizeof...(Ts) > 0) and maybe_same_shape_as<Ts...>
#else
  template<typename...Ts, std::enable_if_t<(indexible<Ts> and ...) and (sizeof...(Ts) > 0) and
    maybe_same_shape_as<Ts...>, int> = 0>
#endif
  constexpr decltype(auto)
  sum(Ts&&...ts)
  {
    auto s {internal::make_fixed_size_adapter_like<Ts...>(detail::sum_impl(std::forward<Ts>(ts)...))};
    constexpr auto t = triangle_type_of_v<Ts...>;
    if constexpr (t != TriangleType::any and not triangular_matrix<decltype(s), t>)
      return make_triangular_matrix<t>(std::move(s));
    else if constexpr ((... and hermitian_matrix<Ts>) and not hermitian_matrix<decltype(s)>)
      return make_hermitian_matrix(std::move(s));
    else
      return s;
  }

} // namespace OpenKalman


#endif //OPENKALMAN_SUM_HPP
