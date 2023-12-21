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
    template<std::size_t...I, typename T>
    constexpr decltype(auto) sum_impl(std::index_sequence<I...>, T&& t) { return std::forward<T>(t); }


    template<std::size_t...I, typename T0, typename T1, typename...Ts>
    constexpr decltype(auto) sum_impl(std::index_sequence<I...> seq, T0&& t0, T1&& t1, Ts&&...ts)
    {
      if constexpr ((zero<T0> or zero<T1> or (constant_matrix<T0> and constant_matrix<T1>)) and not same_shape_as<T0, T1>)
      {
        if (not same_shape(t0, t1))
          throw std::invalid_argument {"In sum function, vector space descriptors of arguments do not match"};
      }

      if constexpr (zero<T0>)
      {
        return sum_impl(seq, std::forward<T1>(t1), std::forward<Ts>(ts)...);
      }
      else if constexpr (zero<T1>)
      {
        return sum_impl(seq, std::forward<T0>(t0), std::forward<Ts>(ts)...);
      }
      else if constexpr ((constant_matrix<T0> and constant_matrix<T1>))
      {
        auto c = constant_coefficient{t0} + constant_coefficient{t1};
        auto cm = make_constant<T0>(std::move(c), internal::best_vector_space_descriptor(
          get_vector_space_descriptor<I>(t0), get_vector_space_descriptor<I>(t1))...);
        auto ret {sum_impl(seq, std::move(cm), std::forward<Ts>(ts)...)};
        return ret;
      }
      else if constexpr (constant_matrix<T0> and sizeof...(Ts) > 0) // Shift T0 right in hopes that it will combine with another constant
      {
        return sum_impl(seq, std::forward<T1>(t1), sum_impl(seq, std::forward<T0>(t0), std::forward<Ts>(ts)...));
      }
      else if constexpr (constant_matrix<T1> and sizeof...(Ts) > 0) // Shift T1 right in hopes that it will combine with another constant
      {
        return sum_impl(seq, std::forward<T0>(t0), sum_impl(seq, std::forward<T1>(t1), std::forward<Ts>(ts)...));
      }
      else if constexpr (diagonal_matrix<T0> and diagonal_matrix<T1>)
      {
        auto ret {sum_impl(seq, to_diagonal(sum_impl(seq, diagonal_of(std::forward<T0>(t0)), diagonal_of(std::forward<T1>(t1)))), std::forward<Ts>(ts)...)};
        return ret;
      }
      else if constexpr (interface::sum_defined_for<std::decay_t<T0>, T0&&, T1&&, Ts&&...>)
      {
        // \todo Do further pruning
        auto ret {interface::library_interface<std::decay_t<T0>>::sum(std::forward<T0>(t0), std::forward<T1>(t1), std::forward<Ts>(ts)...)};
        return ret;
      }
      else if constexpr (interface::sum_defined_for<std::decay_t<T0>, T0&&, T1&&>)
      {
        auto ret {sum_impl(seq, interface::library_interface<std::decay_t<T0>>::sum(std::forward<T0>(t0), std::forward<T1>(t1)), std::forward<Ts>(ts)...)};
        return ret;
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
    constexpr std::make_index_sequence<std::max({index_count_v<Ts>...})> seq;

    if constexpr (sizeof...(Ts) == 1)
    {
      return (std::forward<Ts>(ts),...);
    }
    else if constexpr ((... and constant_matrix<Ts>))
    {
      auto ret {detail::sum_impl(seq, std::forward<Ts>(ts)...)};
      return ret;
    }
    else if constexpr ((... and diagonal_matrix<Ts>))
    {
      auto ret {to_diagonal(detail::sum_impl(seq, diagonal_of(std::forward<Ts>(ts))...))};
      return ret;
    }
    else if constexpr (triangle_type_of_v<Ts...> != TriangleType::any)
    {
      constexpr auto t = triangle_type_of_v<Ts...>;
      auto f = [](auto&& a) -> decltype(auto) {
        if constexpr (triangular_adapter<decltype(a)>) return nested_object(std::forward<decltype(a)>(a));
        else return std::forward<decltype(a)>(a);
      };
      auto ret {make_triangular_matrix<t>(detail::sum_impl(seq, f(std::forward<Ts>(ts))...))};
      return ret;
    }
    else if constexpr ((... and hermitian_matrix<Ts>))
    {
      constexpr auto t = hermitian_adapter_type_of_v<Ts...> == HermitianAdapterType::any ?
        HermitianAdapterType::lower : hermitian_adapter_type_of_v<Ts...>;
      auto f = [](auto&& a) -> decltype(auto) {
        static_assert(hermitian_adapter_type_of_v<decltype(a)> != HermitianAdapterType::any);
        if constexpr (hermitian_adapter_type_of_v<decltype(a)> == t) return nested_object(std::forward<decltype(a)>(a));
        else if constexpr (hermitian_adapter<decltype(a)>) return transpose(nested_object(std::forward<decltype(a)>(a)));
        else return std::forward<decltype(a)>(a);
      };
      auto ret {make_hermitian_matrix<t>(detail::sum_impl(seq, f(std::forward<Ts>(ts))...))};
      return ret;
    }
    else
    {
      auto ret {internal::make_fixed_size_adapter_like<Ts...>(detail::sum_impl(seq, std::forward<Ts>(ts)...))};
      return ret;
    }
  }

} // namespace OpenKalman


#endif //OPENKALMAN_SUM_HPP
