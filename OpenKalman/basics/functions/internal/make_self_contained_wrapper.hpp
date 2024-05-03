/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref make_self_contained_wrapper function.
 */

#ifndef OPENKALMAN_MAKE_SELF_CONTAINED_WRAPPER_HPP
#define OPENKALMAN_MAKE_SELF_CONTAINED_WRAPPER_HPP

namespace OpenKalman::internal
{

  namespace detail
  {
    template<typename S_tup, typename P_tup>
    struct sc_parameters_match : std::true_type {};

    template<typename S, typename...Ss, typename P, typename...Ps>
    struct sc_parameters_match<std::tuple<S, Ss...>, std::tuple<P, Ps...>> : std::bool_constant<
      sizeof...(Ss) <= sizeof...(Ps) and
      (std::is_constructible_v<std::decay_t<S>, std::add_lvalue_reference_t<P>> or sc_parameters_match<std::tuple<S, Ss...>, std::tuple<Ps...>>{}) and
      (not std::is_constructible_v<std::decay_t<S>, std::add_lvalue_reference_t<P>> or sc_parameters_match<std::tuple<Ss...>, std::tuple<Ps...>>{})
    > {};


    template<typename T, typename S_tup, typename P_tup, std::size_t...Px>
    constexpr bool
    need_self_contained(std::index_sequence<>, std::index_sequence<Px...>) { return false; }


    template<typename T, typename S_tup, typename P_tup, std::size_t Sx0, std::size_t...Sx, std::size_t Px0, std::size_t...Px>
    constexpr bool
    need_self_contained(std::index_sequence<Sx0, Sx...>, std::index_sequence<Px0, Px...>)
    {
      using S = std::tuple_element_t<Sx0, S_tup>;
      using P = std::tuple_element_t<Px0, P_tup>;
      if constexpr (std::is_constructible_v<std::decay_t<S>, std::add_lvalue_reference_t<P>>)
      {
        return (std::is_lvalue_reference_v<S> and not std::is_lvalue_reference_v<P>) or
          need_self_contained<T, S_tup, P_tup>(std::index_sequence<Sx...>{}, std::index_sequence<Px...>{});
      }
      else
      {
        return need_self_contained<T, S_tup, P_tup>(std::index_sequence<Sx0, Sx...>{}, std::index_sequence<Px...>{});
      }
    }
  } // namespace detail


  /**
   * \brief Make a self-contained, wrapped object of type T, using constructor arguments Ps...
   * \details Creates a \ref SelfContainedWrapper
   * \tparam T The native library type to be made self-contained.
   * \tparam Ss The parameter types as stored internally in T (optional).
   * If provided, every Ss must correspond to a Ps type from which it may be constructed.
   * The Ps types must be in the same order as Ss types, although there may be additional Ps types interspersed between
   * them, which will be skipped. This information allows this function to avoid constructing a \ref SelfContainedWrapper
   * unless necessary to avoid dangling references.
   * If no Ss types are provided, this function will construct a \ref SelfContainedWrapper in all cases.
   * \tparam Ps A set of parameters needed to construct T.
   */
#ifdef __cpp_concepts
  template<indexible T, typename...Ss, typename...Ps> requires std::constructible_from<T, std::add_lvalue_reference_t<Ps>...> and
    detail::sc_parameters_match<std::tuple<Ss...>, std::tuple<Ps...>>::value
#else
  template<typename T, typename...Ss, typename...Ps, std::enable_if_t<indexible<T> and
    std::is_constructible_v<T, std::add_lvalue_reference_t<Ps>...> and detail::sc_parameters_match<std::tuple<Ss...>, std::tuple<Ps...>>::value, int> = 0>
#endif
  inline auto
  make_self_contained_wrapper(Ps&&...ps)
  {
    constexpr std::index_sequence_for<Ss...> seq_S;
    constexpr std::index_sequence_for<Ps...> seq_P;
    if constexpr (sizeof...(Ss) == 0 or detail::need_self_contained<T, std::tuple<Ss...>, std::tuple<Ps...>>(seq_S, seq_P))
      return SelfContainedWrapper<T, Ps...> {std::forward<Ps>(ps)...};
    else if constexpr (std::is_constructible_v<T, Ps&&...>)
      return T {std::forward<Ps>(ps)...};
    else
      return T {ps...};
  }

} // namespace OpenKalman::internal


#endif //OPENKALMAN_MAKE_SELF_CONTAINED_WRAPPER_HPP
