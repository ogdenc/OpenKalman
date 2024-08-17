/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref assign function.
 */

#ifndef OPENKALMAN_ASSIGN_HPP
#define OPENKALMAN_ASSIGN_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename M, typename Arg, typename...J>
    static void copy_tensor_elements(M& m, const Arg& arg, std::index_sequence<>, J...j)
    {
      set_component(m, get_component(arg, j...), j...);
    }


    template<typename M, typename Arg, std::size_t I, std::size_t...Is, typename...J>
    static void copy_tensor_elements(M& m, const Arg& arg, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<I>(arg); i++)
        copy_tensor_elements(m, arg, std::index_sequence<Is...> {}, j..., i);
    }
  } // namespace detail


  /**
   * \brief Assign a writable object from an indexible object.
   * \tparam To The writable object to be assigned.
   * \tparam From The indexible object from which to assign.
   * \return the assigned object as modified
   */
#ifdef __cpp_concepts
  template<writable To, indexible From>
#else
  template<typename To, typename From, std::enable_if_t<writable<To> and indexible<From>, int> = 0>
#endif
  constexpr To&&
  assign(To&& a, From&& b)
  {
    if constexpr (interface::assign_defined_for<To, std::add_lvalue_reference_t<To>, From&&>)
    {
      interface::library_interface<std::decay_t<To>>::assign(a, std::forward<From>(b));
    }
    else if constexpr (std::is_assignable_v<std::add_lvalue_reference_t<To>, From&&>)
    {
      a = std::forward<From>(b);
    }
    else if constexpr (std::is_assignable_v<std::add_lvalue_reference_t<To>, decltype(to_native_matrix<To>(std::declval<From&&>()))>)
    {
      a = to_native_matrix<To>(std::forward<From>(b));
    }
    else
    {
      detail::copy_tensor_elements(a, b, std::make_index_sequence<index_count_v<To>>{});
    }
    return std::forward<To>(a);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_ASSIGN_HPP
