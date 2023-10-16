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
 * \brief Definitions for \ref from_euclidean function.
 */

#ifndef OPENKALMAN_FROM_EUCLIDEAN_HPP
#define OPENKALMAN_FROM_EUCLIDEAN_HPP

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Arg, typename C, typename = void>
    struct from_euclidean_exists : std::false_type {};

    template<typename Arg, typename C>
    struct from_euclidean_exists<Arg, C, std::void_t<decltype(
        interface::library_interface<std::decay_t<Arg>>::template from_euclidean(std::declval<Arg&&>(), std::declval<const C&>()))>> :
      std::true_type {};
  } // namespace detail
#endif

#ifdef __cpp_concepts
  template<wrappable Arg, vector_space_descriptor C>
  requires (dynamic_vector_space_descriptor<C> or dynamic_dimension<Arg, 0> or has_untyped_index<Arg, 0> or
    equivalent_to<C, vector_space_descriptor_of_t<Arg, 0>>)
#else
  template<typename Arg, typename C, std::enable_if_t<wrappable<Arg> and vector_space_descriptor<C> and
    (dynamic_vector_space_descriptor<C> or dynamic_dimension<Arg, 0> or has_untyped_index<Arg, 0> or
      equivalent_to<C, vector_space_descriptor_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg, const C& c) noexcept
  {
    if constexpr (dynamic_dimension<Arg, 0> and not euclidean_vector_space_descriptor<vector_space_descriptor_of_t<Arg, 0>>)
      if (not get_vector_space_descriptor_is_euclidean(get_vector_space_descriptor<0>(arg)) and c != get_vector_space_descriptor<0>(arg))
        throw std::domain_error {"In from_euclidean, specified vector space descriptor does not match that of the object's index 0"};
    using Interface = interface::library_interface<std::decay_t<Arg>>;

    if constexpr (euclidean_vector_space_descriptor<C>)
    {
      return std::forward<Arg>(arg);
    }
#ifdef __cpp_concepts
    else if constexpr (requires { Interface::template from_euclidean(std::forward<Arg>(arg), c); })
#else
    else if constexpr (detail::from_euclidean_exists<Arg, C>::value)
#endif
    {
      return Interface::from_euclidean(std::forward<Arg>(arg), c);
    }
    else
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_wrappable(arg))
        throw std::domain_error {"Argument of from_euclidean is not wrappable"};

      return FromEuclideanExpr<C, Arg>(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<all_fixed_indices_are_euclidean Arg>
#else
  template<typename Arg, std::enable_if_t<all_fixed_indices_are_euclidean<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg)
  {
    return from_euclidean(std::forward<Arg>(arg), get_vector_space_descriptor<0>(arg));
  }



} // namespace OpenKalman

#endif //OPENKALMAN_FROM_EUCLIDEAN_HPP
