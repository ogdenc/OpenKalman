/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for whether \ref interface::number_traits are defined.
 */

#ifndef OPENKALMAN_NUMBER_TRAITS_DEFINED_HPP
#define OPENKALMAN_NUMBER_TRAITS_DEFINED_HPP

#include "number_traits.hpp"

namespace OpenKalman::interface
{
#ifdef __cpp_concepts
  template<typename Arg>
  concept real_defined_for = requires(Arg arg) { number_traits<std::decay_t<Arg>>::real(std::forward<Arg>(arg)); };
#else
  namespace detail
  {
    template<typename Arg, typename = void>
    struct real_defined_for_impl: std::false_type {};

    template<typename Arg>
    struct real_defined_for_impl<Arg, std::void_t<decltype(number_traits<std::decay_t<Arg>>::real(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename Arg>
  constexpr bool real_defined_for = detail::real_defined_for_impl<Arg>::value;
#endif


#ifdef __cpp_concepts
  template<typename Arg>
  concept imag_defined_for = requires(Arg arg) { number_traits<std::decay_t<Arg>>::imag(std::forward<Arg>(arg)); };
#else
  namespace detail
  {
    template<typename Arg, typename = void>
    struct imag_defined_for_impl: std::false_type {};

    template<typename Arg>
    struct imag_defined_for_impl<Arg, std::void_t<decltype(number_traits<std::decay_t<Arg>>::imag(std::declval<Arg>()))>>
      : std::true_type {};
  }

  template<typename Arg>
  constexpr bool imag_defined_for = detail::imag_defined_for_impl<Arg>::value;
#endif


} // namespace OpenKalman::interface

#endif //OPENKALMAN_NUMBER_TRAITS_DEFINED_HPP
